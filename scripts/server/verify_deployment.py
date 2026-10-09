"""Wait for verified weights, validate deployment, then release smoke-test GPU processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import requests

root = Path('/root/autodl-tmp/opd-sql-agent')
results = root/'results/server'
report = results/'deployment-verification.json'
state = {'status': 'running', 'phase': 'waiting_student', 'training_started': False}


def save(phase):
    state['phase'] = phase
    report.write_text(json.dumps(state, indent=2))
    print(phase, flush=True)


def wait_model(name):
    while True:
        try:
            download = json.loads((results/'model-download.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(10)
            continue
        if download['status'] == 'failed':
            raise RuntimeError(f'Model download failed: {download.get("error")}')
        if any(Path(item['path']).name == name and item['status'] == 'complete'
               for item in download['models']):
            return
        time.sleep(10)


def check_result(name, key):
    result = json.loads((results/name).read_text())
    if result.get(key) not in (True, 'pass'):
        raise RuntimeError(f'{name} did not pass')
    return result


server = None
try:
    save('waiting_student')
    check_result('runtime-verification.json', 'passed')
    wait_model('Qwen3.5-9B')
    save('starting_vllm')
    session = requests.Session()
    session.trust_env = False
    try:
        session.get('http://127.0.0.1:8001/health', timeout=2)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError('Port 8001 already has a service; refusing to adopt or stop an unrelated process')
    with (results/'vllm-server.log').open('w') as log:
        server = subprocess.Popen(['bash', str(root/'scripts/server/start_rollout.sh')],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (results/'vllm-smoke.pid').write_text(str(server.pid))
    started = time.monotonic()
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'vLLM exited during startup: {server.returncode}')
        try:
            response = session.get('http://127.0.0.1:8001/health', timeout=3)
            if response.ok:
                break
        except requests.RequestException:
            pass
        if time.monotonic()-started > 900:
            raise TimeoutError('vLLM did not become healthy within 15 minutes')
        time.sleep(5)
    state['vllm_startup_seconds'] = round(time.monotonic()-started, 2)
    save('verifying_vllm_weight_sync')
    sender_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    with (results/'vllm-sync.log').open('w') as log:
        subprocess.run([sys.executable, str(root/'scripts/server/verify_vllm_sync.py'),
                        '--output', str(results/'vllm-sync-verification.json')],
                       env=sender_env, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    check_result('vllm-sync-verification.json', 'status')
    state['vllm_sync'] = 'pass'
    save('stopping_smoke_rollout')
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=20)
    server = None
    save('waiting_teacher')
    wait_model('Qwen3.8-27B')
    save('verifying_teacher_student_residency')
    with (results/'teacher-verification.log').open('w') as log:
        subprocess.run([sys.executable, str(root/'scripts/server/verify_teacher.py'),
                        '--output', str(results/'teacher-verification.json')],
                       env=sender_env, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    check_result('teacher-verification.json', 'passed')
    state.update(status='pass', teacher_student_short_smoke='pass', rollout_left_running=False)
    save('complete')
except Exception as error:
    state.update(status='fail', error=f'{type(error).__name__}: {error}')
    save('failed')
    raise
finally:
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=20)

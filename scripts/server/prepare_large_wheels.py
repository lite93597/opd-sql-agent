"""Prepare checksummed multipart downloads for the pinned CUDA stack."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename

root = Path('/root/autodl-tmp')
directory = root / 'wheels'
directory.mkdir(exist_ok=True)


def metadata(name, version):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=40) as response:
        return json.load(response)


torch = metadata('torch', '2.11.0')
packages = {'torch': '2.11.0', 'vllm': '0.20.1', 'flashinfer-cubin': '0.6.8.post1'}
extras = {'nvcc', 'cudart', 'cccl', 'crt', 'nvvm'}
for raw in torch['info']['requires_dist']:
    requirement = Requirement(raw)
    if requirement.name.startswith('nvidia-') or requirement.name == 'triton':
        packages[requirement.name] = next(iter(requirement.specifier)).version
    if requirement.name == 'cuda-toolkit':
        extras.update(requirement.extras)
for raw in metadata('cuda-toolkit', '13.0.2')['info']['requires_dist']:
    requirement = Requirement(raw)
    if requirement.marker and any(requirement.marker.evaluate({'extra': extra}) for extra in extras):
        packages[requirement.name] = next(iter(requirement.specifier)).version.rstrip('.*')
tags = set(sys_tags())


def select(item):
    name, version = item
    if name == 'torch':
        official = 'https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl'
        return dict(name=name, version='2.11.0+cu130',
                    filename='torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl',
                    url=official, official_url=official, bytes=531146695,
                    sha256='96911323dcfcd42028c7e8edde7bdf25bb187753234e8775f0f3f112e86a22db')
    candidates = metadata(name, version)['urls']
    for candidate in candidates:
        if not candidate['filename'].endswith('.whl') or candidate['size'] < 20_000_000:
            continue
        wheel_tags = parse_wheel_filename(candidate['filename'])[3]
        if tags & wheel_tags:
            record = dict(name=name, version=version, filename=candidate['filename'],
                        url=candidate['url'].replace('https://files.pythonhosted.org/', 'https://mirrors.aliyun.com/pypi/'),
                        official_url=candidate['url'],
                        bytes=candidate['size'], sha256=candidate['digests']['sha256'])
            releases = {
                'flashinfer-cubin': ('43636d4cd39e694a83d76a89f87fefcdf4cecb4c4f7dd22dac25ec368c1e901f', 'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.8.post1/flashinfer_cubin-0.6.8.post1-py3-none-any.whl'),
                'vllm': ('11907857c94c226caf82ada92ab09b1e0dbf538b5b80a93aed74be29e861020c', 'https://github.com/vllm-project/vllm/releases/download/v0.20.1/vllm-0.20.1-cp38-abi3-manylinux_2_35_x86_64.whl'),
            }
            if name in releases and record['sha256'] == releases[name][0]:
                record['release_url'] = releases[name][1]
            return record


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    wheels = [wheel for wheel in executor.map(select, packages.items()) if wheel]
manifest = root / 'opd-sql-agent/results/server/large-wheels.json'
manifest.write_text(json.dumps(wheels, indent=2))
lines = []
for wheel in wheels:
    mirrors = [wheel['release_url'],wheel['url']] if wheel.get('release_url') else [wheel['url']]
    if 'files.pythonhosted.org' in wheel['official_url']:
        mirrors.extend([wheel['official_url'].replace('https://files.pythonhosted.org/', 'https://pypi.tuna.tsinghua.edu.cn/'), wheel['official_url']])
    lines.extend(['\t'.join(mirrors), f" out={wheel['filename']}", f" checksum=sha-256={wheel['sha256']}"])
(directory / 'aria2-input.txt').write_text('\n'.join(lines) + '\n')
print('wheel_count',len(wheels),'total_bytes',sum(wheel['bytes'] for wheel in wheels),flush=True)

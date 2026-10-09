"""Small SSH transport for server setup; credentials stay in process memory."""
import argparse
import base64
import hashlib
import os
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["exec", "upload", "download"])
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    known_hosts = Path(__file__).resolve().parents[1] / "results/server/known_hosts"
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))

    class TrustFirstConnection(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            digest = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            print(f"SSH host key: {key.get_name()} SHA256:{digest}", flush=True)
            client.get_host_keys().add(hostname, key.get_name(), key)
            client.save_host_keys(str(known_hosts))

    client.set_missing_host_key_policy(TrustFirstConnection())
    client.connect(
        os.environ["OPD_SSH_HOST"], port=int(os.environ["OPD_SSH_PORT"]),
        username=os.environ.get("OPD_SSH_USER", "root"),
        password=os.environ.pop("OPD_SSH_PASSWORD"),
        timeout=30, auth_timeout=30, banner_timeout=30,
        allow_agent=False, look_for_keys=False,
    )
    try:
        if args.action == "exec":
            command = Path(args.source).read_text(encoding="utf-8-sig")
            stdin, stdout, stderr = client.exec_command("bash -s", timeout=args.timeout)
            stdin.write(command)
            stdin.channel.shutdown_write()
            print(stdout.read().decode("utf-8", errors="replace"), end="")
            print(stderr.read().decode("utf-8", errors="replace"), end="")
            raise SystemExit(stdout.channel.recv_exit_status())
        with client.open_sftp() as sftp:
            if args.action == "upload":
                sftp.put(args.source, args.destination)
            else:
                Path(args.destination).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(args.source, args.destination)
            print(f"{args.action}: {args.source} -> {args.destination}")
    finally:
        client.close()


if __name__ == "__main__":
    main()

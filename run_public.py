"""Run the NeuroGene app and print a public link you can send to anyone.

Usage (from the folder that contains alzheimer.py):
    python run_public.py

Keep this window open: the link works only while the script is running.
"""
import atexit
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

APP_FILE = "alzheimer.py"
HERE = Path(__file__).resolve().parent
BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")

procs = []


def cleanup():
    for p in procs:
        if p.poll() is None:
            p.terminate()


atexit.register(cleanup)


def free_port(start=8501):
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found between 8501 and 8520.")


def get_cloudflared():
    found = shutil.which("cloudflared")
    if found:
        return found
    system = platform.system()
    if system == "Windows":
        name = "cloudflared-windows-amd64.exe"
    elif system == "Linux":
        name = "cloudflared-linux-amd64"
    else:
        sys.exit("Install cloudflared first (macOS: brew install cloudflared), then run again.")
    local = HERE / name
    if not local.exists():
        print(f"Downloading cloudflared ({name}) ...")
        urllib.request.urlretrieve(BASE + name, local)
        local.chmod(0o755)
    return str(local)


def wait_for_streamlit(port, timeout=90):
    url = f"http://127.0.0.1:{port}/_stcore/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main():
    if not (HERE / APP_FILE).exists():
        sys.exit(f"{APP_FILE} not found next to run_public.py")

    port = free_port()
    print(f"Starting Streamlit on port {port} ...")
    st = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP_FILE,
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=HERE)
    procs.append(st)

    if not wait_for_streamlit(port):
        sys.exit("Streamlit did not start. Run:  streamlit run alzheimer.py  and read the error.")

    cf = get_cloudflared()
    print("Opening the public tunnel ...")
    tun = subprocess.Popen(
        [cf, "tunnel", "--url", f"http://127.0.0.1:{port}",
         "--no-autoupdate", "--protocol", "http2"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    procs.append(tun)

    lines = queue.Queue()

    def pump():
        for line in tun.stdout:
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()

    url, tail, end = None, [], time.time() + 60
    while time.time() < end and url is None:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if tun.poll() is not None:
                break
            continue
        tail.append(line.rstrip())
        m = URL_RE.search(line)
        if m:
            url = m.group(0)

    if not url:
        print("\nCould not get a public link. Last cloudflared output:")
        print("\n".join(tail[-15:]))
        sys.exit(1)

    (HERE / "public_url.txt").write_text(url, encoding="utf-8")
    print("\n" + "=" * 60)
    print("  Public link (send this):")
    print(f"  {url}")
    print(f"  Local:  http://localhost:{port}")
    print("=" * 60)
    print("Keep this window open. Press Ctrl+C to stop.\n")

    try:
        while st.poll() is None and tun.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

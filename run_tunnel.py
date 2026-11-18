import subprocess
import time
import re
import sys

def start_tunnel():
    # Use 127.0.0.1:8000 explicitly to avoid IPv6 ::1 mismatch
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-R", "80:127.0.0.1:8000", "nokey@localhost.run"]
    print("[*] Launching direct zero-password public tunnel to 127.0.0.1:8000...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    url = None
    for line in proc.stdout:
        clean_line = line.strip()
        print(clean_line)
        match = re.search(r"https://[a-zA-Z0-9.-]+\.lhr\.life", clean_line)
        if match:
            url = match.group(0)
            print("\n" + "=" * 60)
            print("  PUBLIC LIVE DOMAIN (ZERO PASSWORD - DIRECT ACCESS):")
            print(f"  >>> {url} <<<")
            print("=" * 60 + "\n")
            with open("tunnel_url.txt", "w") as f:
                f.write(url)

    proc.wait()

if __name__ == "__main__":
    start_tunnel()

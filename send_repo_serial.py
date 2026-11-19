"""Send the latest commit (HEAD) of the HackEdge repo to the SK-AM62B over the serial console,
then turn /root/hackedge on the board into a git repo pointed at GitHub.

Run on the Mac, with miniterm CLOSED and the board logged in as root:
    python3 send_repo_serial.py . [/dev/cu.usbserial-XXXX0]
    python3 send_repo_serial.py --dir LOCAL_FOLDER REMOTE_FOLDER     # any folder, e.g. pip wheels

Reliable over a noisy serial line: the data is sent in small chunks, each chunk's checksum is
verified on the board and any damaged chunk is re-sent. If it is interrupted, just run it again:
chunks that already arrived intact are skipped.
Skips the video, PDFs, notebook, __pycache__, .DS_Store and pass.md (too big / not needed / secret).
"""
import base64
import glob
import hashlib
import io
import re
import tarfile
from pathlib import Path
import subprocess
import sys
import time

import serial

SKIP = ('.mov', '.pdf', '.ipynb', '.pyc', '.DS_Store', 'pass.md')
DEST, CHUNKS = '/root/hackedge', '/tmp/hkc'
REMOTE_URL = 'https://github.com/nhattruong0205/HackEdge.git'
LINES_PER_CHUNK = 300                       # ~23 KB of base64, ~2 s at 115200 baud
MARK = re.compile(r'@@([0-9a-f]{32}|none)@@')


def read_until(s, pattern, timeout):
    buf, end = '', time.time() + timeout
    while time.time() < end:
        try:
            buf += s.read(s.in_waiting or 1).decode(errors='replace')
        except serial.SerialException:
            sys.exit('\nLost the serial port: another program (miniterm?) opened it, or the cable came out.\n'
                     'Close miniterm and run the same command again; it resumes where it stopped.')
        m = pattern.search(buf)
        if m:
            return m, buf
    return None, buf


def run(s, line, timeout=10):
    """Run a shell command on the board and return its md5-style marker value (or None)."""
    s.reset_input_buffer()
    s.write((line + '\n').encode()); s.flush()
    m, _ = read_until(s, MARK, timeout)
    return m.group(1) if m else None


def remote_md5(s, path):
    return run(s, f'if [ -f {path} ]; then echo @@$(md5sum < {path} | cut -c1-32)@@; else echo @@none@@; fi')


def send_chunk(s, path, text):
    s.reset_input_buffer()
    s.write(f'cat > {path}\n'.encode()); s.flush()
    time.sleep(0.3)                                       # let `cat` start before the data arrives
    for ln in text.splitlines():
        s.write((ln + '\n').encode())
        s.flush()                                         # wait until each line is on the wire
    time.sleep(0.2)
    s.write(b'\x04'); s.flush()                           # Ctrl-D ends `cat`
    time.sleep(0.3)


def main():
    global CHUNKS
    args = sys.argv[1:]
    dir_mode = args[:1] == ['--dir']
    ports = sorted(glob.glob('/dev/cu.usbserial-*0'))
    if dir_mode:
        local, remote = Path(args[1]).expanduser().resolve(), args[2]
        port = args[3] if len(args) > 3 else (ports[0] if ports else None)
    else:
        repo = args[0] if args else '.'
        port = args[1] if len(args) > 1 else (ports[0] if ports else None)
    if not port:
        sys.exit("No board serial port. Plug the USB cable into the board's UART micro-USB port.")

    if dir_mode:
        files = sorted(p for p in local.rglob('*') if p.is_file() and p.name != '.DS_Store')
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            for p in files:
                tar.add(p, arcname=str(p.relative_to(local)))
        data, what = buf.getvalue(), f'folder {local.name}'
        CHUNKS = f'/tmp/hkc_{hashlib.md5(data).hexdigest()[:8]}'
        per_chunk = 1000                                   # bigger chunks for big payloads
    else:
        files = [f for f in subprocess.check_output(['git', '-C', repo, 'ls-files'], text=True).splitlines()
                 if not f.endswith(SKIP)]
        data = subprocess.check_output(['git', '-C', repo, 'archive', '--format=tar.gz', 'HEAD', '--', *files])
        commit = subprocess.check_output(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        what, per_chunk = f'commit {commit}', LINES_PER_CHUNK
    md5 = hashlib.md5(data).hexdigest()
    lines = base64.encodebytes(data).decode().splitlines()
    chunks = ['\n'.join(lines[i:i + per_chunk]) + '\n' for i in range(0, len(lines), per_chunk)]
    print(f'{port}: {what}, {len(files)} files, {len(data)/1e6:.1f} MB in {len(chunks)} chunks (md5 {md5})')
    print(f'Estimated time: ~{len(data) * 4 / 3 / 10500 / 60 + len(chunks) * 0.9 / 60:.0f} min. '
          'Do NOT open miniterm until it says Done.')

    with serial.Serial(port, 115200, timeout=0.2) as s:
        s.write(b'\x03\n'); time.sleep(1)                  # Ctrl-C: clear any half-typed command
        s.write(b'stty -echo\n'); s.flush(); time.sleep(1)
        if run(s, 'echo @@none@@', 5) is None:
            sys.exit('The board did not answer. Is miniterm closed and the board logged in as root?')
        run(s, f'mkdir -p {CHUNKS}; echo @@none@@')

        t0, resent = time.time(), 0
        for i, text in enumerate(chunks):
            path, want = f'{CHUNKS}/{i:05d}', hashlib.md5(text.encode()).hexdigest()
            tries = 0
            while remote_md5(s, path) != want:
                if tries == 5:
                    sys.exit(f'\nChunk {i} keeps arriving damaged. Check the cable and run the script again.')
                resent += tries > 0
                send_chunk(s, path, text)
                tries += 1
            d = (i + 1) / len(chunks)
            print(f'\r  {d:6.1%}  chunk {i+1}/{len(chunks)}  re-sent {resent}  '
                  f'~{(time.time()-t0)/d*(1-d)/60:4.1f} min left', end='', flush=True)

        print('\nJoining and checking on the board…')
        got = run(s, f'cat {CHUNKS}/* | base64 -d > /tmp/hk_repo.tgz && echo @@$(md5sum < /tmp/hk_repo.tgz | cut -c1-32)@@', 300)
        if got != md5:
            sys.exit(f'Final checksum mismatch ({got} != {md5}). Run the script again.')

        if dir_mode:
            ok = run(s, f'mkdir -p {remote} && tar xzf /tmp/hk_repo.tgz -C {remote} && rm -rf {CHUNKS} /tmp/hk_repo.tgz '
                        f'&& echo @@none@@', 120)
            s.write(b'stty echo\n'); s.flush(); time.sleep(0.5)
            if ok is None:
                sys.exit(f'Unpacking did not finish. Open miniterm and check {remote}.')
            print(f'Done: {len(files)} files are in {remote} on the board.')
            return

        print('Checksum OK. Unpacking into', DEST, 'and setting up git…')
        git = 'git -c user.name=board -c user.email=board@localhost'
        ok = run(s, f'mkdir -p {DEST} && tar xzf /tmp/hk_repo.tgz -C {DEST} && cd {DEST} && '
                    f'(test -d .git || git init -q -b main) && '
                    f'(git remote get-url origin >/dev/null 2>&1 || git remote add origin {REMOTE_URL}) && '
                    f"printf '%s\\n' '*.pyc' '__pycache__/' '.DS_Store' 'pass.md' > .git/info/exclude && "
                    f'{git} add -A && ({git} commit -qm "snapshot of {commit} from Mac" || true) && '
                    f'cd /root && rm -rf {CHUNKS} && echo @@none@@', 60)
        s.write(b'stty echo\n'); s.flush(); time.sleep(0.5)
        if ok is None:
            sys.exit('Unpacking did not finish. Open miniterm and check /root/hackedge.')
        print(f'Done: {DEST} on the board now holds commit {commit} as a git repo (origin = GitHub).')


if __name__ == '__main__':
    main()

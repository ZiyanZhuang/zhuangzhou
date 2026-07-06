import os
import subprocess
import shutil

def win_to_wsl(path: str) -> str:
    """Convert Windows path to WSL path."""
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    return path

def run_wsl(cmd: str, cwd: str = None, timeout: int = 3600):
    """Run a command in WSL bash with login environment."""
    quoted_cwd = f"'{cwd}'" if cwd else "'/tmp'"
    # Note: adjust the path to your amber.sh
    full_cmd = (
        f'wsl -e bash -l -c "source ~/amber25/amber.sh && cd {win_to_wsl(quoted_cwd)} && {cmd}"'
    )
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result

def write_file(path: str, content: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

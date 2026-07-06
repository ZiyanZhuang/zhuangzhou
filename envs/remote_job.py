"""
Remote Job Orchestrator
========================
High-level API for offloading compute jobs to SSH nodes.

Handles the full lifecycle:
  1. Upload input files (local -> remote)
  2. Execute computation (SSH, optionally detached via tmux)
  3. Poll progress (log streaming)
  4. Download results (remote -> local)

This is the bridge layer that pipeline modules (gromacs_md, qm_engine, etc.)
call instead of subprocess.run() when remote execution is selected.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sandbox import SSHSandbox, SandboxResult

logger = logging.getLogger(__name__)


@dataclass
class RemoteJob:
    """Tracks a remote compute job."""
    job_id: str
    host_name: str
    session_name: str  # tmux session
    remote_work_dir: str
    local_work_dir: str
    status: str = 'pending'  # pending, uploading, running, downloading, completed, failed
    command: str = ''
    input_files: List[str] = field(default_factory=list)
    output_patterns: List[str] = field(default_factory=list)  # glob patterns for results
    error: str = ''


def submit_job(
    ssh_host: str,
    command: str,
    local_work_dir: str,
    input_files: List[str] = None,
    output_patterns: List[str] = None,
    conda_env: str = None,
    detached: bool = True,
) -> Dict[str, Any]:
    """
    Submit a compute job to a remote SSH host with robust status tracking.
    """
    sandbox = SSHSandbox(ssh_host)
    if not sandbox.host_profile:
        return {'error': f"SSH host '{ssh_host}' not found",
                'hint': 'Configure ~/.zhuangzhou/ssh_compute.ini'}

    job_id = f'zz_{uuid.uuid4().hex[:8]}'
    hp = sandbox.host_profile
    remote_dir = f'{hp.remote_work_dir}/{job_id}'

    # 1. Create remote directory
    r = sandbox.run(f'mkdir -p {remote_dir}', timeout=15, use_conda=False)
    if not r.success:
        return {'error': f'Cannot create remote dir: {r.error}'}

    # 2. Upload input files
    local_dir = os.path.abspath(local_work_dir)
    if input_files:
        files_to_upload = [os.path.join(local_dir, f) for f in input_files]
        r = sandbox.upload_files(files_to_upload, remote_dir)
    else:
        r = sandbox.sync_to_remote(local_dir, remote_dir)
    if not r.success:
        return {'error': f'Upload failed: {r.error}'}
    logger.info("Uploaded inputs to %s:%s", ssh_host, remote_dir)

    # 3. Wrap command with sentinel files and log redirection
    # This ensures we know the status even if tmux fails/server reboots.
    wrapped_command = (
        f'echo "RUNNING" > status.txt; '
        f'{{ {command}; }} > job.log 2>&1; '
        f'echo $? > exit_code.txt; '
        f'echo "FINISHED" > status.txt'
    )

    # 4. Execute command
    if detached:
        r = sandbox.run_detached(wrapped_command, session_name=job_id,
                                 work_dir=remote_dir)
    else:
        r = sandbox.run(wrapped_command, work_dir=remote_dir)

    if not r.success:
        return {'error': f'Execution failed: {r.error}'}

    return {
        'job_id': job_id,
        'host': ssh_host,
        'session': job_id if detached else None,
        'remote_dir': remote_dir,
        'local_dir': local_dir,
        'status': 'running' if detached else ('completed' if r.success else 'failed'),
        'output': r.data if not detached else None,
    }


def check_job(ssh_host: str, job_id: str) -> Dict[str, Any]:
    """Check the status of a remote job using sentinel files."""
    sandbox = SSHSandbox(ssh_host)
    hp = sandbox.host_profile
    remote_dir = f'{hp.remote_work_dir}/{job_id}'
    
    # Query sentinel files
    check_cmd = (
        f'cd {remote_dir} && '
        f'[ -f exit_code.txt ] && echo "EXIT:$(cat exit_code.txt)" || '
        f'([ -f status.txt ] && echo "STATUS:$(cat status.txt)" || echo "STATUS:PENDING")'
    )
    r = sandbox.run(check_cmd, timeout=15, use_conda=False)
    
    if not r.success:
        # Fallback to tmux if directory is missing or SSH failed
        return sandbox.check_session(job_id).data

    output = r.stdout.strip()
    if output.startswith('EXIT:'):
        try:
            code = int(output.split(':')[1].strip())
        except (ValueError, IndexError):
            code = -1
        return {
            'job_id': job_id,
            'status': 'finished' if code == 0 else 'failed',
            'exit_code': code
        }

    status = output.split(':')[1].strip() if ':' in output else 'pending'
    # Double check with tmux if it says RUNNING
    if status == 'RUNNING':
        tmux_r = sandbox.check_session(job_id)
        if tmux_r.success and tmux_r.data.get('status') == 'finished':
            # Tmux is gone but no exit_code.txt? This is an abnormal failure.
            return {'job_id': job_id, 'status': 'failed', 'error': 'Process vanished'}
            
    return {'job_id': job_id, 'status': status.lower()}


def get_job_log(ssh_host: str, job_id: str, lines: int = 50) -> Dict[str, Any]:
    """Get recent output from the job.log file or tmux buffer."""
    sandbox = SSHSandbox(ssh_host)
    hp = sandbox.host_profile
    remote_dir = f'{hp.remote_work_dir}/{job_id}'
    
    # Try reading the redirected log file first (more reliable)
    log_cmd = f'tail -n {lines} {remote_dir}/job.log 2>/dev/null || echo "LOG_MISSING"'
    r = sandbox.run(log_cmd, timeout=15, use_conda=False)
    
    if r.success and "LOG_MISSING" not in r.stdout:
        return {'log': r.stdout, 'session': job_id, 'source': 'file'}
    
    # Fallback to tmux capture
    r = sandbox.get_session_output(job_id, lines)
    if not r.success:
        return {'error': r.error, 'hint': 'Log file not found and tmux session missing'}
    return {'log': r.stdout, 'session': job_id, 'source': 'tmux'}


def collect_results(
    ssh_host: str,
    job_id: str,
    local_work_dir: str,
    remote_dir: str = None,
    output_patterns: List[str] = None,
) -> Dict[str, Any]:
    """
    Download results from a completed remote job.

    Args:
        ssh_host: SSH host name
        job_id: Job ID
        local_work_dir: Local directory to save results
        remote_dir: Remote work directory (auto-resolved if None)
        output_patterns: Specific file patterns to download (None = all)
    """
    sandbox = SSHSandbox(ssh_host)
    hp = sandbox.host_profile
    if not hp:
        return {'error': f"SSH host '{ssh_host}' not found"}

    remote = remote_dir or f'{hp.remote_work_dir}/{job_id}'

    if output_patterns:
        # Download specific files matching patterns
        # List remote files first
        r = sandbox.run(f'ls {remote}/', timeout=15, use_conda=False)
        if not r.success:
            return {'error': f'Cannot list remote dir: {r.error}'}

        import fnmatch
        remote_files = r.stdout.strip().split('\n')
        to_download = []
        for pat in output_patterns:
            for f in remote_files:
                f = f.strip()
                if f and fnmatch.fnmatch(f, pat):
                    to_download.append(f'{remote}/{f}')

        if not to_download:
            return {'error': 'No files matched output patterns',
                    'remote_files': remote_files,
                    'patterns': output_patterns}

        r = sandbox.download_files(to_download, local_work_dir)
    else:
        r = sandbox.sync_from_remote(remote, local_work_dir)

    if not r.success:
        return {'error': f'Download failed: {r.error}'}

    return {
        'status': 'collected',
        'local_dir': local_work_dir,
        'files': r.data,
    }


def cancel_job(ssh_host: str, job_id: str) -> Dict[str, Any]:
    """Cancel a running remote job."""
    sandbox = SSHSandbox(ssh_host)
    r = sandbox.kill_session(job_id)
    return {'status': 'cancelled' if r.success else 'error',
            'error': r.error if not r.success else ''}


def run_and_wait(
    ssh_host: str,
    command: str,
    local_work_dir: str,
    input_files: List[str] = None,
    output_patterns: List[str] = None,
    poll_interval: int = 30,
    max_wait: int = 86400,
    on_progress: callable = None,
) -> Dict[str, Any]:
    """
    Submit a job, wait for completion, and collect results.

    This is the synchronous convenience wrapper for the full lifecycle.
    Suitable for interactive use where the user wants to wait.

    Args:
        on_progress: Optional callback(job_id, log_lines) called each poll cycle
        max_wait: Maximum wait time in seconds (default: 24h). Prevents infinite polling.
    """
    # Submit
    result = submit_job(ssh_host, command, local_work_dir,
                        input_files, output_patterns, detached=True)
    if 'error' in result:
        return result

    job_id = result['job_id']
    remote_dir = result['remote_dir']
    start_time = time.time()

    # Poll until done (with max_wait guard)
    while True:
        if time.time() - start_time > max_wait:
            return {
                'error': f'Job {job_id} exceeded max wait time ({max_wait}s)',
                'job_id': job_id,
                'remote_dir': remote_dir,
                'hint': f'Job may still be running. Check with check_job("{ssh_host}", "{job_id}")',
            }
        time.sleep(poll_interval)
        status = check_job(ssh_host, job_id)
        if 'error' in status:
            return status

        if on_progress:
            log = get_job_log(ssh_host, job_id, lines=20)
            on_progress(job_id, log.get('log', ''))

        job_status = status.get('status', '')
        if job_status == 'finished':
            break
        if job_status == 'failed':
            # Collect log for diagnosis before returning
            log = get_job_log(ssh_host, job_id, lines=100)
            return {
                'error': f'Remote job {job_id} failed',
                'exit_code': status.get('exit_code', -1),
                'log': log.get('log', ''),
                'job_id': job_id,
                'remote_dir': remote_dir,
            }

    # Collect results
    return collect_results(ssh_host, job_id, local_work_dir,
                           remote_dir, output_patterns)

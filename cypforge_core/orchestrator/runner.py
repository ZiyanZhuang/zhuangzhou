"""ModuleRunner — execute StepDef commands via subprocess with full logging."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunConfig, StepDef, StepRecord


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _win_to_wsl(path: str) -> str:
    """Convert a Windows path to its WSL /mnt/<drive>/... equivalent."""
    from ..io import _win_to_wsl as _w2w

    return _w2w(path)


def _format_cmd(template: str, values: dict[str, Any]) -> str:
    """Safe-ish template substitution. Removes --flag tokens when value is empty."""
    import re
    result = template
    # First pass: remove --flag {key} patterns where value is empty/None
    for key, val in values.items():
        val_str = str(val) if val is not None else ""
        if not val_str.strip():
            # Remove the --flagname {key} (with optional quotes) from the template
            result = re.sub(
                r'\s*--\S+\s+\{' + re.escape(key) + r'\}\s*',
                ' ', result
            )
    # Second pass: substitute remaining placeholders
    for key, val in values.items():
        val_str = str(val) if val is not None else ""
        result = result.replace("{" + key + "}", val_str)
    # Normalize path separators (backslash to forward slash on non-Windows)
    if sys.platform != "win32":
        result = result.replace("\\", "/")
    # Clean up
    result = re.sub(r'  +', ' ', result)
    result = result.strip()
    return result


def _build_env(run_config: RunConfig) -> dict[str, str]:
    env = os.environ.copy()
    python_path = run_config.python_path
    if python_path:
        existing = env.get("PYTHONPATH", "")
        sep = ";" if sys.platform == "win32" else ":"
        env["PYTHONPATH"] = (
            python_path + sep + existing if existing else python_path
        )
    if run_config.amber_sh:
        env["AMBER_SH"] = run_config.amber_sh
    if run_config.multiwfn_bin:
        env["MULTIWFN_BIN"] = run_config.multiwfn_bin
    return env


class ModuleRunner:
    """Execute workflow module steps with logging."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.project_root = Path(config.project_root)

    def _resolve_values(self) -> dict[str, Any]:
        return {
            "run_root": self.config.run_root,
            "run_root_wsl": _win_to_wsl(self.config.run_root),
            "project_root": self.config.project_root,
            "project_root_wsl": _win_to_wsl(self.config.project_root),
            "raw_protein_heme_pdb": self.config.raw_protein_heme_pdb,
            "ligand_template_sdf": self.config.ligand_template_sdf,
            "heme_state": self.config.heme_state,
            "heme_resname": self.config.heme_resname,
            "heme_chain": self.config.heme_chain,
            "protein_chain": self.config.protein_chain,
            "axial_cys_resid": self.config.axial_cys_resid or "",
            "ligand_resname": self.config.ligand_resname,
            "ligand_chain": self.config.ligand_chain,
            "formal_charge": self.config.formal_charge,
            "spin": self.config.spin,
            "basis": self.config.basis,
            "points_per_atom": self.config.points_per_atom,
            "fit_method": self.config.fit_method,
            "pre_resp_relax": self.config.pre_resp_relax,
            "protonation_decision_json": self.config.protonation_decision_json,
            "protein_force_field": self.config.protein_force_field,
            "water_leaprc": self.config.water_leaprc,
            "water_model": self.config.water_model,
            "box_type": self.config.box_type,
            "buffer_a": self.config.buffer_a,
            "neutralizing_anion": self.config.neutralizing_anion,
        }

    def run_step(self, step: StepDef, log_dir: Path) -> StepRecord:
        """Execute a single step. Returns a StepRecord with full logging."""
        values = self._resolve_values()
        command = _format_cmd(step.command_template, values)

        record = StepRecord(
            name=step.name,
            command=command,
            working_dir=str(self.project_root),
            started_at=_ts(),
        )

        log_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{_ts()}_{step.name}"

        stdout_path = log_dir / f"{prefix}_stdout.txt"
        stderr_path = log_dir / f"{prefix}_stderr.txt"
        cmd_path = log_dir / f"{prefix}_command.txt"

        record.stdout_path = str(stdout_path)
        record.stderr_path = str(stderr_path)

        # Write command log
        cmd_path.write_text(
            f"name: {step.name}\n"
            f"kind: {step.kind}\n"
            f"command: {command}\n"
            f"working_dir: {self.project_root}\n"
            f"started_at: {record.started_at}\n",
            encoding="utf-8",
        )

        try:
            if step.kind == "python_script":
                record = self._run_python(command, record, stdout_path, stderr_path, step)
            elif step.kind == "wsl_command":
                record = self._run_wsl(command, record, stdout_path, stderr_path, step)
            elif step.kind == "python_inline":
                record = self._run_python_inline(command, record, stdout_path, stderr_path, step)
            else:
                record.status = "FAIL"
                record.error_message = f"Unknown step kind: {step.kind}"
        except Exception as exc:
            record.status = "FAIL"
            record.error_message = str(exc)

        record.completed_at = _ts()

        # Append completion info to command log
        with cmd_path.open("a", encoding="utf-8") as fh:
            fh.write(f"status: {record.status}\n")
            fh.write(f"exit_code: {record.exit_code}\n")
            fh.write(f"completed_at: {record.completed_at}\n")
            if record.error_message:
                fh.write(f"error: {record.error_message}\n")

        return record

    def _run_python(
        self,
        command: str,
        record: StepRecord,
        stdout_path: Path,
        stderr_path: Path,
        step: StepDef,
    ) -> StepRecord:
        """Run a python script command via subprocess."""
        env = _build_env(self.config)
        parts = self._parse_command(command)
        if parts and parts[0] == "python":
            parts[0] = sys.executable
        timeout = step.timeout_seconds

        try:
            proc = subprocess.run(
                parts,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            record.status = "FAIL"
            record.exit_code = -1
            record.error_message = f"Step timed out after {timeout}s"
            return record

        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        record.exit_code = proc.returncode
        record.status = "PASS" if proc.returncode == 0 else "FAIL"

        if proc.returncode != 0:
            record.error_message = (
                f"Exit code {proc.returncode}. See {stderr_path}"
            )

        return record

    def _run_wsl(
        self,
        command: str,
        record: StepRecord,
        stdout_path: Path,
        stderr_path: Path,
        step: StepDef,
    ) -> StepRecord:
        """Run a bash command through WSL. If already inside WSL, runs directly."""
        amber_sh = self.config.amber_sh or os.environ.get(
            "AMBER_SH", os.environ.get("AMBERHOME", "")
        )
        if amber_sh and not amber_sh.endswith("amber.sh"):
            amber_sh = str(Path(amber_sh) / "amber.sh")

        wsl_user = self.config.wsl_user
        if amber_sh:
            bash_cmd = f"source {amber_sh} && {command}"
        else:
            bash_cmd = command

        timeout = step.timeout_seconds

        # Detect if we are already inside WSL
        is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") if sys.platform == "linux" else False

        if is_wsl:
            # Already inside WSL, run bash directly
            parts = ["bash", "-lc", bash_cmd]
            full_cmd = f"bash -lc \"{bash_cmd}\""
            run_kwargs = {}
        else:
            parts = ["wsl", "-u", wsl_user, "-e", "bash", "-lc", bash_cmd]
            full_cmd = f"wsl -u {wsl_user} -e bash -lc \"{bash_cmd}\""
            run_kwargs = {}

        try:
            proc = subprocess.run(
                parts,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            record.status = "FAIL"
            record.exit_code = -1
            record.error_message = f"WSL step timed out after {timeout}s"
            record.command = full_cmd
            return record

        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        record.exit_code = proc.returncode
        record.status = "PASS" if proc.returncode == 0 else "FAIL"
        record.command = full_cmd

        if proc.returncode != 0:
            record.error_message = (
                f"WSL exit code {proc.returncode}. See {stderr_path}"
            )

        return record

    def _run_python_inline(
        self,
        command: str,
        record: StepRecord,
        stdout_path: Path,
        stderr_path: Path,
        step: StepDef,
    ) -> StepRecord:
        """Run a python -c '<code>' inline command."""
        env = _build_env(self.config)
        timeout = step.timeout_seconds

        try:
            proc = subprocess.run(
                [sys.executable, "-c", command],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            record.status = "FAIL"
            record.exit_code = -1
            record.error_message = f"Step timed out after {timeout}s"
            return record

        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        record.exit_code = proc.returncode
        record.status = "PASS" if proc.returncode == 0 else "FAIL"

        if proc.returncode != 0:
            record.error_message = (
                f"Exit code {proc.returncode}. See {stderr_path}"
            )

        return record

    @staticmethod
    def _parse_command(command: str) -> list[str]:
        """Minimal argv parser for python script commands."""
        import shlex
        return shlex.split(command, posix=(sys.platform != "win32"))

"""Shell backend whose ``execute`` returns when the *shell* exits.

deepagents' ``LocalShellBackend.execute`` captures output through pipes
(``subprocess.run(capture_output=True)``). A pipe stays open as long as *any*
process holds it, so ``python3 -m http.server &`` or ``nginx &`` — the shell is
long gone, the backgrounded child still holds stdout — blocks the call for the
full timeout, reports "Command timed out", and leaves the model believing its
server never started. Terminal-Bench is full of tasks whose fix *is* a running
service (a PyPI mirror, an SSH-able VM, a git web server), and that pattern
broke every one of them.

Writing the output to a temporary file instead of a pipe removes the coupling:
``run`` waits for the shell only, background children keep the file handle (it
is unlinked, harmless), and their later output simply never reaches the tool.
Each command runs in its own session so a timeout can stop the whole process
group rather than orphaning a busy child that keeps the CPU.

POSIX only; elsewhere the deepagents implementation is used unchanged.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

_POSIX = os.name == "posix"


class LoomShellBackend(LocalShellBackend):
    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not _POSIX:
            return super().execute(command, timeout=timeout)
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.", exit_code=1, truncated=False
            )
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")

        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(  # noqa: S602 — LLM-driven shell, by design
                    command,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    env=self._env,
                    cwd=str(self.cwd),
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001 — same contract as upstream
                return ExecuteResponse(
                    output=f"Error executing command ({type(exc).__name__}): {exc}",
                    exit_code=1,
                    truncated=False,
                )
            try:
                returncode = proc.wait(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                hint = (
                    "(custom timeout). The command may be stuck or require more time."
                    if timeout is not None
                    else "For long-running commands, re-run using the timeout parameter."
                )
                partial = _read(out, self._max_output_bytes)
                tail = f"\nOutput before the timeout:\n{partial}" if partial else ""
                return ExecuteResponse(
                    output=(
                        f"Error: Command timed out after {effective_timeout} seconds {hint} "
                        "To leave a server or other long-lived process running, start it "
                        "in the background with its output redirected, e.g. "
                        "`nohup cmd > /tmp/cmd.log 2>&1 &`." + tail
                    ),
                    exit_code=124,
                    truncated=False,
                )
            stdout = _read(out, self._max_output_bytes + 1)
            stderr = _read(err, self._max_output_bytes + 1)

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.extend(f"[stderr] {line}" for line in stderr.strip().split("\n"))
        output = "\n".join(parts) if parts else "<no output>"
        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True
        if returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {returncode}"
        return ExecuteResponse(output=output, exit_code=returncode, truncated=truncated)


def _read(fh, limit: int) -> str:
    fh.seek(0)
    return fh.read(limit).decode("utf-8", errors="replace")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

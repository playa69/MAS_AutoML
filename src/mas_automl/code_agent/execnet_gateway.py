# -*- coding: utf-8 -*-

import execnet
import textwrap
import sys
from dataclasses import dataclass
import time
import traceback
import logging
from typing import Optional

LOGFILE = "execnet_gateway.log"

# --- logging setup ---
logger = logging.getLogger("execnet_gateway")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
# console
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
# file
fh = logging.FileHandler(LOGFILE, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str


class PythonSandboxClient:
    """
    Robust execnet sandbox client.
    - uses the exact sys.executable (your environment)
    - preserves globals between runs
    - auto-restarts gateway/channel on failure (with limited retries)
    - logs to console and file
    """

    _instance: Optional["PythonSandboxClient"] = None
    _remote_template = textwrap.dedent("""
        import sys, io, traceback

        # Persistent globals for session
        _globals = {}

        for code in channel:
            # Capture outputs
            sys_stdout, sys_stderr = io.StringIO(), io.StringIO()
            sys.stdout, sys.stderr = sys_stdout, sys_stderr

            # Keep a small stderr note for visibility in host logs
            sys.__stderr__.write("[sandbox] received code\\n")
            sys.__stderr__.flush()

            result = {"ok": True, "stdout": "", "stderr": ""}

            try:
                # Execute in persistent context
                exec(code, _globals)
            except Exception:
                result["ok"] = False
                result["stderr"] = traceback.format_exc()

            # Collect outputs
            result["stdout"] = sys_stdout.getvalue()
            result["stderr"] += sys_stderr.getvalue()

            # Send result back
            channel.send(result)
    """)

    def __init__(self, max_retries: int = 2, restart_backoff: float = 0.25):
        if PythonSandboxClient._instance is not None:
            raise RuntimeError("Use PythonSandboxClient.get()")

        self.max_retries = max_retries
        self.restart_backoff = restart_backoff
        self._create_gateway()

    @classmethod
    def get(cls) -> "PythonSandboxClient":
        if cls._instance is None:
            cls._instance = PythonSandboxClient()
        return cls._instance

    def _create_gateway(self):
        """(Re)create gateway and remote channel."""
        logger.info("Creating new execnet gateway with python=%s", sys.executable)
        try:
            # Keep the same interpreter
            self.gw = execnet.makegateway(f"popen//python={sys.executable}")
            self.channel = self.gw.remote_exec(self._remote_template)
            logger.info("Gateway and channel created.")
        except Exception:
            logger.exception("Failed to create execnet gateway.")
            raise

    def _close_gateway_safe(self):
        try:
            if getattr(self, "channel", None):
                try:
                    self.channel.close()
                except Exception:
                    logger.debug("Channel close raised", exc_info=True)
            if getattr(self, "gw", None):
                try:
                    self.gw.exit()
                except Exception:
                    logger.debug("Gateway exit raised", exc_info=True)
        finally:
            self.channel = None
            self.gw = None

    def run(self, code: str) -> SandboxResult:
        """
        Execute code in persistent remote interpreter.
        If channel/gateway is dead, attempt to recreate and retry up to max_retries.
        """
        last_exc = None
        attempt = 0

        while attempt <= self.max_retries:
            attempt += 1
            try:
                if not getattr(self, "channel", None):
                    logger.warning("No active channel: creating gateway before attempt %d", attempt)
                    self._create_gateway()

                logger.debug("Sending code to sandbox (attempt %d):\n%s", attempt, self._shorten(code))
                self.channel.send(code)
                logger.debug("Code sent, waiting for result...")
                raw = self.channel.receive()
                logger.debug("Received result from sandbox (attempt %d).", attempt)

                ok = bool(raw.get("ok", False))
                stdout = raw.get("stdout", "") or ""
                stderr = raw.get("stderr", "") or ""

                # Log returned streams in host logger
                if stdout:
                    logger.info("[sandbox stdout]\n%s", stdout.rstrip())
                if stderr:
                    logger.warning("[sandbox stderr]\n%s", stderr.rstrip())

                return SandboxResult(ok=ok, stdout=stdout, stderr=stderr)

            except OSError as ose:
                # Channel is closed or pipe died
                last_exc = ose
                logger.error("OSError during send/receive (attempt %d): %s", attempt, ose)
                logger.exception("Traceback:")
                # try to recreate gateway/channel and retry
                try:
                    logger.info("Recreating gateway (attempt %d)...", attempt)
                    self._close_gateway_safe()
                    time.sleep(self.restart_backoff)
                    self._create_gateway()
                except Exception as e2:
                    logger.error("Failed to recreate gateway on attempt %d: %s", attempt, e2)
                    last_exc = e2
                    time.sleep(self.restart_backoff)
                    continue  # try again until attempts exhausted

            except Exception as ex:
                # Unexpected error - capture traceback and return it to caller
                last_exc = ex
                logger.exception("Unexpected exception while running code (attempt %d).", attempt)
                # If channel is still alive, try to receive remaining message (best-effort)
                try:
                    if getattr(self, "channel", None) and not self.channel.isclosed():
                        # non-blocking? channel.receive will block until message; avoid calling here
                        pass
                except Exception:
                    logger.debug("Error while probing channel after exception.", exc_info=True)
                # If this was last attempt — return failure
                if attempt > self.max_retries:
                    tb = traceback.format_exc()
                    return SandboxResult(ok=False, stdout="", stderr=f"Exception: {ex}\n{tb}")
                else:
                    # try to recreate and retry
                    try:
                        self._close_gateway_safe()
                        time.sleep(self.restart_backoff)
                        self._create_gateway()
                    except Exception as e2:
                        logger.error("Failed to recreate gateway after unexpected exception: %s", e2)
                        return SandboxResult(ok=False, stdout="", stderr=f"Restart failed: {e2}")

        # exhausted attempts
        logger.error("All attempts exhausted. Last exception: %s", last_exc)
        return SandboxResult(ok=False, stdout="", stderr=f"All attempts failed: {last_exc}")

    def close(self):
        """Close channel and gateway and reset singleton."""
        logger.info("Closing sandbox gateway.")
        try:
            self._close_gateway_safe()
        finally:
            PythonSandboxClient._instance = None

    @staticmethod
    def _shorten(s: str, maxlen: int = 1000) -> str:
        if not s:
            return ""
        s2 = s.strip()
        if len(s2) <= maxlen:
            return s2
        return s2[:maxlen] + "...[truncated]"


import execnet
import textwrap
import sys
from dataclasses import dataclass


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str


class PythonSandboxClient:
    """
    Singleton-клиент песочницы.
    Использует один persistent интерпретатор Python,
    в котором сохраняются импорты, переменные и состояние.
    """

    _instance = None

    def __init__(self):
        # Запрет прямой инициализации
        if PythonSandboxClient._instance is not None:
            raise RuntimeError("Используй PythonSandboxClient.get()")

        self.gw = execnet.makegateway(f"popen//python={sys.executable}")

        # persistent remote session
        self.channel = self.gw.remote_exec(textwrap.dedent("""
            import sys, io, traceback

            # Постоянный REPL: выполняем код, храним переменные в globals()
            for code in channel:
                sys_stdout, sys_stderr = io.StringIO(), io.StringIO()
                sys.stdout, sys.stderr = sys_stdout, sys_stderr

                result = {"ok": True, "stdout": "", "stderr": ""}

                try:
                    exec(code, globals())  # сохраняем контекст между вызовами
                except Exception:
                    result["ok"] = False
                    result["stderr"] = traceback.format_exc()

                result["stdout"] = sys_stdout.getvalue()
                result["stderr"] += sys_stderr.getvalue()

                channel.send(result)
        """))

    @classmethod
    def get(cls):
        """Возвращает singleton-песочницу."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(self, code: str) -> SandboxResult:
        """Выполняет код в текущем интерпретаторе."""
        self.channel.send(code)
        result = self.channel.receive()
        return SandboxResult(
            ok=result["ok"],
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
        )

    def close(self):
        """Закрывает песочницу полностью."""
        try:
            self.channel.close()
        except Exception:
            pass
        try:
            self.gw.exit()
        except Exception:
            pass
        PythonSandboxClient._instance = None



# if __name__=="__main__":
#     sandbox = PythonSandboxClient.get()

#     # Импорт один раз
#     print(sandbox.run(
#         "import pandas as pd; print('Pandas loaded!')"
#     ))

#     # Переменные сохраняются
#     print(sandbox.run(
#         "df = pd.DataFrame({'x':[1,2,3]}); print(df)"
#     ))

#     # Следующий вызов видит df
#     print(sandbox.run(
#         "print('mean =', df['x'].mean())"
#     ))

#     sandbox.close()

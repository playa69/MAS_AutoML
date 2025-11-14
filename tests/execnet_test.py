import execnet
import textwrap
import sys

class PythonSandbox:
    """Песочница на execnet с одним интерпретатором (сохраняет окружение)."""

    def __init__(self):
        self.gw = execnet.makegateway(f"popen//python={sys.executable}")
        self.channel = self.gw.remote_exec(textwrap.dedent("""
            import sys, io, traceback, builtins

            # Перехватываем входящие куски кода и исполняем их последовательно
            for code in channel:
                sys_stdout, sys_stderr = io.StringIO(), io.StringIO()
                sys.stdout, sys.stderr = sys_stdout, sys_stderr

                result = {"ok": True, "stdout": "", "stderr": ""}

                try:
                    exec(code, globals())  # 👈 сохраняем контекст
                except Exception:
                    result["ok"] = False
                    result["stderr"] = traceback.format_exc()

                result["stdout"] = sys_stdout.getvalue()
                result["stderr"] += sys_stderr.getvalue()

                # Возвращаем результат
                channel.send(result)
        """)) 

    def run(self, code: str):
        """Выполняет код в существующем окружении."""
        self.channel.send(code)
        return self.channel.receive()

    def close(self):
        self.channel.close()
        self.gw.exit()


sandbox = PythonSandbox()

# Загружаем зависимости один раз
print(sandbox.run("from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate; print('✅ LANGCHAIN ready')"))

# # Теперь можем использовать pd без переимпорта
# print(sandbox.run("df = pd.DataFrame({'x':[1,2,3]}); print(df.describe())"))

# # Следующий код видит df
# print(sandbox.run("print('Mean:', df['x'].mean())"))

sandbox.close()

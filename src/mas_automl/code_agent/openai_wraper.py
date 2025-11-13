from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class LLMConfig:
    #model: str = "deepseek/deepseek-chat-v3.1:free"
    temperature: float = 0.0


class LLMClient:
    """Обёртка над LangChain ChatOpenAI (OpenRouter совместимый клиент) с fallback-логикой."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        _load_env_file()
        print("🔑 OPENAI_API_KEY =", os.getenv("MAS_LLM__API_KEY")[:7])
        print("🌐 BASE_URL =", os.getenv("MAS_LLM__BASE_URL"))

        self.config = config or LLMConfig()
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("MAS_LLM__API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MAS_LLM__BASE_URL")

        self._client: ChatOpenAI | None = None
        model_from_env = os.getenv("MAS_LLM__MODEL")
        if model_from_env:
            self.config.model = model_from_env

        if api_key:
            client_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "openai_api_key": api_key,
                "openai_api_base": base_url or "https://openrouter.ai/api/v1",

            }
            print(f"KWARGS - {client_kwargs}")
            try:
                self._client = ChatOpenAI(**client_kwargs)
                print("✅ ChatOpenAI клиент инициализирован для OpenRouter")
            except Exception as e:
                import traceback
                print("❌ Ошибка инициализации клиента:", repr(e))
                traceback.print_exc()
                self._client = None

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        fallback: str = "",
        temperature: float | None = None,
    ) -> str:
        if self._client is None:
            return fallback or self._default_fallback(prompt)

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        try:
            llm = self._client.bind(temperature=temperature or self.config.temperature)
            response = llm.invoke(messages)
            content = getattr(response, "content", None)
            if not content:
                raise ValueError("LLM вернул пустой ответ.")
            return content.strip()
        except Exception as e:
            import traceback
            print("❌ Ошибка при вызове ChatOpenAI:", repr(e))
            traceback.print_exc()
            return fallback or self._default_fallback(prompt)

    def _default_fallback(self, prompt: str) -> str:
        """Минимальная эвристика на случай отсутствия доступа к LLM."""
        if "framework" in prompt.lower():
            return json.dumps(
                {
                    "framework": "AutoGluon",
                    "reason": "Выбран по умолчанию: хорошо работает с табличной классификацией.",
                },
                ensure_ascii=False,
            )
        return "# Fallback: LLM недоступен."


def _load_env_file() -> None:
    env_loaded_flag = "_MAS_CODE_AGENT_ENV_LOADED"
    if os.environ.get(env_loaded_flag):
        return

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        os.environ[env_loaded_flag] = "1"
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value  # <-- ВАЖНО! заменяем setdefault()

    os.environ[env_loaded_flag] = "1"


__all__ = ["LLMClient", "LLMConfig"]

if __name__ == "__main__":
    print("🚀 Старт диагностики LLMClient")
    try:
        client = LLMClient(LLMConfig())
        print("✅ LLMClient создан.")
    except Exception as e:
        print("❌ Ошибка при создании LLMClient:", repr(e))
        raise SystemExit(1)

    print("\n🔍 Проверяем объект клиента:")
    print("   _client =", type(getattr(client, "_client", None)).__name__)

    if client._client is None:
        print("⚠️  Клиент не инициализирован. Проверь API_KEY и BASE_URL.")
        raise SystemExit(2)

    print("\n💬 Пробуем простой запрос к модели...")
    try:
        reply = client.chat("Скажи 'тест связи'")
        print("✅ Ответ от модели:", repr(reply))
    except Exception as e:
        print("❌ Ошибка при вызове chat():", repr(e))

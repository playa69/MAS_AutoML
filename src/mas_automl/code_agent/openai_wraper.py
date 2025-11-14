from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - библиотека может отсутствовать в окружении
    OpenAI = None  # type: ignore


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class LLMClient:
    """Простой обёртки над OpenAI API с fallback-логикой."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        _load_env_file()
        self.config = config or LLMConfig()
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("MAS_LLM__API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MAS_LLM__BASE_URL")
        self._client: Any | None = None
        if OpenAI is not None and api_key:
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            print(client_kwargs)
            if base_url:
                client_kwargs["base_url"] = base_url
            try:
                self._client = OpenAI(**client_kwargs)  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - падение клиента
                self._client = None

    def chat(self, prompt: str, *, system: str | None = None, fallback: str = "", temperature: float | None = None) -> str:
        temperature = temperature if temperature is not None else self.config.temperature
        if self._client is None:
            return fallback or self._default_fallback(prompt)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.config.model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content  # type: ignore[index]
            if content is None:
                raise ValueError("LLM вернул пустой ответ.")
            return content.strip()
        except Exception:  # pragma: no cover - сетевые ошибки, отсутствие модели и т.п.
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
        os.environ.setdefault(key, value)

    os.environ[env_loaded_flag] = "1"


__all__ = ["LLMClient", "LLMConfig"]

from __future__ import annotations

import json
import os
from typing import Any
import httpx

from ..config.settings import settings


def _gigachat_client():
    # GigaChat provides an OpenAI-compatible API. We use openai SDK with custom base_url.
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("GIGACHAT_API_KEY") or settings.llm.api_key
    if not api_key:
        raise RuntimeError("GIGACHAT_API_KEY is not set")
    base_url = (
        os.getenv("GIGACHAT_BASE_URL")
        or settings.llm.base_url
        or "https://gigachat.devices.sberbank.ru/api/v1"
    )
    # Model name: prefer configured, else a sane default for GigaChat
    configured_model = settings.llm.model or ""
    model = configured_model if configured_model.lower().startswith("gigachat") else "GigaChat:latest"
    # Networking hardening
    timeout = float(os.getenv("GIGACHAT_TIMEOUT", "30"))
    insecure = os.getenv("GIGACHAT_INSECURE", "0") in ("1", "true", "yes")
    verify = False if insecure else True
    http_client = httpx.Client(timeout=timeout, verify=verify)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client, max_retries=2)
    return client, model


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    # Try raw
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def try_generate_code_agent_recommendation_llm(
    validation_report: dict, metafeatures: dict
) -> dict[str, Any] | None:
    """Attempt to get an LLM-generated preprocessing recipe; return None on failure."""
    # Guard on provider or env
    provider = settings.llm.provider
    has_gigakey = os.getenv("GIGACHAT_API_KEY") or (settings.llm.api_key and provider == "gigachat")
    if provider != "gigachat" and not has_gigakey:
        return None
    try:
        client, model = _gigachat_client()
        system_prompt = (
            "You are Data Agent — AetherML. Given validation_report and metafeatures, "
            "generate a Code Agent preprocessing recipe as strict JSON with fields: "
            "summary, priority, steps, example_pipeline_snippet, frameworks_recommended, "
            "rationale, estimated_complexity, confidence. Do not include any extra text."
        )
        user_prompt = json.dumps(
            {
                "instruction": "Return only JSON object, no prose.",
                "validation_report": validation_report,
                "metafeatures": metafeatures,
            },
            ensure_ascii=False,
        )
        # First attempt: with response_format
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback: without response_format (some providers don't support it)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
        content = resp.choices[0].message.content or "{}"
        data = _parse_json_loose(content) or {}
        # minimal shape validation
        if not isinstance(data, dict) or "summary" not in data or "steps" not in data:
            return None
        # ensure mandatory keys exist
        data.setdefault("frameworks_recommended", ["pandas", "scikit-learn"])
        data.setdefault("estimated_complexity", "medium")
        if "confidence" in data:
            try:
                data["confidence"] = float(data["confidence"])
            except Exception:
                data["confidence"] = 0.8
        else:
            data["confidence"] = 0.8
        return data
    except Exception:
        return None


def llm_healthcheck() -> dict[str, Any]:
    """Return diagnostics: provider, base_url, model, can_list_models, can_chat."""
    out: dict[str, Any] = {
        "provider": settings.llm.provider,
        "base_url": os.getenv("GIGACHAT_BASE_URL") or settings.llm.base_url or "",
        "model_configured": settings.llm.model,
        "https_proxy": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or "",
        "verify_ssl": "false" if os.getenv("GIGACHAT_INSECURE", "0") in ("1", "true", "yes") else "true",
    }
    try:
        client, model = _gigachat_client()
        out["resolved_model"] = model
        # try list models
        try:
            models = client.models.list()
            out["models_list_ok"] = True
            # include first few ids if present
            ids = []
            for m in getattr(models, "data", [])[:5]:
                mid = getattr(m, "id", None)
                if mid:
                    ids.append(mid)
            out["models_sample"] = ids
        except Exception as e:
            out["models_list_ok"] = False
            out["models_error"] = f"{e.__class__.__name__}: {e}"
        # try minimal chat (no response_format)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
                temperature=0.0,
                max_tokens=16,
            )
            content = resp.choices[0].message.content or ""
            out["chat_ok"] = True
            out["chat_content"] = content[:200]
        except Exception as e:
            out["chat_ok"] = False
            out["chat_error"] = f"{e.__class__.__name__}: {e}"
    except Exception as e:
        out["client_init_ok"] = False
        out["client_init_error"] = f"{e.__class__.__name__}: {e}"
    return out



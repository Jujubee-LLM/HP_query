from functools import lru_cache

from openai import OpenAI
from .settings import settings


def _client_kwargs(*, timeout_seconds: float, api_key: str, base_url: str | None) -> dict:
    base_url = (base_url or "").strip() or None
    kwargs: dict = {"api_key": api_key, "timeout": float(timeout_seconds)}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _embed_config() -> tuple[str, str | None, str]:
    api_key = (settings.embed_api_key or "").strip() or settings.openai_api_key
    base_url = (settings.embed_base_url or "").strip() or settings.openai_base_url
    model = (settings.embed_model or "").strip() or settings.openai_embed_model
    return api_key, base_url, model


def _chat_config() -> tuple[str, str | None, str]:
    api_key = (settings.chat_api_key or "").strip() or settings.openai_api_key
    base_url = (settings.chat_base_url or "").strip() or settings.openai_base_url
    model = (settings.chat_model or "").strip() or settings.openai_chat_model
    return api_key, base_url, model


@lru_cache(maxsize=1)
def get_embed_client() -> OpenAI:
    timeout = float(settings.openai_embed_timeout_seconds or settings.openai_timeout_seconds)
    api_key, base_url, _model = _embed_config()
    return OpenAI(**_client_kwargs(timeout_seconds=timeout, api_key=api_key, base_url=base_url))


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
    timeout = float(settings.openai_chat_timeout_seconds or settings.openai_timeout_seconds)
    api_key, base_url, _model = _chat_config()
    return OpenAI(**_client_kwargs(timeout_seconds=timeout, api_key=api_key, base_url=base_url))


def embed_texts(texts: list[str], *, timeout_seconds: float | None = None) -> list[list[float]]:
    api_key, base_url, model = _embed_config()
    client = (
        get_embed_client()
        if timeout_seconds is None
        else OpenAI(**_client_kwargs(timeout_seconds=float(timeout_seconds), api_key=api_key, base_url=base_url))
    )
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def chat_complete(system: str, user: str, *, model: str | None = None) -> str:
    client = get_chat_client()
    _api_key, _base_url, default_model = _chat_config()
    resp = client.chat.completions.create(
        model=(model or default_model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=int(settings.openai_chat_max_tokens),
    )
    return resp.choices[0].message.content or ""


def chat_complete_stream(system: str, user: str):
    client = get_chat_client()
    _api_key, _base_url, default_model = _chat_config()
    stream = client.chat.completions.create(
        model=default_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=int(settings.openai_chat_max_tokens),
        stream=True,
    )
    for event in stream:
        try:
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
        except Exception:
            content = None
        if content:
            yield content

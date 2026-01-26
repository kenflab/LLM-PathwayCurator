# LLM-PathwayCurator/src/llm_pathway_curator/backends.py
from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.request
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any


# -------------------------
# Env helpers
# -------------------------
def _getenv(*names: str, default: str | None = None) -> str | None:
    """
    Return the first non-empty env var among names; otherwise default.
    """
    for n in names:
        v = os.environ.get(n, None)
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        return s
    return default


def _try_parse_float(s: str) -> tuple[bool, float]:
    try:
        return True, float(s)
    except Exception:
        return False, 0.0


def _getfloat(*names: str, default: float) -> float:
    v = _getenv(*names, default=None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _getint(*names: str, default: int) -> int:
    v = _getenv(*names, default=None)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _gettimeout_pair(
    *,
    connect_names: tuple[str, ...],
    read_names: tuple[str, ...],
    legacy_names: tuple[str, ...],
    default_connect: float,
    default_read: float,
) -> tuple[float, float]:
    """
    Resolve (connect_timeout, read_timeout) from env with backward compatibility.

    Priority:
      1) explicit connect/read envs
      2) legacy single timeout envs (applies to both)
      3) defaults

    Robustness:
      - If an explicit connect/read env is present but NOT parseable as float,
        we treat it as "not explicitly set" for the purpose of legacy override.
    """
    # Read raw env strings first (so we can tell "present but invalid").
    c_raw = _getenv(*connect_names, default=None)
    r_raw = _getenv(*read_names, default=None)

    c_ok = False
    r_ok = False
    c = float(default_connect)
    r = float(default_read)

    if c_raw is not None:
        c_ok, c_val = _try_parse_float(c_raw)
        if c_ok:
            c = float(c_val)

    if r_raw is not None:
        r_ok, r_val = _try_parse_float(r_raw)
        if r_ok:
            r = float(r_val)

    legacy = _getenv(*legacy_names, default=None)
    if legacy is not None:
        l_ok, l_val = _try_parse_float(legacy)
        if l_ok:
            v = float(l_val)
            # only override if connect/read are not explicitly set (or explicitly invalid)
            if not c_ok:
                c = v
            if not r_ok:
                r = v

    # sanity
    c = max(0.5, float(c))
    r = max(1.0, float(r))
    return c, r


def get_backend_from_env(seed: int | None = None) -> BaseLLMBackend:
    """
    Create backend based on env.

    Accepted env keys (backward compatible):
      - BACKEND / LLMPATH_BACKEND / LPC_BACKEND
        values: openai|gemini|ollama|local|offline

    Notes:
      - We accept both LLMPATH_* and LPC_* prefixes for compatibility.
      - For keys that exist in both, LPC_* takes priority, then standard vendor env, then LLMPATH_*.
    """
    backend = (
        _getenv(
            "LPC_BACKEND",
            "BACKEND",
            "LLMPATH_BACKEND",
            default="ollama",
        )
        or "ollama"
    ).lower()

    seed_val = 42 if seed is None else int(seed)

    if backend == "openai":
        # prefer LPC_OPENAI_API_KEY, fallback to OPENAI_API_KEY, then LLMPATH_OPENAI_API_KEY
        api_key = _getenv("LPC_OPENAI_API_KEY", "OPENAI_API_KEY", "LLMPATH_OPENAI_API_KEY")
        if not api_key:
            raise KeyError(
                "Missing OpenAI API key. Set: "
                "LPC_OPENAI_API_KEY or OPENAI_API_KEY or LLMPATH_OPENAI_API_KEY"
            )
        return OpenAIBackend(
            api_key=api_key,
            model_name=_getenv(
                "LPC_OPENAI_MODEL",
                "OPENAI_MODEL",
                "LLMPATH_OPENAI_MODEL",
                default="gpt-4o",
            )
            or "gpt-4o",
            temperature=_getfloat("LPC_TEMPERATURE", "LLMPATH_TEMPERATURE", default=0.0),
            seed=seed_val,
        )

    if backend == "gemini":
        api_key = _getenv("LPC_GEMINI_API_KEY", "GEMINI_API_KEY", "LLMPATH_GEMINI_API_KEY")
        if not api_key:
            raise KeyError(
                "Missing Gemini API key. Set: "
                "LPC_GEMINI_API_KEY or GEMINI_API_KEY or LLMPATH_GEMINI_API_KEY"
            )
        return GeminiBackend(
            api_key=api_key,
            model_name=_getenv(
                "LPC_GEMINI_MODEL",
                "GEMINI_MODEL",
                "LLMPATH_GEMINI_MODEL",
                default="models/gemini-2.0-flash",
            )
            or "models/gemini-2.0-flash",
            temperature=_getfloat("LPC_TEMPERATURE", "LLMPATH_TEMPERATURE", default=0.0),
        )

    if backend == "ollama":
        return OllamaBackend(
            host=_getenv("LPC_OLLAMA_HOST", "OLLAMA_HOST", "LLMPATH_OLLAMA_HOST", default=None),
            model_name=_getenv(
                "LPC_OLLAMA_MODEL", "OLLAMA_MODEL", "LLMPATH_OLLAMA_MODEL", default=None
            ),
            temperature=_getfloat(
                "LPC_OLLAMA_TEMPERATURE", "LLMPATH_OLLAMA_TEMPERATURE", default=0.0
            ),
            timeout=_getfloat("LPC_OLLAMA_TIMEOUT", "LLMPATH_OLLAMA_TIMEOUT", default=120.0),
        )

    if backend in {"local", "offline"}:
        return LocalLLMBackend()

    raise ValueError(
        f"Unknown backend={backend!r} (expected: openai|gemini|ollama|local). "
        "Set: LPC_BACKEND or BACKEND or LLMPATH_BACKEND."
    )


class BaseLLMBackend(ABC):
    """
    Backend-agnostic LLM interface.

    Contract:
      - Input: prompt string
      - Output:
          - json_mode=False: a single string (free-form). On error, implementations MAY return
            a human-readable string or a standardized soft error JSON.
          - json_mode=True: MUST return either:
              (a) a valid JSON string parseable by json.loads, OR
              (b) a standardized soft error JSON string:
                  {"error": {"message": "...", "type": "...", "retryable": true/false}}
    """

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        json_mode = bool(kwargs.get("json_mode", False))
        return self.generate(str(prompt), json_mode=json_mode)

    def call(self, prompt: str, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)

    def chat(self, messages: Any, **kwargs: Any) -> str:
        prompt = ""
        try:
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    prompt = str(last.get("content", "") or "")
                else:
                    prompt = str(last)
            else:
                prompt = str(messages or "")
        except Exception:
            prompt = ""
        return self.invoke(prompt, **kwargs)

    # JSON helper names (optional)
    def chat_json(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(str(prompt), json_mode=True)

    def complete_json(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(str(prompt), json_mode=True)

    def generate_json(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(str(prompt), json_mode=True)

    def json(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(str(prompt), json_mode=True)


def _is_soft_error_json(s: str) -> bool:
    """
    Detect a standardized soft error JSON payload.

    Standard:
      {"error": {"message": "...", "type": "...", "retryable": true/false}}
    """
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    err = obj.get("error")
    return isinstance(err, dict) and isinstance(err.get("message", ""), str)


def _soft_error_json(
    message: str,
    *,
    err_type: str = "backend_error",
    retryable: bool = False,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "error": {"message": str(message), "type": str(err_type), "retryable": bool(retryable)}
    }
    if extra:
        # keep small + JSON-safe (caller responsibility)
        payload["error"]["extra"] = extra
    return json.dumps(payload, ensure_ascii=False)


def _extract_retryable_from_soft_error(s: str) -> tuple[bool, str]:
    """
    Returns (retryable, message) if s is a soft error JSON; otherwise (False, "").
    """
    try:
        obj = json.loads(s)
    except Exception:
        return False, ""
    if not isinstance(obj, dict):
        return False, ""
    err = obj.get("error")
    if not isinstance(err, dict):
        return False, ""
    msg = str(err.get("message", "") or "")
    retryable = bool(err.get("retryable", False))
    return retryable, msg


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1.0):
    """
    Exponential backoff retries for backend calls.

    Retries on:
      - retryable exceptions (heuristics)
      - plain-text soft errors (legacy): "OpenAI Error: ...", "Gemini Error: ...",
        "Ollama Error: ..."
      - standardized soft error JSON payloads: {"error": {...}}
      - json_mode parse failures (at most one retry)

    Notes:
      - json_mode is inferred from kwargs OR (self, prompt, json_mode) positional ABI.
      - In json_mode, we prefer standardized soft error JSON; non-JSON outputs are treated
        as parse failures and may be retried once.
      - If a backend returns None (contract violation), we normalize to a standardized soft error.
    """

    NON_RETRYABLE_PATTERNS = [
        "401",
        "403",
        "invalid_api_key",
        "unauthorized",
        "forbidden",
        "invalid_request",
        "model not found",
        "insufficient_quota",
        "billing",
        "response_format",
        "unsupported",
    ]

    RETRYABLE_HINTS = [
        "429",
        "rate limit",
        "ratelimit",
        "too many requests",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "api connection",
        "temporarily",
        "unavailable",
        "service unavailable",
        "overloaded",
    ]

    def _should_retry_error_text(err_text: str) -> bool:
        t = (err_text or "").lower()
        if any(p in t for p in NON_RETRYABLE_PATTERNS):
            return False
        if any(h in t for h in RETRYABLE_HINTS):
            return True
        return False

    def _infer_json_mode(args, kwargs) -> bool:
        if "json_mode" in kwargs:
            return bool(kwargs.get("json_mode", False))
        # Positional ABI: (self, prompt, json_mode=False)
        # args[0]=self, args[1]=prompt, args[2]=json_mode (optional)
        if len(args) >= 3 and isinstance(args[2], bool):
            return bool(args[2])
        return False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            json_mode = _infer_json_mode(args, kwargs)

            parsefail_used = False  # allow at most one retry for JSON parse failures in json_mode

            while True:
                try:
                    result = func(*args, **kwargs)

                    # Normalize contract violation (None) into a standardized soft error JSON.
                    if result is None:
                        result = _soft_error_json(
                            "Backend returned None (contract violation).",
                            err_type="backend_contract_violation",
                            retryable=True,
                        )

                    retryable = False

                    if isinstance(result, str):
                        s = result.strip()

                        # standardized soft error JSON
                        if s.startswith("{") and s.endswith("}") and _is_soft_error_json(s):
                            rflag, msg = _extract_retryable_from_soft_error(s)
                            retryable = bool(rflag) or _should_retry_error_text(msg)

                        # legacy plain-text soft errors
                        elif s.startswith(("Gemini Error:", "OpenAI Error:", "Ollama Error:")):
                            retryable = _should_retry_error_text(s)

                        # json_mode requested but got invalid JSON -> allow one retry
                        elif json_mode:
                            try:
                                json.loads(s)
                            except Exception:
                                retryable = not parsefail_used
                                parsefail_used = True

                    if retryable:
                        if attempt >= retries:
                            return result
                        sleep = backoff_in_seconds * (2**attempt)
                        sleep *= 1.0 + random.uniform(-0.1, 0.1)
                        logging.debug("LLM retry. attempt=%d sleep=%.2fs", attempt + 1, sleep)
                        time.sleep(max(0.0, sleep))
                        attempt += 1
                        continue

                    return result

                except Exception as e:
                    msg = str(e)
                    if not _should_retry_error_text(msg):
                        raise
                    if attempt >= retries:
                        raise
                    sleep = backoff_in_seconds * (2**attempt)
                    sleep *= 1.0 + random.uniform(-0.1, 0.1)
                    logging.debug(
                        "LLM retry (exception). attempt=%d sleep=%.2fs err=%s",
                        attempt + 1,
                        sleep,
                        msg,
                    )
                    time.sleep(max(0.0, sleep))
                    attempt += 1

        return wrapper

    return decorator


class GeminiBackend(BaseLLMBackend):
    def __init__(
        self, api_key: str, model_name: str = "models/gemini-2.0-flash", temperature: float = 0.0
    ):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("Please install google-generativeai to use GeminiBackend.") from e

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = float(temperature)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                generation_config = {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                }
            else:
                generation_config = {"temperature": self.temperature}

            response = self.model.generate_content(prompt, generation_config=generation_config)
            txt = response.text

            # Enforce json_mode contract: return valid JSON or soft error JSON
            if json_mode:
                try:
                    json.loads(txt)
                    return txt
                except Exception:
                    return _soft_error_json(
                        f"Gemini returned non-JSON in json_mode: {str(txt)[:200]}",
                        err_type="gemini_non_json",
                        retryable=True,
                    )

            return txt

        except Exception as e:
            msg = str(e)
            retryable = any(
                h in msg.lower() for h in ["429", "timeout", "503", "unavailable", "rate limit"]
            )
            if json_mode:
                return _soft_error_json(msg, err_type="gemini_error", retryable=retryable)
            return f"Gemini Error: {msg}"


class OpenAIBackend(BaseLLMBackend):
    def __init__(
        self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.0, seed: int = 42
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai package to use OpenAIBackend.") from e

        self.client = OpenAI(api_key=api_key)
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.seed = int(seed)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        def _call(with_seed: bool) -> str:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if with_seed:
                kwargs["seed"] = self.seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        first_exc: Exception | None = None
        try:
            txt = _call(with_seed=True)
        except Exception as e:
            first_exc = e
            try:
                txt = _call(with_seed=False)
            except Exception as e2:
                err = e2 if e2 is not None else first_exc
                msg = str(err)

                retryable = any(
                    h in msg.lower()
                    for h in [
                        "429",
                        "timeout",
                        "503",
                        "unavailable",
                        "rate limit",
                        "too many requests",
                        "overloaded",
                    ]
                )
                if json_mode:
                    return _soft_error_json(msg, err_type="openai_error", retryable=retryable)
                return f"OpenAI Error: {msg}"

        # Enforce json_mode contract post-hoc as well
        if json_mode:
            try:
                json.loads(txt)
                return txt
            except Exception:
                return _soft_error_json(
                    f"OpenAI returned non-JSON in json_mode: {str(txt)[:200]}",
                    err_type="openai_non_json",
                    retryable=True,
                )

        return txt


class OllamaBackend(BaseLLMBackend):
    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ):
        host = (
            host
            if host is not None
            else os.environ.get("LLMPATH_OLLAMA_HOST", "http://ollama:11434")
        )
        model_name = (
            model_name
            if model_name is not None
            else os.environ.get("LLMPATH_OLLAMA_MODEL", "llama3.1:8b")
        )
        if temperature is None:
            temperature = float(os.environ.get("LLMPATH_OLLAMA_TEMPERATURE", "0.0"))

        # Backward compatibility:
        # - timeout arg (if provided) applies to both connect/read
        # - legacy env: LLMPATH_OLLAMA_TIMEOUT / LPC_OLLAMA_TIMEOUT
        # New envs:
        # - LLMPATH_OLLAMA_CONNECT_TIMEOUT / LPC_OLLAMA_CONNECT_TIMEOUT
        # - LLMPATH_OLLAMA_READ_TIMEOUT / LPC_OLLAMA_READ_TIMEOUT
        if timeout is not None:
            connect_timeout = float(timeout)
            read_timeout = float(timeout)
        else:
            connect_timeout, read_timeout = _gettimeout_pair(
                connect_names=("LPC_OLLAMA_CONNECT_TIMEOUT", "LLMPATH_OLLAMA_CONNECT_TIMEOUT"),
                read_names=("LPC_OLLAMA_READ_TIMEOUT", "LLMPATH_OLLAMA_READ_TIMEOUT"),
                legacy_names=("LPC_OLLAMA_TIMEOUT", "LLMPATH_OLLAMA_TIMEOUT"),
                default_connect=10.0,
                default_read=120.0,
            )

        self.host = str(host).rstrip("/")
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Ollama backend call.

        Contract:
          - Always returns a string.
          - If json_mode=True: returns a valid JSON string OR standardized soft error JSON.
          - If json_mode=False: returns free-form text (best-effort), OR legacy "Ollama Error: ...".

        Notes:
          - urllib timeout is a SINGLE float seconds (cannot separate connect/read).
          - We keep connect_timeout for documentation/metadata only.
        """
        url = f"{self.host}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": str(prompt),
            "stream": False,
            "options": {"temperature": float(self.temperature)},
        }
        if json_mode:
            # In Ollama, "format": "json" nudges the model to emit JSON in `response`.
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")

        # Adaptive read-timeout escalation (deterministic ladder).
        max_escalations = _getint(
            "LPC_OLLAMA_TIMEOUT_ESCALATIONS",
            "LLMPATH_OLLAMA_TIMEOUT_ESCALATIONS",
            default=2,
        )
        factor = _getfloat(
            "LPC_OLLAMA_TIMEOUT_FACTOR",
            "LLMPATH_OLLAMA_TIMEOUT_FACTOR",
            default=2.0,
        )
        max_read = _getfloat(
            "LPC_OLLAMA_READ_TIMEOUT_MAX",
            "LLMPATH_OLLAMA_READ_TIMEOUT_MAX",
            default=1800.0,
        )

        read_timeout = float(self.read_timeout)
        escalations_used = 0

        while True:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=float(read_timeout)) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")

                obj = json.loads(raw) if raw.strip() else {}
                if not isinstance(obj, dict):
                    obj = {}

                content = (obj.get("response") or "").strip()

                if not json_mode:
                    # free-form mode: return as-is
                    return content

                # json_mode: enforce strict contract
                try:
                    json.loads(content)
                    return content
                except Exception:
                    return _soft_error_json(
                        (f"Ollama(/api/generate) returned non-JSON in json_mode: {content[:200]}"),
                        err_type="ollama_non_json",
                        retryable=True,
                        extra={
                            "connect_timeout_ignored_by_urllib": float(self.connect_timeout),
                            "read_timeout": float(read_timeout),
                            "escalations_used": int(escalations_used),
                        },
                    )

            except Exception as e:
                msg = str(e)
                is_timeout = ("timed out" in msg.lower()) or ("timeout" in msg.lower())

                if is_timeout and escalations_used < int(max_escalations):
                    escalations_used += 1
                    read_timeout = min(float(max_read), float(read_timeout) * float(factor))
                    continue

                retryable = any(
                    h in msg.lower()
                    for h in ["429", "timeout", "503", "unavailable", "rate limit", "overloaded"]
                )

                if json_mode:
                    return _soft_error_json(
                        msg,
                        err_type="ollama_error",
                        retryable=retryable,
                        extra={
                            "connect_timeout_ignored_by_urllib": float(self.connect_timeout),
                            "read_timeout": float(read_timeout),
                            "escalations_used": int(escalations_used),
                        },
                    )

                # legacy text error (keeps retry_with_backoff compatibility)
                return f"Ollama Error: {msg}"


class LocalLLMBackend(BaseLLMBackend):
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        if json_mode:
            return _soft_error_json(
                "Local backend not implemented", err_type="local_pending", retryable=False
            )
        return "Local backend not implemented"

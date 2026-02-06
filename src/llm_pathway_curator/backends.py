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
    Return the first non-empty environment variable among candidates.

    Parameters
    ----------
    *names
        Environment variable names to check in order.
    default
        Fallback value when none of the candidate variables is set to a non-empty
        value.

    Returns
    -------
    str or None
        The first non-empty value found (stripped), otherwise `default`.

    Notes
    -----
    - Values are stripped; empty strings are treated as missing.
    - Returned type is always `str` when a value is found.
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
    """
    Parse a string as float without raising.

    Parameters
    ----------
    s
        Input string.

    Returns
    -------
    tuple[bool, float]
        `(ok, value)` where `ok` indicates whether parsing succeeded.
        On failure, `value` is `0.0`.
    """
    try:
        return True, float(s)
    except Exception:
        return False, 0.0


def _getfloat(*names: str, default: float) -> float:
    """
    Read a float value from environment variables with fallback.

    Parameters
    ----------
    *names
        Environment variable names to check in order.
    default
        Fallback value when not set or not parseable.

    Returns
    -------
    float
        Parsed float value, or `default` when missing/invalid.
    """
    v = _getenv(*names, default=None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _getint(*names: str, default: int) -> int:
    """
    Read an integer value from environment variables with fallback.

    Parameters
    ----------
    *names
        Environment variable names to check in order.
    default
        Fallback value when not set or not parseable.

    Returns
    -------
    int
        Parsed integer value, or `default` when missing/invalid.
    """
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
    Resolve (connect_timeout, read_timeout) from environment variables.

    Parameters
    ----------
    connect_names
        Environment variable names for connect timeout.
    read_names
        Environment variable names for read timeout.
    legacy_names
        Legacy single-timeout variable names. If set and parseable, the value may
        apply to both connect/read depending on override rules.
    default_connect
        Default connect timeout in seconds.
    default_read
        Default read timeout in seconds.

    Returns
    -------
    tuple[float, float]
        `(connect_timeout, read_timeout)` in seconds.

    Notes
    -----
    Priority order:
    1) Explicit connect/read envs
    2) Legacy single timeout env (applies to both, unless explicit connect/read was
       successfully parsed)
    3) Defaults

    Robustness:
    - If an explicit connect/read env exists but is not parseable as float, it is
      treated as "not explicitly set" for legacy override purposes.
    - A lower bound is applied: connect >= 0.5s, read >= 1.0s.
    """
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
    Create an LLM backend based on environment variables.

    Parameters
    ----------
    seed
        Optional seed for backends that support seeded generation.

    Returns
    -------
    BaseLLMBackend
        Instantiated backend.

    Raises
    ------
    KeyError
        If a required API key is missing for the selected backend.
    ValueError
        If the backend name is unknown.

    Notes
    -----
    Backend selection envs (first non-empty wins):
    - LPC_BACKEND, BACKEND, LLMPATH_BACKEND

    Supported backends:
    - "openai": uses OpenAI chat completions
    - "gemini": uses Google Generative AI
    - "ollama": uses Ollama HTTP API
    - "local" / "offline": stub backend (no real generation)

    Compatibility:
    - Both "LLMPATH_*" and "LPC_*" prefixes are accepted for most settings.
    - For overlapping keys, LPC_* is preferred over vendor env, then LLMPATH_*.
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
        timeout_raw = _getenv("LPC_OLLAMA_TIMEOUT", "LLMPATH_OLLAMA_TIMEOUT", default=None)
        timeout_val = None
        if timeout_raw is not None:
            try:
                timeout_val = float(timeout_raw)
            except Exception:
                timeout_val = None
        return OllamaBackend(
            host=_getenv("LPC_OLLAMA_HOST", "OLLAMA_HOST", "LLMPATH_OLLAMA_HOST", default=None),
            model_name=_getenv(
                "LPC_OLLAMA_MODEL", "OLLAMA_MODEL", "LLMPATH_OLLAMA_MODEL", default=None
            ),
            temperature=_getfloat(
                "LPC_OLLAMA_TEMPERATURE", "LLMPATH_OLLAMA_TEMPERATURE", default=0.0
            ),
            timeout=timeout_val,
        )

    if backend in {"local", "offline"}:
        return LocalLLMBackend()

    raise ValueError(
        f"Unknown backend={backend!r} (expected: openai|gemini|ollama|local|offline). "
        "Set: LPC_BACKEND or BACKEND or LLMPATH_BACKEND."
    )


class BaseLLMBackend(ABC):
    """
    Backend-agnostic LLM interface.

    This class defines a minimal contract for generating text or JSON strings.

    Contract
    --------
    Input
        prompt : str

    Output
        json_mode=False
            Returns a single string (free-form). Implementations may return a
            human-readable error string on failure.
        json_mode=True
            Must return either:
            (a) a valid JSON string parseable by `json.loads`, or
            (b) a standardized soft error JSON string:
                {"error": {"message": "...", "type": "...", "retryable": true/false}}

    Notes
    -----
    Convenience aliases are provided (`invoke`, `call`, `complete`, `chat`, and
    `*_json` helpers). Subclasses should implement `generate`.
    """

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Generate a completion for a given prompt.

        Parameters
        ----------
        prompt
            Input prompt string.
        json_mode
            If True, the backend must return a JSON string (or a standardized soft
            error JSON). If False, free-form text is allowed.

        Returns
        -------
        str
            Model output. See class-level contract for json_mode behavior.

        Raises
        ------
        NotImplementedError
            If the backend does not implement this method.
        """
        raise NotImplementedError

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the backend with a prompt (alias for `generate`).

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments. `json_mode` is recognized.

        Returns
        -------
        str
            Model output string.
        """
        json_mode = bool(kwargs.get("json_mode", False))
        return self.generate(str(prompt), json_mode=json_mode)

    def call(self, prompt: str, **kwargs: Any) -> str:
        """
        Alias for `invoke`.

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments.

        Returns
        -------
        str
            Model output string.
        """
        return self.invoke(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Alias for `invoke`.

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments.

        Returns
        -------
        str
            Model output string.
        """
        return self.invoke(prompt, **kwargs)

    def chat(self, messages: Any, **kwargs: Any) -> str:
        """
        Best-effort chat wrapper.

        Parameters
        ----------
        messages
            Chat-like messages. Typically a list of dicts or strings. If a list is
            provided, the last element's "content" field (if dict) is used as prompt.
        **kwargs
            Optional keyword arguments passed to `invoke`.

        Returns
        -------
        str
            Model output string.

        Notes
        -----
        This is intentionally lightweight and is not a full chat protocol
        implementation. It extracts a prompt and delegates to `invoke`.
        """
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
        """
        Generate JSON output from a prompt (chat-style helper).

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments (ignored except for future compatibility).

        Returns
        -------
        str
            JSON string or standardized soft error JSON string.
        """
        return self.generate(str(prompt), json_mode=True)

    def complete_json(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate JSON output from a prompt (completion-style helper).

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments (ignored except for future compatibility).

        Returns
        -------
        str
            JSON string or standardized soft error JSON string.
        """
        return self.generate(str(prompt), json_mode=True)

    def generate_json(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate JSON output from a prompt (explicit helper).

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments (ignored except for future compatibility).

        Returns
        -------
        str
            JSON string or standardized soft error JSON string.
        """
        return self.generate(str(prompt), json_mode=True)

    def json(self, prompt: str, **kwargs: Any) -> str:
        """
        Alias for JSON generation helpers.

        Parameters
        ----------
        prompt
            Input prompt string.
        **kwargs
            Optional keyword arguments.

        Returns
        -------
        str
            JSON string or standardized soft error JSON string.
        """
        return self.generate(str(prompt), json_mode=True)


def _is_soft_error_json(s: str) -> bool:
    """
    Detect whether a string is a standardized soft error JSON payload.

    Parameters
    ----------
    s
        Candidate JSON string.

    Returns
    -------
    bool
        True if `s` parses to a dict with key "error" containing at least a
        string "message" field, otherwise False.

    Notes
    -----
    Standard payload shape:
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
    """
    Build a standardized soft error JSON payload.

    Parameters
    ----------
    message
        Human-readable error message.
    err_type
        Error type identifier.
    retryable
        Whether the error is considered retryable.
    extra
        Optional extra metadata to embed under "error.extra". Caller should keep
        this small and JSON-serializable.

    Returns
    -------
    str
        JSON string with the standardized error schema.
    """
    payload: dict[str, Any] = {
        "error": {"message": str(message), "type": str(err_type), "retryable": bool(retryable)}
    }
    if extra:
        # keep small + JSON-safe (caller responsibility)
        payload["error"]["extra"] = extra
    return json.dumps(payload, ensure_ascii=False)


def _extract_retryable_from_soft_error(s: str) -> tuple[bool, str]:
    """
    Extract retryability and message from a soft error JSON string.

    Parameters
    ----------
    s
        Candidate JSON string.

    Returns
    -------
    tuple[bool, str]
        `(retryable, message)` if `s` matches the standardized soft error schema.
        Otherwise returns `(False, "")`.
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
    Decorator factory for exponential backoff retries on backend calls.

    Parameters
    ----------
    retries
        Maximum number of retry attempts (not counting the initial call).
    backoff_in_seconds
        Base backoff duration in seconds. Sleep time grows as:
        `backoff_in_seconds * 2**attempt`, with small jitter.

    Returns
    -------
    callable
        A decorator that wraps a function and retries under certain conditions.

    Retry conditions
    ----------------
    - Retryable exceptions inferred by message heuristics (status/keywords).
    - Legacy plain-text soft errors:
      "OpenAI Error: ...", "Gemini Error: ...", "Ollama Error: ..."
    - Standardized soft error JSON payloads:
      {"error": {"message": "...", "type": "...", "retryable": ...}}
    - When json_mode=True: invalid JSON outputs are treated as parse failures
      and retried at most once.

    Notes
    -----
    `json_mode` is inferred from kwargs (`json_mode=`) or from positional ABI:
    (self, prompt, json_mode=False) when present.
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
    """
    Google Gemini backend via `google-generativeai`.

    Parameters
    ----------
    api_key
        Gemini API key.
    model_name
        Gemini model identifier (e.g., "models/gemini-2.0-flash").
    temperature
        Sampling temperature.

    Notes
    -----
    - In json_mode, response is requested with MIME type "application/json" and
      validated. Non-JSON output is converted to standardized soft error JSON.
    """

    def __init__(
        self, api_key: str, model_name: str = "models/gemini-2.0-flash", temperature: float = 0.0
    ):
        """
        Initialize the Gemini backend.

        Parameters
        ----------
        api_key
            Gemini API key.
        model_name
            Gemini model identifier.
        temperature
            Sampling temperature.

        Raises
        ------
        ImportError
            If `google-generativeai` is not installed.
        """
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("Please install google-generativeai to use GeminiBackend.") from e

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = float(temperature)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Generate a completion using Gemini.

        Parameters
        ----------
        prompt
            Input prompt string.
        json_mode
            If True, attempts to enforce JSON output and validates with `json.loads`.

        Returns
        -------
        str
            Free-form text (json_mode=False), or a JSON string / standardized soft
            error JSON (json_mode=True).
        """
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
    """
    OpenAI backend using the `openai` Python SDK (chat completions).

    Parameters
    ----------
    api_key
        OpenAI API key.
    model_name
        Model name (e.g., "gpt-4o").
    temperature
        Sampling temperature.
    seed
        Seed used when supported by the API/model. If seeding fails, a fallback
        call without seed is attempted.

    Notes
    -----
    - In json_mode, `response_format={"type": "json_object"}` is used and output is
      validated. Non-JSON output is converted to standardized soft error JSON.
    """

    def __init__(
        self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.0, seed: int = 42
    ):
        """
        Initialize the OpenAI backend.

        Parameters
        ----------
        api_key
            OpenAI API key.
        model_name
            Model name.
        temperature
            Sampling temperature.
        seed
            Seed value for deterministic sampling when supported.

        Raises
        ------
        ImportError
            If the `openai` package is not installed.
        """
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
        """
        Generate a completion using OpenAI chat completions.

        Parameters
        ----------
        prompt
            Input prompt string.
        json_mode
            If True, requests JSON object output and validates with `json.loads`.

        Returns
        -------
        str
            Free-form text (json_mode=False), or a JSON string / standardized soft
            error JSON (json_mode=True).

        Notes
        -----
        If the seeded call fails, a second call without seed is attempted.
        """

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
    """
    Ollama backend using HTTP API (`/api/generate`) via urllib.

    Parameters
    ----------
    host
        Ollama server base URL (e.g., "http://ollama:11434").
    model_name
        Ollama model name (e.g., "llama3.1:8b").
    temperature
        Sampling temperature.
    timeout
        Legacy single timeout (seconds) applied to both connect/read timeouts.

    Notes
    -----
    - urllib accepts a single timeout value. This implementation stores both
      connect/read timeouts but uses read_timeout for urllib's timeout.
    - In json_mode, payload includes "format": "json" and output is validated.
      Non-JSON output is converted to standardized soft error JSON.
    """

    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ):
        """
        Initialize the Ollama backend.

        Parameters
        ----------
        host
            Base URL for Ollama server. If None, falls back to env defaults.
        model_name
            Model name. If None, falls back to env defaults.
        temperature
            Sampling temperature. If None, falls back to env default.
        timeout
            Legacy single timeout applied to both connect/read.

        Notes
        -----
        Timeout resolution supports:
        - New envs:
          LPC_OLLAMA_CONNECT_TIMEOUT / LLMPATH_OLLAMA_CONNECT_TIMEOUT
          LPC_OLLAMA_READ_TIMEOUT / LLMPATH_OLLAMA_READ_TIMEOUT
        - Legacy env:
          LPC_OLLAMA_TIMEOUT / LLMPATH_OLLAMA_TIMEOUT
        """
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

    @retry_with_backoff(retries=5)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Generate a completion using Ollama `/api/generate`.

        Parameters
        ----------
        prompt
            Input prompt string.
        json_mode
            If True, requests JSON output and validates with `json.loads`.

        Returns
        -------
        str
            Free-form text (json_mode=False), or a JSON string / standardized soft
            error JSON (json_mode=True).

        Notes
        -----
        - Adaptive read-timeout escalation is applied on timeout errors:
          `read_timeout *= factor` up to a max, for a limited number of escalations.
        - connect_timeout is stored for metadata/documentation only and is not used by
          urllib (single-timeout limitation).
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
    """
    Local/offline backend stub.

    This backend does not perform real generation. It exists to support offline
    workflows and testing paths.

    Notes
    -----
    - In json_mode, returns a standardized soft error JSON payload.
    - In text mode, returns a human-readable placeholder string.
    """

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Return a placeholder response (local/offline stub).

        Parameters
        ----------
        prompt
            Input prompt string (ignored).
        json_mode
            If True, returns standardized soft error JSON.

        Returns
        -------
        str
            Placeholder text or standardized soft error JSON.
        """
        if json_mode:
            return _soft_error_json(
                "Local backend not implemented", err_type="local_pending", retryable=False
            )
        return "Local backend not implemented"

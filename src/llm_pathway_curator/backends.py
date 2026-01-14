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


class BaseLLMBackend(ABC):
    """
    Backend-agnostic LLM interface.

    Contract:
      - Input: prompt string
      - Output: a single string (free-form or JSON text)
    """

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError


def _is_soft_error_json(s: str) -> bool:
    """
    Detect a standardized soft error JSON payload.

    We standardize on:
      {"error": {"message": "...", "type": "...", "retryable": true/false}}
    """
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    if "error" not in obj:
        return False
    err = obj.get("error")
    return isinstance(err, dict) and isinstance(err.get("message", ""), str)


def _soft_error_json(
    message: str, *, err_type: str = "backend_error", retryable: bool = False
) -> str:
    return json.dumps({"error": {"message": message, "type": err_type, "retryable": retryable}})


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1.0):
    """
    Exponential backoff retries for backend calls.

    Retries on:
      - retryable exceptions (heuristics)
      - "soft error" string payloads (e.g., "OpenAI Error: ...")
      - standardized soft error JSON payloads: {"error": {...}}
    """

    NON_RETRYABLE_PATTERNS = [
        "401",
        "403",
        "invalid_api_key",
        "unauthorized",
        "forbidden",
        "400",
        "bad request",
        "invalid_request",
        "model not found",
        "insufficient_quota",
        "quota",
        "billing",
    ]

    RETRYABLE_HINTS = [
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "unavailable",
    ]

    def _should_retry_error_text(err_text: str) -> bool:
        t = (err_text or "").lower()
        if any(p in t for p in NON_RETRYABLE_PATTERNS):
            return False
        if any(h in t for h in RETRYABLE_HINTS):
            return True
        return False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            json_mode = bool(kwargs.get("json_mode", False))
            if "json_mode" not in kwargs and len(args) >= 3:
                json_mode = bool(args[2])

            parsefail_used = False  # allow at most one retry for JSON parse failures

            while True:
                try:
                    result = func(*args, **kwargs)

                    retryable = False
                    if result is None:
                        retryable = True

                    elif isinstance(result, str):
                        s = result.strip()

                        # plain-text soft errors
                        if s.startswith(("Gemini Error:", "OpenAI Error:", "Ollama Error:")):
                            retryable = _should_retry_error_text(s)

                        # standardized JSON soft error
                        elif json_mode and s.startswith("{") and s.endswith("}"):
                            if _is_soft_error_json(s):
                                try:
                                    err = json.loads(s)["error"]
                                    retryable = bool(
                                        err.get("retryable", False)
                                    ) or _should_retry_error_text(err.get("message", ""))
                                except Exception:
                                    retryable = False
                            else:
                                # json_mode requested but got unparsable/invalid JSON
                                # -> allow one retry
                                try:
                                    json.loads(s)
                                except Exception:
                                    retryable = not parsefail_used
                                    parsefail_used = True

                    if retryable:
                        if x == retries:
                            return result
                        sleep = backoff_in_seconds * (2**x)
                        sleep *= 1.0 + random.uniform(-0.1, 0.1)
                        logging.debug("LLM retry. attempt=%d sleep=%.2fs", x + 1, sleep)
                        time.sleep(max(0.0, sleep))
                        x += 1
                        continue

                    return result

                except Exception as e:
                    msg = str(e)
                    if not _should_retry_error_text(msg):
                        raise
                    if x == retries:
                        raise
                    sleep = backoff_in_seconds * (2**x)
                    sleep *= 1.0 + random.uniform(-0.1, 0.1)
                    logging.debug(
                        "LLM retry (exception). attempt=%d sleep=%.2fs err=%s", x + 1, sleep, msg
                    )
                    time.sleep(max(0.0, sleep))
                    x += 1

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
            return response.text
        except Exception as e:
            if json_mode:
                # Mark retryable conservatively only if it looks transient
                msg = str(e)
                retryable = any(
                    h in msg.lower() for h in ["429", "timeout", "503", "unavailable", "rate limit"]
                )
                return _soft_error_json(msg, err_type="gemini_error", retryable=retryable)
            return f"Gemini Error: {e}"


class OpenAIBackend(BaseLLMBackend):
    def __init__(
        self, api_key: str, model_name: str = "gpt-4o", temperature: float = 0.0, seed: int = 42
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install openai package to use OpenAIBackend.") from e

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = float(temperature)
        self.seed = int(seed)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        def _call(with_seed: bool) -> str:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if with_seed:
                kwargs["seed"] = self.seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        first_exc = None
        try:
            return _call(with_seed=True)
        except Exception as e:
            first_exc = e
            try:
                return _call(with_seed=False)
            except Exception as e2:
                err = e2 if e2 is not None else first_exc
                if json_mode:
                    msg = str(err)
                    retryable = any(
                        h in msg.lower()
                        for h in ["429", "timeout", "503", "unavailable", "rate limit"]
                    )
                    return _soft_error_json(msg, err_type="openai_error", retryable=retryable)
                return f"OpenAI Error: {err}"


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
        if timeout is None:
            timeout = float(os.environ.get("LLMPATH_OLLAMA_TIMEOUT", "120"))

        self.host = str(host).rstrip("/")
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)

            content = ""
            if isinstance(obj, dict):
                msg = obj.get("message") or {}
                if isinstance(msg, dict):
                    content = (msg.get("content") or "").strip()
            if not content and isinstance(obj, dict):
                content = (obj.get("response") or "").strip()

            if json_mode:
                s = (content or "").strip()
                try:
                    json.loads(s)
                    return s
                except Exception:
                    return _soft_error_json(
                        f"Ollama returned non-JSON in json_mode: {s[:200]}",
                        err_type="ollama_non_json",
                        retryable=False,
                    )

            return content if content else raw.strip()

        except Exception as e:
            if json_mode:
                msg = str(e)
                retryable = any(
                    h in msg.lower() for h in ["429", "timeout", "503", "unavailable", "rate limit"]
                )
                return _soft_error_json(msg, err_type="ollama_error", retryable=retryable)
            return f"Ollama Error: {e}"


class LocalLLMBackend(BaseLLMBackend):
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        if json_mode:
            return _soft_error_json(
                "Local backend not implemented", err_type="local_pending", retryable=False
            )
        return "Local backend not implemented"

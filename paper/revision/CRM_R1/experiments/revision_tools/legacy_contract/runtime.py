"""Append-only attempts; exact successful responses only are cacheable."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .checks import apply_review, incomplete, prepare
from .models import ModelConfig, PromptPolicy, Review
from .prompt import canonical_json, check_references, digest, make_request, strict_json


class TechnicalError(Exception):
    def __init__(self, code: str, message: str, *, retryable=False, raw=b""):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.raw = raw


@dataclass
class WireResponse:
    raw: bytes
    observations: dict = field(default_factory=dict)


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_new(path: Path, value) -> None:
    """Publish a complete local record atomically, refusing existing destinations."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".record-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Exclusive destination; no overwrite, including on resume.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_response(raw: bytes, request: dict) -> Review:
    try:
        wire = strict_json(raw)
        expected_model = request["envelope"]["model_config"]["model"]
        if not isinstance(wire, dict) or wire.get("model") != expected_model:
            raise ValueError("Missing or wrong response model")
        if wire.get("error") or wire.get("done") is not True or wire.get("done_reason") != "stop":
            raise ValueError("Error, incomplete generation, or truncated generation")
        if not isinstance(wire.get("response"), str):
            raise ValueError("Missing response string")
        review = Review.model_validate(strict_json(wire["response"]))
        check_references(review, request["envelope"]["payload"])
        return review
    except (ValueError, TypeError, KeyError, UnicodeError, ValidationError) as exc:
        raise TechnicalError("RESPONSE_SCHEMA_INVALID", str(exc), raw=raw) from exc


class AttemptStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self, request: dict):
        key = request["key"]
        if digest(request["envelope"]) != key:
            raise TechnicalError("REQUEST_INTEGRITY_ERROR", "Request key mismatch")
        folder = self.root / key
        folder.mkdir(exist_ok=True)
        with (folder / ".writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TechnicalError(
                    "CONCURRENT_WRITER", "Another process owns this request"
                ) from exc
            try:
                record = folder / "request.json"
                if record.exists():
                    if strict_json(record.read_bytes()) != request:
                        raise TechnicalError("CACHE_INTEGRITY_ERROR", "Stored request differs")
                else:
                    write_new(record, request)
                yield folder
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def successful(self, folder: Path, request: dict) -> Review | None:
        marker = folder / "SUCCESS.json"
        if not marker.exists():
            # A process may stop after committing a successful attempt but before its marker.
            # Reuse that first valid success, never resample an already completed judgment.
            for path in sorted(folder.glob("attempt-*.result.json")):
                try:
                    prior = strict_json(path.read_bytes())
                    if prior.get("status") != "SUCCEEDED":
                        continue
                    index = int(path.name.split("-")[1].split(".")[0])
                    if prior.get("attempt") != index or prior.get("request_key") != request["key"]:
                        raise ValueError("Orphan success has wrong request or attempt")
                    raw = base64.b64decode(prior["raw_response_base64"], validate=True)
                    if hashlib.sha256(raw).hexdigest() != prior["raw_response_sha256"]:
                        raise ValueError("Orphan raw response digest mismatch")
                    review = parse_response(raw, request)
                    if review.model_dump() != prior["parsed_review"]:
                        raise ValueError("Orphan parsed response mismatch")
                    write_new(
                        marker,
                        {
                            "key": request["key"],
                            "attempt": index,
                            "record_sha256": digest(prior),
                        },
                    )
                    return review
                except (ValueError, KeyError, TypeError, TechnicalError) as exc:
                    raise TechnicalError("CACHE_INTEGRITY_ERROR", str(exc)) from exc
            return None
        try:
            rec = strict_json(marker.read_bytes())
            if set(rec) != {"key", "attempt", "record_sha256"} or rec["key"] != request["key"]:
                raise ValueError("Invalid cache success marker")
            if type(rec["attempt"]) is not int or rec["attempt"] < 1:
                raise ValueError("Invalid cache attempt index")
            result = strict_json(
                (folder / f"attempt-{rec['attempt']:06d}.result.json").read_bytes()
            )
            if result["status"] != "SUCCEEDED" or digest(result) != rec["record_sha256"]:
                raise ValueError("Cached attempt status or digest mismatch")
            if result["request_key"] != request["key"]:
                raise ValueError("Cached attempt belongs to a different request")
            raw = base64.b64decode(result["raw_response_base64"], validate=True)
            if hashlib.sha256(raw).hexdigest() != result["raw_response_sha256"]:
                raise ValueError("Cached raw response hash mismatch")
            review = parse_response(raw, request)
            if review.model_dump() != result["parsed_review"]:
                raise ValueError("Cached parsed response differs from raw response")
            return review
        except (OSError, ValueError, KeyError, TypeError, TechnicalError) as exc:
            raise TechnicalError("CACHE_INTEGRITY_ERROR", str(exc)) from exc

    def recover_interrupted(self, folder: Path, request: dict) -> None:
        for started in sorted(folder.glob("attempt-*.started.json")):
            finished = started.with_name(started.name.replace(".started.", ".result."))
            if not finished.exists():
                write_new(
                    finished,
                    {
                        "status": "INTERRUPTED",
                        "request_key": request["key"],
                        "ended_utc": now(),
                        "error_code": "INTERRUPTED_BEFORE_RESULT_COMMIT",
                    },
                )


def run_review(
    evidence_data: dict,
    claim_data: dict,
    model: ModelConfig,
    store: AttemptStore,
    transport: Callable[[dict], WireResponse],
    *,
    policy: PromptPolicy | None = None,
    max_new_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    if type(max_new_attempts) is not int or not 1 <= max_new_attempts <= 3:
        raise ValueError("1 to 3 identical-request attempts per invocation are permitted")
    result, evidence, claim = prepare(evidence_data, claim_data)
    if not evidence or not claim or result["contract_status"] == "VIOLATION":
        return result
    try:
        request = make_request(evidence, claim, model, policy)
    except ValueError as exc:
        return incomplete(result, "REQUEST_CONFIGURATION_INVALID", str(exc))
    result["request_key"] = request["key"]
    try:
        with store.locked(request) as folder:
            cached = store.successful(folder, request)
            if cached is not None:
                result = apply_review(result, cached)
                result["cache_hit"] = True
                return result
            store.recover_interrupted(folder, request)
            for retry in range(max_new_attempts):
                indices = [
                    int(p.name.split("-")[1].split(".")[0])
                    for p in folder.glob("attempt-*.started.json")
                ]
                index = max(indices, default=0) + 1
                started = folder / f"attempt-{index:06d}.started.json"
                finished = folder / f"attempt-{index:06d}.result.json"
                write_new(
                    started,
                    {
                        "request_key": request["key"],
                        "attempt": index,
                        "started_utc": now(),
                        "status": "STARTED",
                    },
                )
                raw, observations = b"", {}
                error = None
                review = None
                try:
                    response = transport(request)
                    if not isinstance(response, WireResponse) or not isinstance(
                        response.raw, bytes
                    ):
                        raise TechnicalError("TRANSPORT_RESPONSE_INVALID", "Expected raw bytes")
                    raw, observations = response.raw, response.observations
                    review = parse_response(raw, request)
                except TechnicalError as exc:
                    error = exc
                    raw = exc.raw or raw
                except Exception as exc:
                    error = TechnicalError("UNEXPECTED_TECHNICAL_ERROR", str(exc))
                record = {
                    "request_key": request["key"],
                    "attempt": index,
                    "ended_utc": now(),
                    "status": "SUCCEEDED" if error is None else "TECHNICAL_ERROR",
                    "raw_response_base64": base64.b64encode(raw).decode("ascii"),
                    "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                    "observations": observations,
                    "parsed_review": review.model_dump() if review is not None else None,
                    "error_code": error.code if error else None,
                    "error_message": str(error) if error else None,
                    "retryable": error.retryable if error else False,
                }
                write_new(finished, record)
                if error is None:
                    write_new(
                        folder / "SUCCESS.json",
                        {
                            "key": request["key"],
                            "attempt": index,
                            "record_sha256": digest(record),
                        },
                    )
                    result = apply_review(result, review)
                    result.update({"cache_hit": False, "successful_attempt": index})
                    return result
                result = incomplete(result, error.code, str(error))
                if not error.retryable or retry + 1 == max_new_attempts:
                    break
                sleep((1.0, 3.0)[retry])
    except (TechnicalError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, TechnicalError) else "RECORDING_OR_CACHE_ERROR"
        return incomplete(result, code, str(exc))
    return result


class OllamaTransport:
    """Local generate API; no proxy fallback and no environment-variable aliases."""

    def __init__(self, model: ModelConfig):
        self.model = model

    def _http(self, suffix: str, body=None) -> bytes:
        request = Request(
            self.model.host + suffix,
            data=canonical_json(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.model.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            raise TechnicalError(
                "HTTP_ERROR",
                str(exc),
                retryable=exc.code == 429 or exc.code >= 500,
                raw=exc.read(),
            ) from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise TechnicalError("CONNECTION_ERROR", str(exc), retryable=True) from exc

    def identity(self) -> dict:
        try:
            tags = strict_json(self._http("/api/tags"))
            version = strict_json(self._http("/api/version"))["version"]
            matches = [x for x in tags["models"] if x.get("name") == self.model.model]
            if len(matches) != 1 or matches[0].get("digest") != self.model.model_digest:
                raise TechnicalError("MODEL_IDENTITY_MISMATCH", "Model digest differs from config")
            if version != self.model.server_version:
                raise TechnicalError(
                    "SERVER_VERSION_MISMATCH", "Ollama version differs from config"
                )
            return {"model": self.model.model, "digest": matches[0]["digest"], "version": version}
        except (KeyError, TypeError, ValueError) as exc:
            raise TechnicalError("IDENTITY_RESPONSE_INVALID", str(exc)) from exc

    def __call__(self, request: dict) -> WireResponse:
        if request["envelope"]["model_config"] != self.model.model_dump():
            raise TechnicalError("REQUEST_MODEL_MISMATCH", "Transport differs from request")
        before = self.identity()
        raw = self._http("/api/generate", request["envelope"]["body"])
        try:
            after = self.identity()
        except TechnicalError as exc:
            # Preserve a received response even when the post-request identity check fails.
            raise TechnicalError(exc.code, str(exc), retryable=exc.retryable, raw=raw) from exc
        return WireResponse(raw, {"identity_before": before, "identity_after": after})

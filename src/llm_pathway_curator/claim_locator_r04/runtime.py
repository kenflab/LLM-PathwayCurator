"""Immutable first outcomes for the R04 source-only probe, including failures."""

from __future__ import annotations

import base64
import hashlib
import time

from ..contract_v161.prompt import digest, strict_json
from ..contract_v161.runtime import AttemptStore, TechnicalError, WireResponse, now, write_new
from .locator import parse_response


class FirstOutcomeStore(AttemptStore):
    def run(self, request, transport, *, allow_new=True):
        with self.locked(request) as folder:
            result_path, started = folder / "FIRST_RESULT.json", folder / "STARTED.json"
            if not result_path.exists() and started.exists():
                if strict_json(started.read_bytes()).get("request_key") != request["key"]:
                    raise TechnicalError("CACHE_INTEGRITY_ERROR", "Interrupted key differs")
                write_new(
                    result_path,
                    {
                        "request_key": request["key"],
                        "ended_utc": now(),
                        "status": "INTERRUPTED",
                        "error_code": "INTERRUPTED",
                        "error_message": "No committed first response",
                        "raw_response_base64": "",
                        "raw_response_sha256": hashlib.sha256(b"").hexdigest(),
                        "parsed_locations": None,
                        "observations": {},
                    },
                )
            if result_path.exists():
                record = strict_json(result_path.read_bytes())
                self.validate_record(record, request)
                seal = folder / "FIRST_RESULT_SHA256.json"
                if seal.exists():
                    if strict_json(seal.read_bytes()) != {"sha256": digest(record)}:
                        raise TechnicalError(
                            "CACHE_INTEGRITY_ERROR", "First result digest mismatch"
                        )
                else:
                    write_new(seal, {"sha256": digest(record)})
                return record, True, False
            if not allow_new:
                return (
                    {
                        "request_key": request["key"],
                        "status": "NOT_RUN_BUDGET",
                        "parsed_locations": None,
                        "error_code": "BUDGET_LIMIT",
                    },
                    False,
                    False,
                )
            write_new(started, {"request_key": request["key"], "started_utc": now()})
            raw, observations, parsed, error = b"", {}, None, None
            try:
                response = transport(request)
                if not isinstance(response, WireResponse) or not isinstance(response.raw, bytes):
                    raise TechnicalError("TRANSPORT_RESPONSE_INVALID", "Expected raw bytes")
                raw, observations = response.raw, response.observations
                model = request["envelope"]["model_config"]
                identity = {
                    "model": model["model"],
                    "digest": model["model_digest"],
                    "version": model["server_version"],
                }
                if any(
                    observations.get(k) != identity for k in ("identity_before", "identity_after")
                ):
                    raise TechnicalError(
                        "BACKEND_IDENTITY_MISMATCH", "Backend receipt differs", raw=raw
                    )
                parsed = parse_response(raw, request)
            except TechnicalError as exc:
                error, raw = exc, exc.raw or raw
            except Exception as exc:
                error = TechnicalError("UNEXPECTED_TECHNICAL_ERROR", str(exc))
            record = {
                "request_key": request["key"],
                "ended_utc": now(),
                "status": "SUCCEEDED" if error is None else "TECHNICAL_ERROR",
                "error_code": error.code if error else None,
                "error_message": str(error) if error else None,
                "raw_response_base64": base64.b64encode(raw).decode("ascii"),
                "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
                "parsed_locations": parsed.model_dump() if parsed else None,
                "observations": observations,
            }
            write_new(result_path, record)
            write_new(folder / "FIRST_RESULT_SHA256.json", {"sha256": digest(record)})
            return record, False, True

    @staticmethod
    def validate_record(record, request):
        if record["request_key"] != request["key"]:
            raise TechnicalError("CACHE_INTEGRITY_ERROR", "Result belongs to another request")
        raw = base64.b64decode(record["raw_response_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != record["raw_response_sha256"]:
            raise TechnicalError("CACHE_INTEGRITY_ERROR", "Raw response differs")
        model = request["envelope"]["model_config"]
        expected_identity = {
            "model": model["model"],
            "digest": model["model_digest"],
            "version": model["server_version"],
        }
        if record["status"] == "SUCCEEDED":
            if parse_response(raw, request).model_dump() != record["parsed_locations"]:
                raise TechnicalError("CACHE_INTEGRITY_ERROR", "Parsed result differs")
            for name in ("identity_before", "identity_after"):
                if record["observations"].get(name) != expected_identity:
                    raise TechnicalError("CACHE_INTEGRITY_ERROR", "Backend receipt differs")
        elif record["status"] not in {"TECHNICAL_ERROR", "INTERRUPTED"}:
            raise TechnicalError("CACHE_INTEGRITY_ERROR", "Invalid first outcome status")
        elif record["parsed_locations"] is not None:
            raise TechnicalError("CACHE_INTEGRITY_ERROR", "Failed outcome contains validated data")


class CallBudget:
    def __init__(self, calls=16, seconds=600):
        if type(calls) is not int or not 0 <= calls <= 16:
            raise ValueError("Require 0 to 16 new requests")
        if not 0 < seconds <= 1800:
            raise ValueError("Require a positive budget up to 1800 seconds")
        self.limit, self.seconds, self.started, self.calls = calls, seconds, time.monotonic(), 0

    def available(self):
        return self.calls < self.limit and time.monotonic() - self.started < self.seconds

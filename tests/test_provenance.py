from __future__ import annotations

import math

import pytest

from verify_agent_memory.provenance import (
    canonical_json_bytes,
    sha256_bytes,
    verify_sha256,
)


def test_canonical_json_is_stable_and_ascii() -> None:
    assert canonical_json_bytes({"z": "值", "a": 1}) == (b'{"a":1,"z":"\\u503c"}\n')


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": math.nan})


def test_hash_verification_fails_closed() -> None:
    content = b"frozen evidence"
    verify_sha256(content, sha256_bytes(content), label="fixture")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(content, "0" * 64, label="fixture")

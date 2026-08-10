"""Platform-stable content identities for provenance evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


HASH_CONTRACT_ID = "dc-energy-provenance-sha256-v2"
CANONICAL_JSON_REPRESENTATION = "canonical-json-utf8-sort-compact-v1"
GIT_BLOB_REPRESENTATION = "git-blob-bytes-v1"
RAW_BYTES_REPRESENTATION = "raw-bytes-v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", newline=None) as handle:
        return json.load(handle)


def canonical_json_file_sha256(path: Path) -> str:
    return canonical_json_sha256(load_json(path))


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def resolve_commit(root: Path, commit: str) -> str:
    return _git(root, "rev-parse", f"{commit}^{{commit}}").decode("ascii").strip()


def git_blob_oid(root: Path, commit: str, path: str) -> str:
    return _git(root, "rev-parse", f"{commit}:{path}").decode("ascii").strip()


def git_blob_bytes(root: Path, commit: str, path: str) -> bytes:
    oid = git_blob_oid(root, commit, path)
    object_type = _git(root, "cat-file", "-t", oid).decode("ascii").strip()
    if object_type != "blob":
        raise ValueError(f"{commit}:{path} is {object_type}, not a blob")
    return _git(root, "cat-file", "blob", oid)


def git_blob_sha256(root: Path, commit: str, path: str) -> str:
    return sha256_bytes(git_blob_bytes(root, commit, path))


def git_blob_reference(root: Path, commit: str, path: str) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "commit": resolve_commit(root, commit),
        "git_blob_oid": git_blob_oid(root, commit, path),
        "path": path,
        "representation": GIT_BLOB_REPRESENTATION,
        "sha256": git_blob_sha256(root, commit, path),
    }


def json_file_reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "path": path.relative_to(root).as_posix(),
        "representation": CANONICAL_JSON_REPRESENTATION,
        "sha256": canonical_json_file_sha256(path),
    }


def raw_file_reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "path": path.relative_to(root).as_posix(),
        "representation": RAW_BYTES_REPRESENTATION,
        "sha256": sha256_file(path),
    }

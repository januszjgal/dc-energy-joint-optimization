"""Tests for platform-stable provenance content identities."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ramp_rl.provenance import (
    canonical_json_file_sha256,
    canonical_json_sha256,
    git_blob_sha256,
    sha256_file,
)


class ProvenanceHashContractTests(unittest.TestCase):
    def test_canonical_json_ignores_format_key_order_and_eol(self) -> None:
        payload = {"z": [1, 2], "a": {"text": "value"}}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            crlf.write_bytes(b'{\r\n  "z": [1, 2],\r\n  "a": {"text": "value"}\r\n}\r\n')
            self.assertEqual(canonical_json_file_sha256(lf), canonical_json_sha256(payload))
            self.assertEqual(canonical_json_file_sha256(crlf), canonical_json_sha256(payload))
            self.assertNotEqual(sha256_file(lf), sha256_file(crlf))

    def test_raw_binary_hash_detects_single_byte_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.zip"
            path.write_bytes(b"\x00binary\r\npayload\xff")
            original = sha256_file(path)
            path.write_bytes(b"\x00binary\r\npayload\xfe")
            self.assertNotEqual(sha256_file(path), original)

    def test_git_blob_hash_is_independent_of_checkout_eol(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "provenance@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Provenance Test"],
                cwd=root,
                check=True,
            )
            path = root / "evidence.py"
            path.write_bytes(b"first\nsecond\n")
            subprocess.run(["git", "add", "evidence.py"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "--quiet", "-m", "fixture"], cwd=root, check=True
            )
            expected = git_blob_sha256(root, "HEAD", "evidence.py")
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(git_blob_sha256(root, "HEAD", "evidence.py"), expected)
            self.assertNotEqual(sha256_file(path), expected)


if __name__ == "__main__":
    unittest.main()

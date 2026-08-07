"""Build the final thesis DOCX from the root Markdown source."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "thesis_paper.md"
DEFAULT_OUTPUT = ROOT / "thesis_paper.docx"
NODE_BUILDER = ROOT / "scripts" / "build_final_thesis_docx.js"
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final thesis DOCX from Markdown."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_source(source: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Thesis source does not exist: {source}")

    text = source.read_text(encoding="utf-8")
    missing: list[Path] = []
    for raw_path in IMAGE_PATTERN.findall(text):
        if re.match(r"^[a-z]+://", raw_path, flags=re.IGNORECASE):
            continue
        image_path = (source.parent / raw_path).resolve()
        if not image_path.is_file():
            missing.append(image_path)
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing thesis image(s):\n{rendered}")


def validate_docx(output: Path) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"DOCX was not created: {output}")

    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/styles.xml",
    }
    with zipfile.ZipFile(output) as package:
        names = set(package.namelist())
        missing = required - names
        if missing:
            raise RuntimeError(f"DOCX is missing required entries: {sorted(missing)}")
        corrupt = package.testzip()
        if corrupt:
            raise RuntimeError(f"DOCX contains a corrupt ZIP member: {corrupt}")
        for name in names:
            if name.endswith(".xml"):
                ElementTree.fromstring(package.read(name))


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    validate_source(source)
    if not NODE_BUILDER.is_file():
        raise FileNotFoundError(f"Node builder does not exist: {NODE_BUILDER}")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "node",
            str(NODE_BUILDER),
            str(source),
            str(output),
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
    )
    validate_docx(output)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Wrote {output}")
    print(f"SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

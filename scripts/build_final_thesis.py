"""Build the final thesis DOCX from the root Markdown source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "thesis_paper.md"
DEFAULT_OUTPUT = ROOT / "thesis_paper.docx"
NODE_BUILDER = ROOT / "scripts" / "build_final_thesis_docx.js"
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
UNRESOLVED_RESULT_PATTERN = re.compile(
    r"\{\{CANONICAL_V4R:[A-Z0-9_]+\}\}"
)
HYPERLINK_RELATIONSHIP_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
)
WORD_SAFE_HYPERLINK_IDS = (
    "rIdgpazldmzsaqs3jmtjnap6",
    "rIdlcv604cskny45r_xtsr72",
    "rIdf92p52yjfvj1hnpcw43fb",
    "rIdo2knrcuw_fop4eazz3xme",
    "rIdo6gg1tyvs5wvapldijwzk",
    "rIdkx8df7z9jrlblht5lmkyr",
    "rIdgegxjdwhdozd88xtnxppl",
    "rIdbz_slxcjojckeebgjcgne",
    "rIdikl4avk_4afim2fldmbew",
    "rIdsqlfw8mw8qkztfwi8e_d1",
)
FIXED_CORE_TIMESTAMP = "2026-08-07T00:00:00.000Z"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
V4R_FIGURE_MANIFEST = ROOT / "docs" / "figures" / "ramp_v6" / "v4r_figure_manifest.json"


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
    unresolved = sorted(set(UNRESOLVED_RESULT_PATTERN.findall(text)))
    if unresolved:
        rendered = "\n".join(f"  - {token}" for token in unresolved)
        raise RuntimeError(
            "Thesis source contains unresolved canonical-result placeholders. "
            "Materialize the source from schema-valid evidence before building:\n"
            f"{rendered}"
        )
    if "data-result-contract-begin" in text or UNRESOLVED_RESULT_PATTERN.search(text):
        from ramp_rl.v4r_thesis import load_verified_evidence
        from scripts.validate_ramp_thesis import validate_text

        errors = validate_text(
            text,
            allow_placeholders=False,
            evidence=load_verified_evidence(),
        )
        if errors:
            raise RuntimeError("Invalid ramp thesis source: " + "; ".join(errors))
    figure_manifest: dict[str, object] | None = None
    if "data-result-contract-begin" in text:
        if not V4R_FIGURE_MANIFEST.is_file():
            raise FileNotFoundError(f"Missing V4R figure manifest: {V4R_FIGURE_MANIFEST}")
        figure_manifest = json.loads(V4R_FIGURE_MANIFEST.read_text(encoding="utf-8"))
        from ramp_rl.v4r_thesis import EXPECTED_CANONICAL_SHA256

        if figure_manifest.get("canonical_evidence_sha256") != EXPECTED_CANONICAL_SHA256:
            raise RuntimeError("V4R figure manifest is bound to the wrong canonical evidence")
    missing: list[Path] = []
    for raw_path in IMAGE_PATTERN.findall(text):
        if re.match(r"^[a-z]+://", raw_path, flags=re.IGNORECASE):
            continue
        source_relative = (source.parent / raw_path).resolve()
        root_relative = (ROOT / raw_path).resolve()
        if not source_relative.is_file() and not root_relative.is_file():
            missing.append(source_relative)
            continue
        if figure_manifest is not None and raw_path.startswith("docs/figures/ramp_v6/"):
            image_path = root_relative if root_relative.is_file() else source_relative
            expected = figure_manifest.get("files", {}).get(image_path.name)
            if not isinstance(expected, str):
                raise RuntimeError(f"V4R figure is not declared in the manifest: {image_path.name}")
            actual = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"V4R figure hash mismatch: {image_path.name}")
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
        document_xml = package.read("word/document.xml").decode("utf-8")
        styles_xml = package.read("word/styles.xml").decode("utf-8")
        drawing_ids = re.findall(
            r'<wp:docPr[^>]*\bid="([^"]+)"',
            document_xml,
        )
        picture_ids = re.findall(
            r'<pic:cNvPr[^>]*\bid="([^"]+)"',
            document_xml,
        )
        style_ids = re.findall(
            r'<w:style[^>]*w:styleId="([^"]+)"',
            styles_xml,
        )
        for name, identifiers in (
            ("wp:docPr", drawing_ids),
            ("pic:cNvPr", picture_ids),
            ("w:style", style_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise RuntimeError(f"DOCX contains duplicate {name} identifiers")
        if re.search(r"<w:tblHeader\b[^>]*w:val=\"false\"", document_xml):
            raise RuntimeError("DOCX contains schema-invalid false table-header flags")


def _replace_indexed_id(
    xml: str,
    pattern: str,
) -> str:
    counter = iter(range(1, 1_000_000))
    return re.sub(
        pattern,
        lambda match: f"{match.group(1)}{next(counter)}{match.group(2)}",
        xml,
    )


def normalize_docx_package(
    output: Path,
    *,
    frozen_v5_hyperlinks: bool,
) -> None:
    """Normalize IDs, metadata, and ZIP timestamps for Word and reproducibility."""
    relationships_path = "word/_rels/document.xml.rels"
    document_path = "word/document.xml"
    core_properties_path = "docProps/core.xml"
    with zipfile.ZipFile(output) as package:
        entries = [
            (entry, package.read(entry.filename))
            for entry in package.infolist()
        ]
    payloads = {entry.filename: data for entry, data in entries}
    relationships = payloads[relationships_path].decode("utf-8")
    document_xml = payloads[document_path].decode("utf-8")
    current_ids = re.findall(
        rf'<Relationship Id="([^"]+)" Type="{re.escape(HYPERLINK_RELATIONSHIP_TYPE)}"',
        relationships,
    )
    if frozen_v5_hyperlinks:
        if len(current_ids) != len(WORD_SAFE_HYPERLINK_IDS):
            raise RuntimeError(
                "The frozen thesis expects exactly "
                f"{len(WORD_SAFE_HYPERLINK_IDS)} external hyperlinks, found "
                f"{len(current_ids)}. Update the compatibility ID pool and revalidate "
                "with Microsoft Word before accepting a changed bibliography."
            )
        replacement_ids = WORD_SAFE_HYPERLINK_IDS
    else:
        replacement_ids = tuple(
            f"rIdrampthesis{index:04d}"
            for index in range(1, len(current_ids) + 1)
        )
    for current_id, safe_id in zip(
        current_ids,
        replacement_ids,
        strict=True,
    ):
        relationships = relationships.replace(
            f'Id="{current_id}"',
            f'Id="{safe_id}"',
        )
        document_xml = document_xml.replace(
            f'r:id="{current_id}"',
            f'r:id="{safe_id}"',
        )
    document_xml = _replace_indexed_id(
        document_xml,
        r'(<wp:docPr\b[^>]*\bid=")[^"]+(")',
    )
    document_xml = _replace_indexed_id(
        document_xml,
        r'(<pic:cNvPr\b[^>]*\bid=")[^"]+(")',
    )
    core_properties = payloads[core_properties_path].decode("utf-8")
    core_properties = re.sub(
        r"(<dcterms:created\b[^>]*>).*?(</dcterms:created>)",
        rf"\g<1>{FIXED_CORE_TIMESTAMP}\g<2>",
        core_properties,
    )
    core_properties = re.sub(
        r"(<dcterms:modified\b[^>]*>).*?(</dcterms:modified>)",
        rf"\g<1>{FIXED_CORE_TIMESTAMP}\g<2>",
        core_properties,
    )
    payloads[relationships_path] = relationships.encode("utf-8")
    payloads[document_path] = document_xml.encode("utf-8")
    payloads[core_properties_path] = core_properties.encode("utf-8")
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as package:
            for entry, _ in entries:
                normalized = zipfile.ZipInfo(
                    entry.filename,
                    date_time=FIXED_ZIP_TIMESTAMP,
                )
                normalized.compress_type = (
                    zipfile.ZIP_STORED
                    if entry.is_dir()
                    else zipfile.ZIP_DEFLATED
                )
                normalized.external_attr = entry.external_attr
                normalized.create_system = entry.create_system
                normalized.comment = entry.comment
                package.writestr(normalized, payloads[entry.filename])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


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
    normalize_docx_package(
        output,
        frozen_v5_hyperlinks=False,
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

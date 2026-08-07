"use strict";

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  ExternalHyperlink,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableOfContents,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const [, , sourceArg, outputArg, rootArg] = process.argv;
if (!sourceArg || !outputArg || !rootArg) {
  throw new Error(
    "Usage: node scripts/build_final_thesis_docx.js SOURCE.md OUTPUT.docx REPO_ROOT",
  );
}

const SOURCE = path.resolve(sourceArg);
const OUTPUT = path.resolve(outputArg);
const ROOT = path.resolve(rootArg);
const CONTENT_WIDTH = 9360;
const BODY_FONT = "Arial";
const MONO_FONT = "Courier New";

function readFrontMatter(lines) {
  const metadata = {};
  if (lines[0] !== "---") {
    return { metadata, start: 0 };
  }
  let index = 1;
  while (index < lines.length && lines[index] !== "---") {
    const match = lines[index].match(/^([^:]+):\s*(.*)$/);
    if (match) {
      metadata[match[1].trim()] = match[2].trim();
    }
    index += 1;
  }
  return { metadata, start: Math.min(index + 1, lines.length) };
}

function parseInline(text, options = {}) {
  const runs = [];
  const tokenPattern =
    /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(tokenPattern)) {
    if (match.index > cursor) {
      runs.push(
        new TextRun({
          text: text.slice(cursor, match.index),
          font: options.font || BODY_FONT,
          size: options.size,
          bold: options.bold,
          italics: options.italics,
        }),
      );
    }
    const token = match[0];
    if (token.startsWith("**")) {
      runs.push(
        new TextRun({
          text: token.slice(2, -2),
          bold: true,
          font: options.font || BODY_FONT,
          size: options.size,
        }),
      );
    } else if (token.startsWith("`")) {
      runs.push(
        new TextRun({
          text: token.slice(1, -1),
          font: MONO_FONT,
          size: options.size || 20,
          shading: { fill: "EEF1F4", type: ShadingType.CLEAR },
        }),
      );
    } else if (token.startsWith("*")) {
      runs.push(
        new TextRun({
          text: token.slice(1, -1),
          italics: true,
          font: options.font || BODY_FONT,
          size: options.size,
        }),
      );
    } else {
      const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (link && /^https?:\/\//i.test(link[2])) {
        runs.push(
          new ExternalHyperlink({
            link: link[2],
            children: [
              new TextRun({
                text: link[1],
                style: "Hyperlink",
                font: options.font || BODY_FONT,
                size: options.size,
              }),
            ],
          }),
        );
      } else if (link) {
        runs.push(
          new TextRun({
            text: `${link[1]} (${link[2]})`,
            font: options.font || BODY_FONT,
            size: options.size,
          }),
        );
      }
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    runs.push(
      new TextRun({
        text: text.slice(cursor),
        font: options.font || BODY_FONT,
        size: options.size,
        bold: options.bold,
        italics: options.italics,
      }),
    );
  }
  return runs.length
    ? runs
    : [
        new TextRun({
          text,
          font: options.font || BODY_FONT,
          size: options.size,
          bold: options.bold,
          italics: options.italics,
        }),
      ];
}

function parseTableCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableSeparator(line) {
  const cells = parseTableCells(line);
  return (
    cells.length > 0 &&
    cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")))
  );
}

function cellBorders() {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "C9D2DC" };
  return { top: border, bottom: border, left: border, right: border };
}

function markdownTable(rows) {
  const columnCount = Math.max(...rows.map((row) => row.length));
  const baseWidth = Math.floor(CONTENT_WIDTH / columnCount);
  const widths = Array.from({ length: columnCount }, (_, index) =>
    index === columnCount - 1
      ? CONTENT_WIDTH - baseWidth * (columnCount - 1)
      : baseWidth,
  );
  const tableRows = rows.map(
    (row, rowIndex) =>
      new TableRow({
        tableHeader: rowIndex === 0,
        children: widths.map(
          (width, columnIndex) =>
            new TableCell({
              width: { size: width, type: WidthType.DXA },
              borders: cellBorders(),
              shading:
                rowIndex === 0
                  ? { fill: "DCE6F1", type: ShadingType.CLEAR }
                  : undefined,
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [
                new Paragraph({
                  spacing: { after: 0 },
                  children: parseInline(row[columnIndex] || "", {
                    size: 18,
                    bold: rowIndex === 0,
                  }),
                }),
              ],
            }),
        ),
      }),
  );
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: tableRows,
  });
}

function pngDimensions(buffer) {
  const signature = buffer.subarray(1, 4).toString("ascii");
  if (signature !== "PNG") {
    return null;
  }
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
  };
}

function imageBlock(rawPath, caption) {
  const candidateFromSource = path.resolve(path.dirname(SOURCE), rawPath);
  const imagePath = fs.existsSync(candidateFromSource)
    ? candidateFromSource
    : path.resolve(ROOT, rawPath);
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Missing image: ${rawPath}`);
  }
  const data = fs.readFileSync(imagePath);
  const extension = path.extname(imagePath).slice(1).toLowerCase();
  if (extension !== "png" && extension !== "jpg" && extension !== "jpeg") {
    throw new Error(`Unsupported image type for DOCX: ${imagePath}`);
  }
  const dimensions = extension === "png" ? pngDimensions(data) : null;
  const nativeWidth = dimensions?.width || 640;
  const nativeHeight = dimensions?.height || 360;
  const scale = Math.min(650 / nativeWidth, 470 / nativeHeight, 1);
  const width = Math.max(1, Math.round(nativeWidth * scale));
  const height = Math.max(1, Math.round(nativeHeight * scale));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({
          type: extension === "jpg" ? "jpg" : extension,
          data,
          transformation: { width, height },
          altText: {
            title: caption || path.basename(imagePath),
            description: caption || path.basename(imagePath),
            name: caption || path.basename(imagePath),
          },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180 },
      children: parseInline(caption, { size: 18, italics: true }),
    }),
  ];
}

function paragraph(text, options = {}) {
  return new Paragraph({
    alignment: options.alignment,
    indent: options.indent,
    border: options.border,
    shading: options.shading,
    spacing: options.spacing || { after: 120, line: 276 },
    keepNext: options.keepNext,
    children: parseInline(text, options),
  });
}

function heading(text, level) {
  const headingLevels = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
    4: HeadingLevel.HEADING_4,
  };
  return new Paragraph({
    heading: headingLevels[Math.min(level, 4)],
    pageBreakBefore: level === 1,
    keepNext: true,
    children: [new TextRun({ text, font: BODY_FONT })],
  });
}

function blockStarts(line, nextLine) {
  const trimmed = line.trim();
  return (
    trimmed === "" ||
    /^#{1,4}\s+/.test(trimmed) ||
    /^```/.test(trimmed) ||
    /^\$\$/.test(trimmed) ||
    /^!\[[^\]]*\]\([^)]+\)$/.test(trimmed) ||
    /^[-*]\s+/.test(trimmed) ||
    /^\d+\.\s+/.test(trimmed) ||
    /^>\s?/.test(trimmed) ||
    /^---+$/.test(trimmed) ||
    /^<!--\s*pagebreak\s*-->$/i.test(trimmed) ||
    (trimmed.startsWith("|") && nextLine && isTableSeparator(nextLine))
  );
}

function bodyBlocks(lines, start) {
  const blocks = [];
  let index = start;
  while (index < lines.length) {
    const raw = lines[index];
    const trimmed = raw.trim();
    if (!trimmed) {
      index += 1;
      continue;
    }
    if (/^<!--\s*pagebreak\s*-->$/i.test(trimmed)) {
      blocks.push(new Paragraph({ children: [new PageBreak()] }));
      index += 1;
      continue;
    }
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      blocks.push(heading(headingMatch[2], headingMatch[1].length));
      index += 1;
      continue;
    }
    if (/^```/.test(trimmed)) {
      const language = trimmed.slice(3).trim();
      index += 1;
      const code = [];
      while (index < lines.length && !/^```/.test(lines[index].trim())) {
        code.push(lines[index]);
        index += 1;
      }
      index += 1;
      if (language) {
        blocks.push(
          paragraph(language, {
            size: 17,
            bold: true,
            font: MONO_FONT,
            spacing: { before: 100, after: 0 },
          }),
        );
      }
      for (const line of code.length ? code : [""]) {
        blocks.push(
          new Paragraph({
            border: {
              left: {
                style: BorderStyle.SINGLE,
                size: 8,
                color: "7F8C8D",
                space: 8,
              },
            },
            shading: { fill: "F4F6F7", type: ShadingType.CLEAR },
            indent: { left: 240, right: 120 },
            spacing: { after: 0 },
            children: [
              new TextRun({ text: line || " ", font: MONO_FONT, size: 17 }),
            ],
          }),
        );
      }
      blocks.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    if (/^\$\$/.test(trimmed)) {
      const equation = [];
      const singleLine = trimmed.match(/^\$\$(.+)\$\$$/);
      if (singleLine) {
        equation.push(singleLine[1].trim());
        index += 1;
      } else {
        index += 1;
        while (index < lines.length && !/\$\$$/.test(lines[index].trim())) {
          equation.push(lines[index].trim());
          index += 1;
        }
        if (index < lines.length) {
          const closingText = lines[index].trim().replace(/\$\$$/, "").trim();
          if (closingText) equation.push(closingText);
          index += 1;
        }
      }
      blocks.push(
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 120, after: 120 },
          children: [
            new TextRun({
              text: equation.join(" "),
              font: "Cambria Math",
              size: 22,
              italics: true,
            }),
          ],
        }),
      );
      continue;
    }
    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      blocks.push(...imageBlock(imageMatch[2], imageMatch[1]));
      index += 1;
      continue;
    }
    if (
      trimmed.startsWith("|") &&
      index + 1 < lines.length &&
      isTableSeparator(lines[index + 1])
    ) {
      const rows = [parseTableCells(lines[index])];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(parseTableCells(lines[index]));
        index += 1;
      }
      blocks.push(markdownTable(rows));
      blocks.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    if (/^[-*]\s+/.test(trimmed)) {
      blocks.push(
        new Paragraph({
          numbering: { reference: "thesis-bullets", level: 0 },
          spacing: { after: 60 },
          children: parseInline(trimmed.replace(/^[-*]\s+/, "")),
        }),
      );
      index += 1;
      continue;
    }
    if (/^\d+\.\s+/.test(trimmed)) {
      blocks.push(
        new Paragraph({
          numbering: { reference: "thesis-numbers", level: 0 },
          spacing: { after: 60 },
          children: parseInline(trimmed.replace(/^\d+\.\s+/, "")),
        }),
      );
      index += 1;
      continue;
    }
    if (/^>\s?/.test(trimmed)) {
      const quote = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quote.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      blocks.push(
        paragraph(quote.join(" "), {
          italics: true,
          indent: { left: 480, right: 480 },
          border: {
            left: {
              style: BorderStyle.SINGLE,
              size: 12,
              color: "4472C4",
              space: 10,
            },
          },
          shading: { fill: "EEF4FB", type: ShadingType.CLEAR },
        }),
      );
      continue;
    }
    if (/^---+$/.test(trimmed)) {
      blocks.push(
        new Paragraph({
          border: {
            bottom: {
              style: BorderStyle.SINGLE,
              size: 6,
              color: "4472C4",
              space: 1,
            },
          },
          spacing: { before: 80, after: 160 },
          children: [],
        }),
      );
      index += 1;
      continue;
    }

    const paragraphLines = [trimmed];
    index += 1;
    while (
      index < lines.length &&
      !blockStarts(lines[index], lines[index + 1])
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const text = paragraphLines.join(" ");
    const reference = /^\[\d+\]\s+/.test(text);
    blocks.push(
      paragraph(text, {
        size: reference ? 18 : undefined,
        indent: reference ? { left: 360, hanging: 360 } : undefined,
        spacing: reference ? { after: 60, line: 240 } : undefined,
      }),
    );
  }
  return blocks;
}

function titlePage(metadata) {
  const title =
    metadata.title ||
    "Safe Joint Energy Optimization for Geo-Distributed Data Centers";
  const subtitle =
    metadata.subtitle ||
    "A reproducible study using Google ClusterData, CAISO, and TD3+BC";
  const author = metadata.author || "Janusz Gal";
  const date = metadata.date || "August 2026";
  const repository =
    metadata.repository || "github.com/januszjgal/dc-energy-joint-optimization";
  return [
    new Paragraph({ spacing: { before: 1800, after: 360 }, children: [] }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 280 },
      children: [
        new TextRun({
          text: title,
          font: BODY_FONT,
          size: 40,
          bold: true,
          color: "1F4E79",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 1000 },
      children: [
        new TextRun({
          text: subtitle,
          font: BODY_FONT,
          size: 25,
          italics: true,
          color: "404040",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180 },
      children: [new TextRun({ text: author, font: BODY_FONT, size: 24 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180 },
      children: [new TextRun({ text: date, font: BODY_FONT, size: 22 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
      children: [
        new TextRun({
          text: repository,
          font: MONO_FONT,
          size: 18,
          color: "4472C4",
        }),
      ],
    }),
    new Paragraph({ children: [new PageBreak()] }),
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "Table of Contents", font: BODY_FONT })],
    }),
    new TableOfContents("", { hyperlink: true, headingStyleRange: "1-3" }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

const markdown = fs.readFileSync(SOURCE, "utf8").replace(/\r\n/g, "\n");
const lines = markdown.split("\n");
const { metadata, start } = readFrontMatter(lines);
const children = [...titlePage(metadata), ...bodyBlocks(lines, start)];

const doc = new Document({
  creator: "GitHub Copilot CLI",
  title: metadata.title || "Joint Energy Optimization Thesis",
  description:
    "Reproducible thesis on safe spatio-temporal data-center energy optimization.",
  styles: {
    default: {
      document: {
        run: { font: BODY_FONT, size: 22, color: "202020" },
        paragraph: { spacing: { after: 120, line: 276 } },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: BODY_FONT, size: 32, bold: true, color: "1F4E79" },
        paragraph: {
          spacing: { before: 240, after: 180 },
          outlineLevel: 0,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: BODY_FONT, size: 27, bold: true, color: "2F5597" },
        paragraph: {
          spacing: { before: 200, after: 140 },
          outlineLevel: 1,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: BODY_FONT, size: 24, bold: true, color: "404040" },
        paragraph: {
          spacing: { before: 160, after: 100 },
          outlineLevel: 2,
        },
      },
      {
        id: "Heading4",
        name: "Heading 4",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: BODY_FONT, size: 22, bold: true, italics: true },
        paragraph: {
          spacing: { before: 140, after: 80 },
          outlineLevel: 3,
        },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "thesis-bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "thesis-numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: {
            top: 1080,
            right: 1080,
            bottom: 1080,
            left: 1080,
            header: 540,
            footer: 540,
          },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              border: {
                bottom: {
                  style: BorderStyle.SINGLE,
                  size: 4,
                  color: "A6A6A6",
                  space: 1,
                },
              },
              children: [
                new TextRun({
                  text: "Safe Joint Energy Optimization",
                  font: BODY_FONT,
                  size: 17,
                  color: "666666",
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Page ", font: BODY_FONT, size: 17 }),
                new TextRun({
                  children: [PageNumber.CURRENT],
                  font: BODY_FONT,
                  size: 17,
                }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.writeFileSync(OUTPUT, buffer);
});

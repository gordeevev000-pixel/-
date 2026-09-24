// Текст выступления к презентации-сайту — отдельным файлом .docx
// Запуск: node build_notes.js <выходной.docx>   (данные — notes.json рядом со скриптом)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber, BorderStyle,
} = require("docx");

const OUT = process.argv[2] || "tekst-vystupleniya.docx";
const notes = JSON.parse(fs.readFileSync(path.join(__dirname, "notes.json"), "utf8"));

const FONT = "Arial";
const PT = (n) => n * 2;
const MM = (n) => Math.round(n * 56.6929);
const WPM = 125; // спокойный темп речи, слов в минуту

const words = (t) => t.split(/\s+/).filter((w) => /[\p{L}\d]/u.test(w)).length;
const secs = (t) => Math.max(5, Math.round(words(t) / WPM * 60 / 5) * 5);
const total = notes.reduce((s, n) => s + secs(n.text), 0);
const mmss = (s) => `${Math.floor(s / 60)} мин ${String(s % 60).padStart(2, "0")} с`;

const run = (text, o = {}) => new TextRun({ text, font: FONT, size: PT(o.size || 13), bold: o.bold, color: o.color });

const head = [
  new Paragraph({ spacing: { after: 80 }, children: [run("Текст выступления", { size: 22, bold: true })] }),
  new Paragraph({ spacing: { after: 60 }, children: [run("От Парижа до Ямайки: Парижская, Генуэзская, Бреттон-Вудская и Ямайская валютные системы", { size: 13 })] }),
  new Paragraph({
    spacing: { after: 360 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1B1406", space: 8 } },
    children: [run(`Тема 2, вопрос 5 · ${notes.length} слайдов · около ${Math.round(total / 60)} минут при спокойном темпе`, { size: 11, color: "555555" })],
  }),
];

const body = notes.flatMap((n) => [
  new Paragraph({
    keepNext: true, spacing: { before: 280, after: 40 },
    children: [run(`СЛАЙД ${String(n.n).padStart(2, "0")} · ${n.section.toUpperCase()} · ≈ ${secs(n.text)} с`, { size: 9.5, bold: true, color: "6B6B6B" })],
  }),
  new Paragraph({ keepNext: true, spacing: { after: 100 }, children: [run(n.title, { size: 15, bold: true })] }),
  new Paragraph({ keepLines: true, spacing: { line: 360, after: 0 }, children: [run(n.text, { size: 13 })] }),
]);

const doc = new Document({
  creator: "Студент",
  title: "Текст выступления — От Парижа до Ямайки",
  styles: { default: { document: { run: { font: FONT, size: PT(13) } } } },
  sections: [{
    properties: { page: { size: { width: MM(210), height: MM(297) }, margin: { top: MM(20), bottom: MM(20), left: MM(22), right: MM(18) } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: PT(10), color: "6B6B6B" })] })] }),
    },
    children: [...head, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT, "·", mmss(total)); });

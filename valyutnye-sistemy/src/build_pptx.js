// Презентация в формате PowerPoint (редактируемая): 20 слайдов 16:9 в дизайне сайта.
// Запуск: node build_pptx.js [выходной.pptx]   (нужен пакет pptxgenjs)
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || path.join(__dirname, "..", "prezentaciya.pptx");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 × 5.625 дюйма
pres.title = "От Парижа до Ямайки";
pres.subject = "Парижская, Генуэзская, Бреттон-Вудская и Ямайская валютные системы";

const W = 10, H = 5.625, MX = 0.55, CW = W - 2 * MX;
const F = "Arial", FB = "Arial Black";
const N = 20;

/* ---------- цвета ---------- */
const rgb = (h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => rgb(a).map((v, i) => Math.round(v + (rgb(b)[i] - v) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
function theme(bg, fg, fg2, o) {
  return { bg, fg, fg2, tint: o.tint || mix(bg, fg, o.tintA), line: mix(bg, fg, o.lineA || 0.2), ghost: o.ghost || mix(bg, fg, o.ghostA),
    invBg: o.invBg, invFg: o.invFg, accent: o.accent, mark: o.mark };
}
const T = {
  gold: theme("E8B23A", "1B1406", "4D3A10", { tintA: 0.08, ghostA: 0.085, invBg: "1B1406", invFg: "F4D27E", accent: "8E3226", mark: "1B1406" }),
  genoa: theme("8E3226", "FFF4EE", "F2C6BA", { tintA: 0.09, ghostA: 0.07, lineA: 0.22, invBg: "FFF4EE", invFg: "6B2118", accent: "F5C451", mark: "F5C451" }),
  bw: theme("16336F", "FFFFFF", "C3CEEA", { tintA: 0.08, ghostA: 0.06, invBg: "FFFFFF", invFg: "16336F", accent: "F5C451", mark: "F5C451" }),
  jam: theme("F5F7F6", "0E1A1F", "4A585E", { tint: "E3EBE9", ghost: mix("F5F7F6", "0B7A75", 0.08), lineA: 0.14, invBg: "0B7A75", invFg: "FFFFFF", accent: "0B7A75", mark: "0B7A75" }),
  plain: theme("FFFFFF", "111318", "50545E", { tint: "EFF1F3", ghostA: 0.05, lineA: 0.13, invBg: "111318", invFg: "FFFFFF", accent: "B03A2E", mark: "111318" }),
};
const ERA = { paris: "E8B23A", genoa: "8E3226", bw: "16336F", jam: "0B7A75" };
const CHART = { paris: "B8840C", genoa: "B03A2E", bw: "2D5BC0", jam: "00908A" };

/* ---------- примитивы ---------- */
// **жирный** внутри строки → отдельные фрагменты
function runs(text, base = {}) {
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((p) =>
    p.startsWith("**") ? { text: p.slice(2, -2), options: { ...base, bold: true } } : { text: p, options: { ...base } });
}
function tx(s, text, o) {
  const opt = {
    x: o.x, y: o.y, w: o.w, h: o.h, margin: 0, isTextBox: true, fontFace: o.font || F, fontSize: o.size || 11,
    color: o.color, bold: !!o.bold, align: o.align || "left", valign: o.valign || "top",
  };
  if (o.cs) opt.charSpacing = o.cs;
  if (o.ls) opt.lineSpacingMultiple = o.ls;
  if (o.strike) opt.strike = "sngStrike";
  s.addText(Array.isArray(text) ? text : runs(text), opt);
}
const label = (s, t, text, x, y, w, color) => tx(s, text.toUpperCase(), { x, y, w, h: 0.2, size: 7.5, bold: true, cs: 1, color: color || t.fg2 });
function rect(s, x, y, w, h, color, shape) {
  s.addShape(shape || pres.shapes.RECTANGLE, { x, y, w, h, fill: { color }, line: { type: "none" } });
}
function line(s, x1, y1, x2, y2, color, width = 1, o = {}) {
  const ln = { color, width };
  if (o.end) ln.endArrowType = "triangle";
  if (o.begin) ln.beginArrowType = "triangle";
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: x2 < x1, flipV: y2 < y1, line: ln });
}
// список: нумерованный или с квадратными маркерами
function list(s, items, o) {
  const out = [];
  items.forEach((it, i) => {
    const base = { fontFace: F, fontSize: o.size || 11, color: o.color };
    // невидимый обычный фрагмент в начале: номер пункта берёт его начертание и не становится жирным
    const r = (o.numbered ? [{ text: "\u200B", options: { ...base } }] : []).concat(runs(it, base));
    // свойства абзаца — только у первого фрагмента, иначе pptxgenjs вставляет лишние <a:pPr> внутрь абзаца
    Object.assign(r[0].options, { bullet: o.numbered ? { type: "number", indent: o.indent || 16 } : { code: "25A0", indent: o.indent || 13 },
      paraSpaceAfter: o.gap ?? 6, lineSpacingMultiple: o.ls || 1.05 });
    if (i < items.length - 1) r[r.length - 1].options.breakLine = true;
    out.push(...r);
  });
  s.addText(out, { x: o.x, y: o.y, w: o.w, h: o.h, margin: 0, isTextBox: true, valign: "top", color: o.color });
}
const pad = (n) => String(n).padStart(2, "0");

// каркас слайда: фон, большой год на фоне, раздел, номер, заголовок
function slide(t, n, section, title, ghost) {
  const s = pres.addSlide();
  s.background = { color: t.bg };
  if (ghost) tx(s, ghost, { x: W - 6.25, y: H - 2.0, w: 6.6, h: 2.6, font: FB, size: 170, color: t.ghost, align: "right" });
  rect(s, MX, 0.345, 0.09, 0.09, t.fg);
  tx(s, section.toUpperCase(), { x: MX + 0.17, y: 0.3, w: 7, h: 0.2, size: 8, bold: true, cs: 1, color: t.fg });
  tx(s, `${pad(n)} / ${N}`, { x: W - MX - 1.2, y: 0.3, w: 1.2, h: 0.2, size: 8, bold: true, color: t.fg2, align: "right" });
  if (title) tx(s, title, { x: MX, y: 0.62, w: CW, h: 0.5, size: 24, bold: true, color: t.fg });
  return s;
}
function facts(s, t, rows, x, y, w, pitch = 0.82) {
  rows.forEach(([k, v], i) => {
    tx(s, k, { x, y: y + i * pitch, w: 1.05, h: 0.35, font: FB, size: 16, color: t.fg });
    tx(s, v, { x: x + 1.15, y: y + i * pitch + 0.02, w: w - 1.15, h: pitch - 0.08, size: 10, color: t.fg2, ls: 1.05 });
  });
}
function invBox(s, t, x, y, w, h, lab, text, size = 10.5) {
  rect(s, x, y, w, h, t.invBg);
  let ty = y + 0.14;
  if (lab) { label(s, t, lab, x + 0.18, ty, w - 0.36, t.invFg); ty += 0.24; }
  tx(s, text, { x: x + 0.18, y: ty, w: w - 0.36, h: h - (ty - y) - 0.1, size, color: t.invFg, ls: 1.08 });
}
const noteLine = (s, t, text, x, y, w, h = 0.4, size = 9) => tx(s, text, { x, y, w, h, size, color: t.fg2, ls: 1.08 });

/* ================= 01 · Титул ================= */
{
  const t = T.gold, s = slide(t, 1, "Доклад · Мировая экономика");
  tx(s, "ТЕМА 2. ВАЛЮТНЫЕ ОТНОШЕНИЯ В МИРОВОЙ ЭКОНОМИКЕ · ВОПРОС 5", { x: MX, y: 1.25, w: 4.4, h: 0.4, size: 8.5, bold: true, cs: 1, color: t.fg, ls: 1.1 });
  tx(s, "От Парижа до Ямайки", { x: MX, y: 1.72, w: 4.6, h: 1.6, size: 44, bold: true, color: t.fg, ls: 0.9 });
  tx(s, "Парижская, Генуэзская, Бреттон-Вудская и Ямайская валютные системы", { x: MX, y: 3.45, w: 4.3, h: 0.7, size: 14, color: t.fg2, ls: 1.08 });
  const X0 = 5.4, BW = W - MX - X0;
  label(s, t, "Сколько прожила каждая система", X0, 1.0, BW, t.fg2);
  [["1867", "47 лет", 0.94, "1B1406", "F4D27E", "Парижская · золотомонетный стандарт"],
   ["1922", "17 лет", 0.4, "8E3226", "FFF4EE", "Генуэзская · золотодевизный стандарт"],
   ["1944", "32 года", 0.64, "16336F", "FFFFFF", "Бреттон-Вудская · золотодолларовый стандарт"],
   ["1976", "50 лет и действует", 1, "0B7A75", "FFFFFF", "Ямайская · стандарт СДР, многовалютный"]].forEach(([yr, d, k, bg, fg, cap], i) => {
    const y = 1.3 + i * 0.93, w = BW * k;
    rect(s, X0, y, w, 0.5, bg);
    tx(s, yr, { x: X0 + 0.12, y: y + 0.04, w: 1.4, h: 0.42, font: FB, size: 22, color: fg, valign: "middle" });
    tx(s, d.toUpperCase(), { x: X0 + w - 1.6, y: y + 0.06, w: 1.48, h: 0.4, size: 7.5, bold: true, cs: 0.5, color: fg, align: "right", valign: "bottom" });
    tx(s, cap, { x: X0, y: y + 0.55, w: BW, h: 0.22, size: 9, bold: true, color: t.fg2 });
  });
}

/* ================= 02 · Понятие ================= */
{
  const t = T.plain, s = slide(t, 2, "Введение · Понятие", "Что такое мировая валютная система");
  tx(s, "Форма организации международных валютных отношений, закреплённая межгосударственными соглашениями.", { x: MX, y: 1.4, w: 4.95, h: 0.8, size: 15, bold: true, color: t.fg, ls: 1.05 });
  label(s, t, "Из чего она состоит", MX, 2.4, 4.9);
  list(s, ["формы мировых денег: золото, резервные валюты, СДР", "условия конвертируемости валют", "режим валютных паритетов и курсов",
    "регулирование международной валютной ликвидности", "правила международных расчётов", "режим валютных рынков и рынков золота",
    "межгосударственные институты регулирования (МВФ и др.)"], { x: MX, y: 2.65, w: 4.95, h: 2.4, size: 10.5, color: t.fg, gap: 4 });
  const X = 5.9, WR = W - MX - X;
  label(s, t, "Главный вопрос каждой эпохи", X, 1.4, WR);
  tx(s, "Что лежит в основе мировых денег?", { x: X, y: 1.62, w: WR, h: 0.35, size: 14, bold: true, color: t.fg });
  [["Золото", "Париж, Генуя", ["E8B23A", "8E3226"]], ["Золото и доллар США", "Бреттон-Вудс", ["16336F"]], ["СДР и ведущие валюты", "Ямайка", ["0B7A75"]]].forEach(([a, b, cs], i) => {
    const y = 2.15 + i * 0.58;
    rect(s, X, y, WR, 0.48, t.tint);
    cs.forEach((c, k) => rect(s, X + 0.14, y + 0.1 + k * (0.28 / cs.length), 0.07, 0.28 / cs.length, c));
    tx(s, a, { x: X + 0.34, y, w: 2.2, h: 0.48, size: 12, bold: true, color: t.fg, valign: "middle" });
    tx(s, b.toUpperCase(), { x: X + WR - 1.4, y, w: 1.28, h: 0.48, size: 7, bold: true, cs: 0.5, color: t.fg2, align: "right", valign: "middle" });
  });
}

/* ================= 03 · Шкала времени ================= */
{
  const t = T.plain, s = slide(t, 3, "Введение · Хронология", "Четыре системы за 160 лет");
  const X = (y) => 2.95 + (y - 1815) / (2030 - 1815) * 6.5;
  const top = 1.5;
  line(s, X(1815), top, X(2030), top, t.line, 1);
  [1820, 1850, 1880, 1910, 1940, 1970, 2000].forEach((y) => {
    tx(s, String(y), { x: X(y) - 0.3, y: top - 0.22, w: 0.6, h: 0.18, size: 8, bold: true, color: t.fg2, align: "center" });
  });
  const ev = [[1914, "1914 · Первая мировая война", 0, "r"], [1929, "1929 · Великая депрессия", 1, "l"], [1971, "1971 · «Никсоновский шок»", 0, "c"],
    [1999, "1999 · евро", 1, "c"], [2016, "2016 · юань в корзине СДР", 2, "r"]];
  ev.forEach(([y, text, lvl, a]) => {
    const ly = 3.95 + lvl * 0.22;
    line(s, X(y), top + 0.05, X(y), ly, t.line, 0.75);
    const w = 2.0, x = a === "r" ? X(y) - w - 0.04 : a === "l" ? X(y) + 0.04 : X(y) - w / 2;
    tx(s, text, { x, y: ly, w, h: 0.2, size: 8, color: t.fg2, align: a === "r" ? "right" : a === "l" ? "left" : "center" });
  });
  const rows = [["Парижская", "золотомонетный стандарт", 1867, 1914, "paris"], ["Генуэзская", "золотодевизный стандарт", 1922, 1939, "genoa"],
    ["Бреттон-Вудская", "золотодолларовый стандарт", 1944, 1976, "bw"], ["Ямайская", "стандарт СДР, многовалютный", 1976, 2026, "jam"]];
  rows.forEach(([n, st, a, b, c], i) => {
    const y = 1.85 + i * 0.52;
    tx(s, n, { x: MX, y: y - 0.02, w: 2.3, h: 0.2, size: 11, bold: true, color: t.fg });
    tx(s, st, { x: MX, y: y + 0.19, w: 2.3, h: 0.18, size: 8.5, color: t.fg2 });
    if (c === "paris") {
      rect(s, X(1821), y, X(1867) - X(1821) - 0.02, 0.28, mix("FFFFFF", CHART.paris, 0.22));
      tx(s, "Великобритания с 1821 г.", { x: X(1821), y, w: X(1867) - X(1821), h: 0.28, size: 7.5, color: t.fg2, align: "center", valign: "middle" });
    }
    rect(s, X(a), y, X(b) - X(a), 0.28, CHART[c]);
    if (c === "bw") rect(s, X(1971), y, X(1976) - X(1971), 0.28, mix(CHART.bw, "FFFFFF", 0.55));
    const lab = c === "jam" ? "с 1976 г. по н. в." : `${a}–${b}`;
    if (X(b) - X(a) > 1.0) tx(s, lab, { x: X(a) + 0.08, y, w: X(b) - X(a) - 0.1, h: 0.28, size: 9, bold: true, color: "FFFFFF", valign: "middle" });
    else tx(s, lab, { x: X(b) + 0.07, y, w: 1.0, h: 0.28, size: 9, bold: true, color: t.fg, valign: "middle" });
  });
  noteLine(s, t, "Каждая смена системы приходится на войну или глубокий кризис.", MX, 4.85, CW, 0.25, 9);
}

/* ================= 04 · Париж: рождение ================= */
{
  const t = T.gold, s = slide(t, 4, "I · Парижская система · 1867–1914", "Золото — единственные мировые деньги", "1867");
  facts(s, t, [["1816", "Великобритания первой переходит к золотому монометаллизму; с 1821 г. банкноты Банка Англии свободно меняются на золото"],
    ["1867", "Парижская монетная конференция признаёт золото единственной формой мировых денег"]], MX, 1.45, 3.9, 0.85);
  invBox(s, t, MX, 3.2, 3.9, 1.3, "Россия", "Реформа С. Ю. Витте 1895–1897 гг.: **1 рубль = 0,774234 г** чистого золота, кредитные билеты свободно меняются на золотую монету.", 10);
  const X0 = 4.8;
  label(s, t, "Как страны переходили к золоту", X0, 1.45, 4.6);
  const X = (y) => 4.95 + (y - 1815) / 90 * 4.35, base = 3.0;
  line(s, 4.8, base, W - MX, base, t.fg, 1.5);
  [1820, 1840, 1860, 1880, 1900].forEach((y) => line(s, X(y), base, X(y), base + 0.07, t.fg, 1.5));
  [1820, 1840, 1860].forEach((y) => tx(s, String(y), { x: X(y) - 0.3, y: base + 0.1, w: 0.6, h: 0.18, size: 8, bold: true, color: t.fg2, align: "center" }));
  rect(s, X(1867) - 0.08, base - 0.08, 0.16, 0.16, "8E3226", pres.shapes.DIAMOND);
  tx(s, "1867 · Парижская конференция", { x: X(1867) - 2.1, y: base + 0.36, w: 2.02, h: 0.2, size: 8.5, bold: true, color: t.fg, align: "right" });
  const pts = [[1821, "Великобритания 1821", 2.35, "l"], [1873, "Германия 1873", 1.9, "r"], [1876, "Латинский союз 1870-е", 3.72, "r"],
    [1879, "США 1879", 2.35, "l"], [1897, "Япония 1897", 1.9, "c"], [1897, "Россия 1897", 3.72, "c"]];
  pts.forEach(([y, text, ly, a]) => {
    const x = X(y), up = ly < base;
    line(s, x, up ? ly + 0.22 : base + 0.08, x, up ? base - 0.08 : ly - 0.02, t.line, 0.75);
    const w = 1.7, bx = a === "r" ? x - w + 0.05 : a === "l" ? x - 0.05 : x - w / 2;
    tx(s, text, { x: bx, y: ly, w, h: 0.2, size: 9.5, bold: true, color: t.fg, align: a === "r" ? "right" : a === "l" ? "left" : "center" });
  });
  pts.forEach(([y]) => rect(s, X(y) - 0.065, base - 0.065, 0.13, 0.13, t.fg, pres.shapes.OVAL));
  noteLine(s, t, "1870–1914 — «золотой век» золотого стандарта и первая глобализация.", X0, 4.25, 4.6, 0.25, 9);
}

/* ================= 05 · Париж: принципы ================= */
{
  const t = T.gold, s = slide(t, 5, "I · Парижская система · 1867–1914", "Шесть принципов золотомонетного стандарта", "1867");
  list(s, ["В обращении — **золотые монеты**, золото выполняет все функции денег", "У каждой валюты законодательно закреплено **золотое содержание**",
    "**Свободный размен** на золото, свободная чеканка, ввоз и вывоз золота", "Золото — **общепризнанные мировые деньги**: платёжное и резервное средство",
    "Денежная масса **жёстко привязана** к золотому запасу; дефицит платёжного баланса покрывается золотом",
    "Курсы складываются на рынке, но колеблются лишь в пределах **«золотых точек»**"], { x: MX, y: 1.45, w: 5.1, h: 3.6, size: 11, color: t.fg, numbered: true, gap: 7 });
  const x = 6.0, y = 1.6, w = W - MX - x;
  rect(s, x, y, w, 2.45, t.invBg);
  label(s, t, "Монетный паритет", x + 0.2, y + 0.18, w - 0.4, t.invFg);
  [["1 фунт стерлингов", "7,32238 г", 0.52], ["1 доллар США", "1,50463 г", 0.84]].forEach(([a, b, dy]) => {
    tx(s, a, { x: x + 0.2, y: y + dy, w: 1.8, h: 0.25, size: 10.5, color: t.invFg });
    tx(s, b, { x: x + w - 1.6, y: y + dy, w: 1.4, h: 0.25, size: 11, bold: true, color: t.invFg, align: "right" });
  });
  tx(s, "7,32238 ÷ 1,50463 (чистого золота)", { x: x + 0.2, y: y + 1.16, w: w - 0.4, h: 0.22, size: 8.5, color: t.invFg });
  line(s, x + 0.2, y + 1.52, x + w - 0.2, y + 1.52, t.invFg, 1.5);
  tx(s, "1 £ =", { x: x + 0.2, y: y + 1.68, w: 1, h: 0.5, size: 11, color: t.invFg, valign: "middle" });
  tx(s, "4,8665 $", { x: x + w - 2.3, y: y + 1.66, w: 2.1, h: 0.5, font: FB, size: 22, color: t.invFg, align: "right", valign: "middle" });
}

/* ================= 06 · Париж: механизм ================= */
{
  const t = T.gold, s = slide(t, 6, "I · Парижская система · механизм", "«Золотые точки» и саморегулирование", "1867");
  const par = 4.8665, c = 0.005, wave = [];
  for (let i = 0; i <= 120; i++) { const q = i / 120 * 13; wave.push(Math.sin(q * 1.05) * Math.cos(q * 0.31) + 0.28 * Math.sin(q * 3.3 + 0.6)); }
  const wmax = Math.max(...wave.map(Math.abs));
  const labels = wave.map((_, i) => String(i));
  const flat = (v) => wave.map(() => v);
  s.addChart(pres.charts.LINE, [
    { name: "Рыночный курс", labels, values: wave.map((w) => +(par + (w / wmax) * c * par * 0.88).toFixed(5)) },
    { name: "Паритет 4,8665", labels, values: flat(par) },
    { name: "Точка вывоза золота 4,8908", labels, values: flat(+(par * (1 + c)).toFixed(4)) },
    { name: "Точка ввоза золота 4,8422", labels, values: flat(+(par * (1 - c)).toFixed(4)) },
  ], {
    x: 0.4, y: 1.35, w: 5.3, h: 2.85, chartColors: ["1B1406", "8E3226", "5C4712", "5C4712"], lineSize: 2, lineDataSymbol: "none",
    catAxisHidden: true, valAxisMinVal: 4.83, valAxisMaxVal: 4.9, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0.00",
    valAxisLabelColor: t.fg2, valAxisLabelFontSize: 8, valAxisLabelFontFace: F, valAxisLineShow: false,
    valGridLine: { color: mix(t.bg, t.fg, 0.14), size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 8, legendFontFace: F, legendColor: t.fg,
    plotArea: { fill: { color: t.bg } }, chartArea: { fill: { color: t.bg }, roundedCorners: false },
  });
  tx(s, [{ text: "Пересылка золота стоит ", options: { bold: true } }, { text: "0,5 % паритета", options: { bold: true, fontFace: FB } }], { x: MX, y: 4.25, w: 5, h: 0.22, size: 10, color: t.fg });
  noteLine(s, t, "Если курс уходит дальше стоимости пересылки, выгоднее платить самим золотом, поэтому курс не выходит из коридора.", MX, 4.5, 5.1, 0.45, 9);
  const X = 6.1, WR = W - MX - X;
  label(s, t, "Механизм Д. Юма «цены — золото — поток»", X, 1.45, WR);
  ["Дефицит платёжного баланса", "Золото уходит из страны", "Денежная масса сокращается", "Цены и доходы снижаются",
    "Экспорт растёт, импорт падает — равновесие восстановлено"].forEach((text, i) => {
    const y = 1.72 + i * 0.5, last = i === 4, h = last ? 0.58 : 0.42;
    rect(s, X, y, WR, h, last ? t.invBg : t.tint);
    tx(s, String(i + 1), { x: X + 0.12, y, w: 0.3, h, font: FB, size: 11, color: last ? t.invFg : t.fg, valign: "middle" });
    tx(s, text, { x: X + 0.45, y, w: WR - 0.55, h, size: 10, color: last ? t.invFg : t.fg, valign: "middle" });
  });
  tx(s, "↺ ПРИ ПРОФИЦИТЕ — ВСЁ В ОБРАТНУЮ СТОРОНУ", { x: X, y: 4.45, w: WR, h: 0.2, size: 7.5, bold: true, cs: 0.5, color: t.fg2 });
}

/* ================= 07 · Париж: итог ================= */
{
  const t = T.gold, s = slide(t, 7, "I · Парижская система · итог", "Стабильность ценой гибкости", "1914");
  [["Достоинства", ["устойчивые валютные курсы и цены", "высокое доверие к валютам", "автоматическое выравнивание платёжных балансов", "быстрый рост мировой торговли и вывоза капитала"]],
   ["Недостатки", ["количество денег зависит от добычи золота, а не от нужд экономики", "дефляционная направленность", "нет самостоятельной денежно-кредитной политики", "внешние шоки бьют по ценам, производству и занятости"]]].forEach(([h, items], i) => {
    const x = MX + i * 4.55, w = 4.35;
    rect(s, x, 1.4, w, 1.95, t.tint);
    tx(s, h, { x: x + 0.2, y: 1.55, w: w - 0.4, h: 0.3, size: 13, bold: true, color: t.fg });
    list(s, items, { x: x + 0.2, y: 1.95, w: w - 0.4, h: 1.35, size: 10, color: t.fg, gap: 3 });
  });
  rect(s, MX, 3.55, CW, 1.2, t.invBg);
  tx(s, "1914", { x: MX + 0.2, y: 3.55, w: 1.7, h: 1.2, font: FB, size: 32, color: t.invFg, valign: "middle" });
  tx(s, "**Первая мировая война.** Воюющие страны прекращают размен банкнот на золото, запрещают его вывоз и печатают деньги для военных расходов. Золотые монеты уходят из обращения — золотомонетный стандарт перестаёт существовать.",
    { x: MX + 2.05, y: 3.55, w: CW - 2.25, h: 1.2, size: 10.5, color: t.invFg, valign: "middle", ls: 1.08 });
}

/* ================= 08 · Генуя: конференция ================= */
{
  const t = T.genoa, s = slide(t, 8, "II · Генуэзская система · 1922–1939", "Генуя, 1922: экономить золото", "1922");
  facts(s, t, [["1922", "10 апреля — 19 мая, Генуэзская международная экономическая конференция"],
    ["29+5", "государств и британских доминионов; Советская Россия (делегация Г. В. Чичерина); США — лишь наблюдатель"]], MX, 1.45, 4.2, 0.8);
  invBox(s, t, MX, 3.2, 4.2, 1.15, "Главная идея", "Вернуться к золоту, но хранить часть резервов в **девизах** — иностранных валютах, которые меняются на золото.", 10.5);
  const X = 5.2, WR = W - MX - X;
  label(s, t, "Принципы золотодевизного стандарта", X, 1.45, WR);
  list(s, ["Основа — **золото и девизы**", "Золотые паритеты сохраняются; размен на золото — **прямой** (США, Великобритания, Франция) или **через девизы** (Германия и ещё около 30 стран)",
    "Восстановлены **свободно колеблющиеся** курсы", "Регулирование — через **конференции и совещания**"], { x: X, y: 1.72, w: WR, h: 2.4, size: 10.5, color: t.fg, numbered: true, gap: 6 });
  noteLine(s, t, "Статус резервной валюты официально не закреплён: за лидерство спорят фунт стерлингов и доллар.", X, 4.2, WR, 0.4, 9);
}

/* ================= 09 · Генуя: формы размена ================= */
{
  const t = T.genoa, s = slide(t, 9, "II · Генуэзская система · формы размена", "Три способа добраться до золота", "1922");
  const cards = [["Золотомонетный", "США · до 1933 г.", [["банкнота"], ["золотая монета", 1]], "Свободный размен, как до войны."],
    ["Золотослитковый", "Великобритания 1925–1931 · Франция 1928–1936", [["банкнота"], ["слиток ≈ 12,4 кг", 1]], "В Лондоне слиток стоил ≈ 1700 £, в Париже — ≈ 215 тыс. франков. Размен доступен только крупным держателям."],
    ["Золотодевизный", "Германия, Австрия, Дания, Норвегия и др.", [["банкнота"], ["девиза: £ или $"], ["золото", 1]], "Золото доступно только через валюту другой страны."]];
  const cw = (CW - 0.4) / 3;
  cards.forEach(([h, who, chips, d], i) => {
    const x = MX + i * (cw + 0.2), y = 1.4;
    rect(s, x, y, cw, 2.25, t.tint);
    tx(s, h, { x: x + 0.18, y: y + 0.15, w: cw - 0.36, h: 0.28, size: 13, bold: true, color: t.fg });
    tx(s, who.toUpperCase(), { x: x + 0.18, y: y + 0.47, w: cw - 0.36, h: 0.32, size: 7, bold: true, cs: 0.5, color: t.fg2, ls: 1.1 });
    let cx = x + 0.18, cy = y + 0.86;
    chips.forEach(([c, au], k) => {
      const w = 0.16 + c.length * 0.066;
      if (cx + w > x + cw - 0.18) { cx = x + 0.18; cy += 0.33; }
      rect(s, cx, cy, w, 0.26, au ? t.accent : t.invBg);
      tx(s, c, { x: cx, y: cy, w, h: 0.26, size: 8.5, bold: true, color: au ? "2A1A02" : t.invFg, align: "center", valign: "middle" });
      cx += w;
      if (k < chips.length - 1) { tx(s, "→", { x: cx, y: cy, w: 0.2, h: 0.26, size: 9, bold: true, color: t.fg, align: "center", valign: "middle" }); cx += 0.2; }
    });
    tx(s, d, { x: x + 0.18, y: cy + 0.4, w: cw - 0.36, h: 0.9, size: 9, color: t.fg2, ls: 1.08 });
  });
  rect(s, MX, 3.85, CW, 0.85, t.invBg);
  tx(s, "**1925 — ошибка Черчилля.** Великобритания вернулась к довоенному паритету 4,86 $ за фунт. Фунт оказался переоценён, британский экспорт подорожал. Дж. М. Кейнс ответил памфлетом «Экономические последствия мистера Черчилля».",
    { x: MX + 0.2, y: 3.85, w: CW - 0.4, h: 0.85, size: 10, color: t.invFg, valign: "middle", ls: 1.08 });
}

/* ================= 10 · Генуя: распад ================= */
{
  const t = T.genoa, s = slide(t, 10, "II · Генуэзская система · распад", "Великая депрессия: бегство от золота", "1931");
  label(s, t, "Когда страны отказались от золотого стандарта", MX, 1.45, 5);
  const x0 = 2.05, x1 = 5.55, X = (d) => x0 + (d - 1931) / 6 * (x1 - x0);
  const data = [["Германия", 1931.54], ["Великобритания", 1931.72], ["Скандинавия", 1931.74], ["Япония", 1931.95], ["США", 1933.29], ["Бельгия", 1935.24],
    ["Франция", 1936.74], ["Нидерланды", 1936.74], ["Швейцария", 1936.74]];
  const y0 = 2.05, pitch = 0.3;
  for (let y = 1931; y <= 1937; y++) {
    line(s, X(y), 1.9, X(y), y0 + (data.length - 1) * pitch + 0.12, t.line, 0.5);
    if (y < 1937) tx(s, String(y), { x: X(y) + 0.03, y: 1.72, w: 0.5, h: 0.16, size: 8, bold: true, color: t.fg2 });
  }
  data.forEach(([n, d], i) => {
    const y = y0 + i * pitch;
    tx(s, n, { x: MX, y: y - 0.1, w: x0 - MX - 0.12, h: 0.2, size: 9.5, bold: true, color: t.fg, align: "right", valign: "middle" });
    line(s, x0, y, X(d), y, mix(t.bg, t.fg, 0.4), 2.5);
    rect(s, X(d) - 0.07, y - 0.07, 0.14, 0.14, t.mark, pres.shapes.OVAL);
  });
  const X2 = 6.0, WR = W - MX - X2;
  label(s, t, "Девальвация доллара, 1934", X2, 1.45, WR);
  tx(s, "20,67 → 35 $", { x: X2, y: 1.68, w: WR, h: 0.5, font: FB, size: 22, color: t.fg });
  noteLine(s, t, "за тройскую унцию: золотое содержание доллара сокращено на 41 %", X2, 2.22, WR, 0.4, 9);
  label(s, t, "Мир распадается на блоки", X2, 2.75, WR);
  list(s, ["стерлинговый — 1931", "долларовый — 1933", "золотой (Франция, Бельгия, Нидерланды, Швейцария, Италия, Польша) — 1933–1936"], { x: X2, y: 3.0, w: WR, h: 1.0, size: 10, color: t.fg, gap: 3 });
  noteLine(s, t, "Конкурентные девальвации, валютные ограничения, пошлины. Трёхстороннее соглашение 1936 г. (США, Великобритания, Франция) — первая попытка договориться.", X2, 4.05, WR, 0.65, 8.5);
}

/* ================= 11 · Бреттон-Вудс: конференция ================= */
{
  const t = T.bw, s = slide(t, 11, "III · Бреттон-Вудская система · 1944–1976", "Бреттон-Вудс, июль 1944: Кейнс против Уайта", "1944");
  const stats = [["1–22.07", "1944 г., штат Нью-Гэмпшир, США"], ["44", "государства, включая СССР"], ["≈ 70 %", "золотых запасов капиталистического мира — у США"], ["МВФ+МБРР", "созданы по итогам конференции"]];
  const sw = (CW - 0.45) / 4;
  stats.forEach(([a, b], i) => {
    const x = MX + i * (sw + 0.15);
    rect(s, x, 1.35, sw, 0.85, t.tint);
    tx(s, a, { x: x + 0.14, y: 1.44, w: sw - 0.28, h: 0.32, font: FB, size: 15, color: t.fg });
    tx(s, b, { x: x + 0.14, y: 1.78, w: sw - 0.28, h: 0.38, size: 8, color: t.fg2, ls: 1.05 });
  });
  const cell = (text, o = {}) => ({ text, options: { fontFace: F, fontSize: 9.5, color: t.fg, valign: "middle", margin: [3, 6, 3, 6], ...o } });
  const k = (text) => cell(text.toUpperCase(), { fontSize: 7.5, bold: true, color: t.fg2 });
  const rows = [
    [cell(""), cell([{ text: "План Кейнса", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "ВЕЛИКОБРИТАНИЯ", options: { fontSize: 7, bold: true, color: t.fg2 } }]),
      cell([{ text: "План Уайта", options: { bold: true, fontSize: 13, breakLine: true } }, { text: "США · ПРИНЯТ В ОСНОВНОМ", options: { fontSize: 7, bold: true, color: t.accent } }])],
    [k("Институт"), cell("Международный клиринговый союз"), cell("Стабилизационный фонд")],
    [k("Резервы"), cell("наднациональная валюта «банкор»"), cell("золото и доллар США")],
    [k("Ресурсы"), cell("кредит по клиринговым счетам"), cell("взносы (квоты) стран-участниц")],
    [k("Дисбалансы"), cell("устраняют и должники, и кредиторы"), cell("устраняют прежде всего страны с дефицитом")],
  ];
  rows.forEach((r, i) => r.forEach((c) => { c.options.fill = { color: i % 2 === 1 ? t.tint : t.bg }; }));
  s.addTable(rows, { x: MX, y: 2.4, w: CW, colW: [1.5, (CW - 1.5) / 2, (CW - 1.5) / 2], rowH: [0.5, 0.36, 0.36, 0.36, 0.36], border: { type: "none" } });
  noteLine(s, t, "СССР участвовал в конференции, но соглашения не ратифицировал. Россия — член МВФ с 1 июня 1992 г.", MX, 4.55, CW, 0.25, 9);
}

/* ================= 12 · Бреттон-Вудс: устройство ================= */
{
  const t = T.bw, s = slide(t, 12, "III · Бреттон-Вудская система · устройство", "Золото — доллар — все остальные", "1944");
  const cx = 2.75;
  rect(s, cx - 0.85, 1.35, 1.7, 0.6, "F5C451");
  tx(s, "Золото", { x: cx - 0.85, y: 1.38, w: 1.7, h: 0.32, size: 14, bold: true, color: "16336F", align: "center" });
  tx(s, "1 тройская унция = 31,1035 г", { x: cx - 0.85, y: 1.68, w: 1.7, h: 0.2, size: 7.5, bold: true, color: "16336F", align: "center" });
  line(s, cx, 1.98, cx, 2.43, t.fg, 1.5, { begin: true, end: true });
  tx(s, "35 $ за унцию", { x: cx + 0.12, y: 2.0, w: 2.0, h: 0.2, size: 10, bold: true, color: t.accent });
  tx(s, "размен только для центральных банков", { x: cx + 0.12, y: 2.2, w: 2.2, h: 0.2, size: 7.5, color: t.fg2 });
  rect(s, cx - 0.36, 2.47, 0.72, 0.72, "FFFFFF", pres.shapes.OVAL);
  tx(s, "$", { x: cx - 0.36, y: 2.47, w: 0.72, h: 0.72, font: FB, size: 22, color: "16336F", align: "center", valign: "middle" });
  tx(s, "Доллар США", { x: cx + 0.47, y: 2.6, w: 1.8, h: 0.22, size: 10, bold: true, color: t.fg });
  tx(s, "резервная валюта и валюта интервенций", { x: cx + 0.47, y: 2.82, w: 1.9, h: 0.3, size: 7.5, color: t.fg2 });
  const boxes = [[1.1, "Фунт стерлингов", "1 £ = 2,80 $", "1949–1967"], [2.75, "Марка ФРГ", "4,20 DM = 1 $", "1949–1961"], [4.4, "Японская иена", "360 ¥ = 1 $", "1949–1971"]];
  boxes.forEach(([bx, a, b, c]) => {
    line(s, cx + (bx - cx) * 0.18, 3.2, bx, 3.85, t.fg, 1.5, { end: true });
    rect(s, bx - 0.72, 3.9, 1.44, 0.8, t.tint);
    tx(s, a, { x: bx - 0.72, y: 3.97, w: 1.44, h: 0.18, size: 7.5, color: t.fg2, align: "center" });
    tx(s, b, { x: bx - 0.72, y: 4.15, w: 1.44, h: 0.28, size: 11, bold: true, color: t.fg, align: "center" });
    tx(s, c, { x: bx - 0.72, y: 4.44, w: 1.44, h: 0.18, size: 7.5, bold: true, color: t.fg2, align: "center" });
  });
  tx(s, "±1 %", { x: 1.2, y: 3.36, w: 0.5, h: 0.18, size: 8, bold: true, color: t.fg2, align: "right" });
  tx(s, "±1 %", { x: cx + 0.06, y: 3.5, w: 0.5, h: 0.18, size: 8, bold: true, color: t.fg2 });
  tx(s, "±1 %", { x: 3.8, y: 3.36, w: 0.5, h: 0.18, size: 8, bold: true, color: t.fg2 });
  const X = 5.35, WR = W - MX - X;
  label(s, t, "Принципы", X, 1.4, WR);
  list(s, ["**Золотодевизный стандарт** на основе золота и двух резервных валют — доллара и фунта; на деле — золотодолларовый",
    "Только доллар меняется на золото: **35 $ за тройскую унцию** для центральных банков и правительств",
    "**Фиксированные курсы**: отклонение от паритета не более ±1 %, ЦБ поддерживают курс интервенциями в долларах",
    "Изменение паритета более чем на 10 % — **только с согласия МВФ**",
    "Регулирование через **МВФ** (кредиты для покрытия дефицита платёжного баланса) и **МБРР**"], { x: X, y: 1.65, w: WR, h: 2.95, size: 9.5, color: t.fg, numbered: true, gap: 5 });
  noteLine(s, t, "Внешняя конвертируемость: Западная Европа — конец 1958 г., Япония — 1964 г.", X, 4.6, WR, 0.3, 8.5);
}

/* ================= 13 · Бреттон-Вудс: Триффин ================= */
{
  const t = T.bw, s = slide(t, 13, "III · Бреттон-Вудская система · противоречие", "Дилемма Триффина", "1960");
  const wl = 4.3;
  rect(s, MX, 1.4, wl, 0.55, t.invBg);
  tx(s, "Мировой экономике нужны доллары, а получить их она может только из США", { x: MX + 0.16, y: 1.4, w: wl - 0.32, h: 0.55, size: 10, bold: true, color: t.invFg, valign: "middle" });
  const fw = (wl - 0.1) / 2;
  [["США сохраняют дефицит платёжного баланса", "Долларов в мире становится больше…", "…но доверие к их размену на золото падает"],
   ["США устраняют дефицит", "Доверие к доллару сохраняется…", "…но миру не хватает ликвидности"]].forEach(([a, b, c], i) => {
    const x = MX + i * (fw + 0.1);
    rect(s, x, 2.03, fw, 1.35, t.tint);
    tx(s, [{ text: a, options: { bold: true, color: t.fg, breakLine: true } }, { text: b, options: { color: t.fg2, breakLine: true } }, { text: c, options: { bold: true, color: t.accent } }],
      { x: x + 0.13, y: 2.12, w: fw - 0.26, h: 1.2, size: 9, ls: 1.12 });
  });
  tx(s, "«Непомерная привилегия»", { x: MX, y: 3.55, w: wl, h: 0.3, size: 13, bold: true, color: t.fg });
  noteLine(s, t, "так министр финансов Франции В. Жискар д'Эстен назвал право США покрывать дефицит своей валютой", MX, 3.87, wl, 0.4, 8.5);
  noteLine(s, t, "Р. Триффин, «Золото и долларовый кризис», 1960.", MX, 4.35, wl, 0.2, 8.5);
  const X = 5.15, WR = W - MX - X;
  label(s, t, "Золотой запас США, млрд долл. по цене 35 $ за унцию", X, 1.4, WR);
  s.addChart(pres.charts.BAR, [{ name: "Золотой запас США", labels: ["1949", "1957", "1960", "1965", "1968", "1971"], values: [24.6, 22.9, 17.8, 13.8, 10.9, 10.2] }], {
    x: X - 0.1, y: 1.6, w: WR + 0.1, h: 2.75, barDir: "col", chartColors: ["F5C451"], barGapWidthPct: 70,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: t.fg, dataLabelFontSize: 9, dataLabelFontBold: true, dataLabelFontFace: F, dataLabelFormatCode: "0.0",
    catAxisLabelColor: t.fg2, catAxisLabelFontSize: 8, catAxisLabelFontFace: F, catAxisLineShow: false,
    valAxisLabelColor: t.fg2, valAxisLabelFontSize: 8, valAxisLabelFontFace: F, valAxisMinVal: 0, valAxisMaxVal: 27, valAxisMajorUnit: 5, valAxisLineShow: false,
    valGridLine: { color: mix(t.bg, t.fg, 0.16), size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
    plotArea: { fill: { color: t.bg } }, chartArea: { fill: { color: t.bg }, roundedCorners: false },
  });
  noteLine(s, t, "К 1960 г. долларовые обязательства США перед иностранцами превысили весь золотой запас страны.", X, 4.42, WR, 0.4, 8.5);
}

/* ================= 14 · Бреттон-Вудс: хроника ================= */
{
  const t = T.bw, s = slide(t, 14, "III · Бреттон-Вудская система · распад", "Хроника распада, 1960–1973", "1971");
  const items = [["10.1960", "Цена золота в Лондоне подскакивает до 40 $ за унцию"], ["1961", "«Золотой пул» восьми центральных банков удерживает цену на 35 $"],
    ["1965", "Франция при Ш. де Голле меняет доллары на золото"], ["11.1967", "Девальвация фунта: 2,80 → 2,40 $ (−14,3 %)"],
    ["03.1968", "Распад пула, «двухъярусный» рынок: 35 $ для ЦБ, свободная цена для остальных"], ["1969", "Созданы СДР (первая поправка к Уставу МВФ); распределения — с 1970 г."],
    ["15.08.1971", "Р. Никсон приостанавливает обмен долларов на золото", 1], ["12.1971", "Смитсоновское соглашение: 38 $ за унцию, коридор ±2,25 %"],
    ["02.1973", "Вторая девальвация доллара: 42,22 $ за унцию"], ["03.1973", "Ведущие валюты переходят к плаванию"]];
  const cw = (CW - 0.2) / 2;
  items.forEach(([d, text, key], i) => {
    const col = Math.floor(i / 5), row = i % 5, x = MX + col * (cw + 0.2), y = 1.38 + row * 0.56;
    rect(s, x, y, cw, 0.5, key ? t.invBg : t.tint);
    tx(s, d, { x: x + 0.12, y, w: 1.05, h: 0.5, font: FB, size: 8.5, color: key ? t.invFg : t.accent, valign: "middle" });
    tx(s, text, { x: x + 1.2, y, w: cw - 1.32, h: 0.5, size: 9.5, bold: !!key, color: key ? t.invFg : t.fg, valign: "middle", ls: 1.05 });
  });
  noteLine(s, t, "Причины: дилемма Триффина, дефицит платёжного баланса и инфляция в США (война во Вьетнаме), усиление Европы и Японии, рынок евродолларов и «горячие деньги».", MX, 4.25, CW, 0.4, 9);
}

/* ================= 15 · Ямайка: принципы ================= */
{
  const t = T.jam, s = slide(t, 15, "IV · Ямайская система · с 1976 г.", "Кингстон, январь 1976: золото больше не деньги", "1976");
  facts(s, t, [["1976", "7–8 января, совещание Временного комитета МВФ в Кингстоне (Ямайка)"], ["1978", "1 апреля вступает в силу Вторая поправка к Статьям Соглашения МВФ"]], MX, 1.45, 3.9, 0.8);
  invBox(s, t, MX, 3.2, 3.9, 1.1, "Золото МВФ", "25 млн унций продано на аукционах, ещё 25 млн возвращено странам-членам — всего треть запаса фонда.", 10);
  const X = 4.85, WR = W - MX - X;
  label(s, t, "Принципы", X, 1.45, WR);
  list(s, ["Вместо золотодевизного — **стандарт СДР**; на практике многовалютный стандарт во главе с долларом",
    "**Демонетизация золота**: отменены официальная цена и золотые паритеты, ЦБ торгуют золотом по рыночной цене",
    "**Свободный выбор** режима валютного курса", "**Надзор МВФ** за курсовой политикой стран (статья IV)",
    "Признаны **региональные валютные системы**: ЕВС и ЭКЮ (1979), евро (1999)"], { x: X, y: 1.72, w: WR, h: 3.1, size: 10.5, color: t.fg, numbered: true, gap: 6 });
}

/* ================= 16 · Ямайка: режимы ================= */
{
  const t = T.jam, s = slide(t, 16, "IV · Ямайская система · курсовые режимы", "Свобода выбора: спектр курсовых режимов", "1976");
  const ws = [1, 1.2, 1].map((k) => k / 3.2 * CW);
  const zones = [["Жёсткая фиксация", "0B7A75", "FFFFFF"], ["Мягкая фиксация", "6FBDB7", "0E1A1F"], ["Плавающие курсы", "CFE8E5", "0E1A1F"]];
  const cols = [["Чужая валюта как законное платёжное средство; валютное управление (currency board)", ["**Эквадор, Панама** — доллар США", "**Гонконг** — валютное управление с 1983 г."]],
    ["Привязка к валюте или корзине; привязка в пределах коридора; ползущая привязка", ["**Саудовская Аравия** — риал привязан к доллару", "**Дания** — крона в коридоре ±2,25 % к евро (ERM II)"]],
    ["Управляемое плавание; свободное плавание", ["**Россия** — плавающий рубль с ноября 2014 г.", "**США, еврозона, Япония, Великобритания**"]]];
  let x = MX;
  zones.forEach(([z, bg, fg], i) => {
    rect(s, x, 1.4, ws[i], 0.45, bg);
    tx(s, z, { x: x + 0.14, y: 1.4, w: ws[i] - 0.2, h: 0.45, size: 12, bold: true, color: fg, valign: "middle" });
    const cw = ws[i] - 0.2;
    label(s, t, "Режимы", x, 2.25, cw);
    tx(s, cols[i][0], { x, y: 2.47, w: cw, h: 0.62, size: 9.5, color: t.fg, ls: 1.05 });
    cols[i][1].forEach((e, k) => {
      rect(s, x, 3.15 + k * 0.5, cw, 0.44, t.tint);
      tx(s, e, { x: x + 0.1, y: 3.15 + k * 0.5, w: cw - 0.2, h: 0.44, size: 8.5, color: t.fg, valign: "middle", ls: 1.02 });
    });
    x += ws[i];
  });
  tx(s, "← УСТОЙЧИВЕЕ КУРС", { x: MX, y: 1.92, w: 3, h: 0.2, size: 7.5, bold: true, cs: 0.5, color: t.fg2 });
  tx(s, "САМОСТОЯТЕЛЬНЕЕ ДЕНЕЖНАЯ ПОЛИТИКА →", { x: W - MX - 4, y: 1.92, w: 4, h: 0.2, size: 7.5, bold: true, cs: 0.5, color: t.fg2, align: "right" });
  noteLine(s, t, "Классификация режимов — по МВФ (Annual Report on Exchange Arrangements and Exchange Restrictions).", MX, 4.35, CW, 0.2, 8);
}

/* ================= 17 · Ямайка: сегодня ================= */
{
  const t = T.jam, s = slide(t, 17, "IV · Ямайская система · сегодня", "Формально — СДР, фактически — доллар", "2026");
  [["00908A", "Вес в корзине СДР (с 01.08.2022), %"], ["0E1A1F", "Доля в мировых валютных резервах (I кв. 2026), %"]].forEach(([c, text], i) => {
    rect(s, MX, 1.42 + i * 0.22, 0.16, 0.1, c);
    tx(s, text, { x: MX + 0.24, y: 1.37 + i * 0.22, w: 4.5, h: 0.2, size: 8.5, color: t.fg2 });
  });
  const cats = ["Доллар США", "Евро", "Юань", "Иена", "Фунт стерлингов"].reverse();
  s.addChart(pres.charts.BAR, [
    { name: "Доля в мировых валютных резервах", labels: cats, values: [57.1, 20.0, 2.0, 5.4, 4.6].reverse() },
    { name: "Вес в корзине СДР", labels: cats, values: [43.38, 29.31, 12.28, 7.59, 7.44].reverse() },
  ], {
    x: 0.4, y: 1.8, w: 5.35, h: 2.6, barDir: "bar", barGrouping: "clustered", chartColors: ["0E1A1F", "00908A"], barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: t.fg, dataLabelFontSize: 8, dataLabelFontFace: F, dataLabelFormatCode: "0.0#",
    catAxisLabelColor: t.fg, catAxisLabelFontSize: 9, catAxisLabelFontFace: F, catAxisLineShow: false,
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 66, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    plotArea: { fill: { color: t.bg } }, chartArea: { fill: { color: t.bg }, roundedCorners: false },
  });
  noteLine(s, t, "Фунт — около 4,6 %; прочие валюты — около 10,8 % резервов. Источники: МВФ (пересмотр корзины СДР 2022 г.; COFER, I кв. 2026 г.).", MX, 4.45, 5.2, 0.4, 8);
  const X = 6.1, WR = W - MX - X;
  label(s, t, "Цена золота за тройскую унцию", X, 1.45, WR);
  tx(s, "35 $", { x: X, y: 1.7, w: 0.85, h: 0.3, font: FB, size: 15, color: t.fg2, strike: true, valign: "middle" });
  tx(s, "официальная цена до 1971 г.", { x: X + 0.9, y: 1.7, w: WR - 0.9, h: 0.3, size: 9, color: t.fg2, valign: "middle" });
  tx(s, "≈ 4 250 $", { x: X, y: 2.05, w: WR, h: 0.62, font: FB, size: 30, color: "8A5F00", valign: "middle" });
  noteLine(s, t, "сентябрь 2026 · в ≈ 120 раз выше · в начале 2026 г. — выше 5 000 $", X, 2.72, WR, 0.4, 8.5);
  invBox(s, t, X, 3.2, WR, 1.05, null, "**289 т** золота купили центральные банки во II квартале 2026 г. — рекорд для второго квартала. В 2022–2024 гг. — более 1000 т в год.", 9.5);
  noteLine(s, t, "World Gold Council; Trading Economics.", X, 4.35, WR, 0.2, 8);
}

/* ================= 18 · Сравнение ================= */
{
  const t = T.plain, s = slide(t, 18, "Итоги · Сравнение", "Четыре системы в одной таблице");
  const head = (name, yrs, c) => ({ text: [{ text: name, options: { bold: true, fontSize: 11, breakLine: true } }, { text: yrs, options: { fontSize: 7, bold: true } }],
    options: { fill: { color: ERA[c] }, color: c === "paris" ? "1B1406" : "FFFFFF", fontFace: F, valign: "middle", margin: [4, 6, 4, 6] } });
  const k = (text) => ({ text: text.toUpperCase(), options: { fontFace: F, fontSize: 7, bold: true, color: t.fg2, fill: { color: t.bg }, valign: "top", margin: [5, 4, 3, 4] } });
  const c = (text) => ({ text, options: { fontFace: F, fontSize: 8.5, color: t.fg, fill: { color: t.tint }, valign: "top", margin: [5, 6, 4, 6] } });
  const rows = [
    [{ text: "", options: { fill: { color: t.bg } } }, head("Парижская", "1867–1914", "paris"), head("Генуэзская", "1922–1939", "genoa"), head("Бреттон-Вудская", "1944–1976", "bw"), head("Ямайская", "с 1976 г.", "jam")],
    [k("Стандарт"), c("Золотомонетный"), c("Золотодевизный"), c("Золотодолларовый"), c("СДР, фактически многовалютный")],
    [k("Роль золота"), c("Единственные мировые деньги, монеты в обращении"), c("Основа наряду с девизами; размен на слитки или через девизы"), c("Размен долларов ЦБ по 35 $ за унцию"), c("Демонетизировано; резервный актив ЦБ")],
    [k("Резервные валюты"), c("Официально нет; ведущая роль фунта"), c("Не закреплены; фунт и доллар"), c("Доллар и фунт"), c("Доллар, евро, иена, фунт, юань")],
    [k("Курсы"), c("В пределах «золотых точек»"), c("Свободно колеблющиеся"), c("Фиксированные, ±1 %"), c("Свободный выбор, преобладает плавание")],
    [k("Регулирование"), c("Рыночный автоматизм"), c("Конференции и совещания"), c("МВФ и МБРР"), c("Надзор МВФ, G7, G20, региональные союзы")],
    [k("Почему рухнула"), c("Первая мировая война, узость золотой базы"), c("Кризис 1929–1933, валютные блоки"), c("Дилемма Триффина, дефицит США"), c("Действует; нестабильность курсов, доминирование доллара")],
  ];
  const cw = (CW - 1.2) / 4;
  s.addTable(rows, { x: MX, y: 1.35, w: CW, colW: [1.2, cw, cw, cw, cw], rowH: [0.46, 0.3, 0.5, 0.42, 0.42, 0.42, 0.5], border: { type: "solid", pt: 2, color: "FFFFFF" } });
}

/* ================= 19 · Выводы ================= */
{
  const t = T.plain, s = slide(t, 19, "Итоги · Выводы", "Четыре сквозные тенденции");
  const arrow = { text: "  →  ", options: { bold: true, color: t.accent } };
  const chain = (parts) => parts.flatMap((p, i) => (i ? [{ ...arrow, options: { ...arrow.options } }] : []).concat([{ text: p, options: {} }]));
  [["Золото", ["единственные мировые деньги", "основа наряду с валютами", "юридически демонетизировано"]],
   ["Регулирование", ["рыночный автоматизм", "конференции", "МВФ, G7, G20"]],
   ["Курсы", ["«золотые точки»", "фиксированные паритеты", "свобода выбора"]],
   ["Лидер", ["фунт стерлингов", "доллар США", "многополярность: евро, юань, национальные валюты?"]]].forEach(([k, parts], i) => {
    const y = 1.4 + i * 0.55;
    rect(s, MX, y, CW, 0.48, t.tint);
    tx(s, k, { x: MX + 0.18, y, w: 1.6, h: 0.48, size: 11, bold: true, color: t.fg, valign: "middle" });
    tx(s, chain(parts), { x: MX + 1.9, y, w: CW - 2.05, h: 0.48, size: 10, color: t.fg, valign: "middle" });
  });
  rect(s, MX, 3.75, CW, 0.9, t.invBg);
  tx(s, "Валютная система живёт, пока её правила совпадают с реальным распределением экономической силы.", { x: MX + 0.25, y: 3.75, w: CW - 0.5, h: 0.9, size: 15, bold: true, color: t.invFg, valign: "middle", ls: 1.05 });
}

/* ================= 20 · Спасибо ================= */
{
  const t = T.gold, s = slide(t, 20, "Спасибо", null, "Au");
  tx(s, "Спасибо за внимание", { x: MX, y: 1.7, w: 4.4, h: 1.7, size: 40, bold: true, color: t.fg, ls: 0.92, valign: "middle" });
  const X = 5.3, WR = W - MX - X;
  label(s, t, "Источники", X, 1.25, WR);
  list(s, ["Международные валютно-кредитные и финансовые отношения / под ред. Л. Н. Красавиной. — 3-е изд. — М.: Финансы и статистика, 2005.",
    "Моисеев С. Р. Международные валютно-кредитные отношения. — 2-е изд. — М.: Дело и Сервис, 2007.",
    "Эйхенгрин Б. Непомерная привилегия: взлёт и падение доллара. — М.: Изд-во Института Гайдара, 2013.",
    "Eichengreen B. Globalizing Capital. — 3rd ed. — Princeton UP, 2019.", "Triffin R. Gold and the Dollar Crisis. — Yale UP, 1960.",
    "Steil B. The Battle of Bretton Woods. — Princeton UP, 2013.", "IMF: Articles of Agreement; Review of the Method of Valuation of the SDR (2022); COFER.",
    "World Gold Council: The Bretton Woods System; Central bank gold statistics (2026)."], { x: X, y: 1.5, w: WR, h: 3.4, size: 8.5, color: t.fg2, numbered: true, gap: 4, indent: 14 });
}

// pptxgenjs ставит <a:pPr> перед каждым фрагментом абзаца; оставляем только первый — иначе ломаются нумерация и отступы
const JSZip = require("jszip");
const fixParagraphs = (xml) => xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, body) => {
  let seen = false;
  return "<a:p>" + body.replace(/<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g, (pp) => (seen ? "" : ((seen = true), pp))) + "</a:p>";
});
pres.write({ outputType: "nodebuffer" }).then(async (buf) => {
  const zip = await JSZip.loadAsync(buf);
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    zip.file(name, fixParagraphs(await zip.file(name).async("string")));
  }
  require("fs").writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("written", OUT);
});

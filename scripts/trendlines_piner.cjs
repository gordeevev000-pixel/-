// Прогон pine/AUTO_TRENDLINES.pine независимым движком Pine v6 (piner) на минутках
// из web/data.js. Пишет JSON: живые на последнем баре линии и маркеры plotshape.
// Его сравнивает с Python-копией scripts/trendlines_check.py --piner.
//
// Установка движка в отдельную папку (не в репозиторий):
//   mkdir -p /tmp/piner && cd /tmp/piner && npm init -y && npm install --ignore-scripts @heyphat/piner
// Запуск:
//   PINER_DIR=/tmp/piner node scripts/trendlines_piner.cjs pine/AUTO_TRENDLINES.pine web/data.js out.json [баров]
// Настройки индикатора можно переопределить: PINER_INPUTS='{"Строить по": "Тела"}'
const path = require('path');
const fs = require('fs');

const piner = require(path.join(process.env.PINER_DIR || '.', 'node_modules/@heyphat/piner/dist/index.cjs'));
const [src, dataJs, out, upto] = process.argv.slice(2);

let compiled;
try {
  compiled = piner.compile(fs.readFileSync(src, 'utf8'));
} catch (e) {
  console.error('ошибка компиляции:', e.message);
  process.exit(1);
}

const s = fs.readFileSync(dataJs, 'utf8');
const d = JSON.parse(s.slice(s.indexOf('{')).trim().replace(/;$/, '')).candles;
const n = upto ? parseInt(upto, 10) : d.t.length;
const bars = [];
for (let i = 0; i < n; i++) {
  bars.push({ time: d.t[i] * 1000, open: d.o[i], high: d.h[i], low: d.l[i], close: d.c[i], volume: 1 });
}

(async () => {
  const inputs = process.env.PINER_INPUTS ? JSON.parse(process.env.PINER_INPUTS) : undefined;
  const eng = new piner.Engine(compiled, new piner.ArrayFeed(bars), { inputs });
  await eng.run({ symbol: 'BTCUSDT', timeframe: '1' });
  const markers = {};
  for (const [, m] of eng.outputs.markers) {
    markers[m.title] = [];
    m.data.forEach((v, i) => { if (v !== null && v !== undefined) markers[m.title].push(i); });
  }
  const lines = eng.drawings.filter(o => o.type === 'line').map(o => o.props);
  fs.writeFileSync(out, JSON.stringify({ bars: n, lines, markers }));
})().catch(e => {
  console.error('ошибка исполнения:', e && e.stack || e);
  process.exit(2);
});

// PDF презентации: 20 страниц 16:9, по одной на слайд. Запуск: node build_pdf.js (нужен Playwright с Chromium)
const { chromium } = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
  await p.goto('file://' + require('path').join(__dirname, '..', 'site', 'index.html')); await p.waitForTimeout(1000);
  await p.addStyleTag({ content: `
    @page { size: 1600px 900px; margin: 0; }
    html, body { margin: 0 !important; padding: 0 !important; background: #fff !important; }
    .top, .howto, .pbar, #tip { display: none !important; }
    .deck { max-width: none !important; margin: 0 !important; display: block !important; }
    .slide { width: 1600px; height: 900px; break-after: page; page-break-after: always; overflow: hidden; }
    .slide:last-child { break-after: auto; page-break-after: auto; }
    .frame { width: 1600px; }
    .slide-inner { animation: none !important; height: 100vh; aspect-ratio: auto !important; break-inside: avoid; }
    .ghost { bottom: 0 !important; line-height: .74 !important; }
    .slider input { display: none !important; }
    .slider { justify-content: flex-start; }
    .slider output { min-width: 0; text-align: left; }
  ` });
  await p.emulateMedia({ media: 'print' });
  await p.waitForTimeout(300);
  await p.pdf({ path: require('path').join(__dirname, '..', 'prezentaciya.pdf'), width: '1600px', height: '900px', printBackground: true, preferCSSPageSize: true });
  await b.close();
})();

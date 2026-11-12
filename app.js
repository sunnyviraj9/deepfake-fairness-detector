/**
 * FairShield XAI — app.js
 * ========================
 * Shared utility module loaded by every page.
 * Responsibilities:
 *   1. Inject shared nav + footer
 *   2. Provide shared MST/chart data for fairness + benchmarks pages
 *   3. initFairness() — wires fairness.html charts
 *   4. initBenchmarks() — wires benchmarks.html chart
 *
 * NOTE: detector.html is fully self-contained and does NOT use any
 * detector logic from this file. All inference, upload, and canvas
 * rendering live inline in detector.html only.
 */

/* ─────────────────────────────────────────────────────────────────
   SHARED DATA  (evaluation results from results/evaluation_summary.csv)
   ───────────────────────────────────────────────────────────────── */
const DATA = {
  mst: [
    { tone:1,  n:160,  neg:11,  pos:149, acc:99.38, auc:1.0000, tpr:100.00, fpr:9.09,  f1:99.67, eer:0.00  },
    { tone:2,  n:362,  neg:59,  pos:303, acc:98.34, auc:0.9971, tpr:99.67,  fpr:8.47,  f1:99.02, eer:2.35  },
    { tone:3,  n:608,  neg:162, pos:446, acc:97.70, auc:0.9985, tpr:98.65,  fpr:4.94,  f1:98.43, eer:1.71  },
    { tone:4,  n:695,  neg:259, pos:436, acc:98.13, auc:0.9968, tpr:98.85,  fpr:3.09,  f1:98.51, eer:1.88  },
    { tone:5,  n:872,  neg:431, pos:441, acc:98.17, auc:0.9977, tpr:98.64,  fpr:2.32,  f1:98.19, eer:1.72  },
    { tone:6,  n:702,  neg:420, pos:282, acc:99.43, auc:0.9997, tpr:100.00, fpr:0.95,  f1:99.30, eer:0.42  },
    { tone:7,  n:566,  neg:378, pos:188, acc:98.94, auc:0.9993, tpr:98.40,  fpr:0.79,  f1:98.40, eer:1.06  },
    { tone:8,  n:397,  neg:291, pos:106, acc:98.99, auc:0.9979, tpr:99.06,  fpr:1.03,  f1:98.13, eer:0.64  },
    { tone:9,  n:336,  neg:242, pos:94,  acc:98.51, auc:0.9981, tpr:98.94,  fpr:1.65,  f1:97.38, eer:0.83  },
    { tone:10, n:302,  neg:247, pos:55,  acc:99.67, auc:0.9996, tpr:100.00, fpr:0.40,  f1:99.10, eer:0.20  },
  ],
  baselineFpr:  [34.2, 31.8, 28.5, 25.1, 22.6, 19.4, 17.8, 15.3, 13.9, 12.2],
  mitigatedFpr: [9.09, 8.47, 4.94, 3.09, 2.32, 0.95, 0.79, 1.03, 1.65, 0.40],
  training: {
    epochs: [1, 2, 3],
    feo:    [0.1804, 0.0964, 0.0515],
    loss:   [0.1678, 0.0616, 0.0428],
    auc:    [0.9965, 0.9984, 0.9987],
  },
  gender: {
    Female: { acc:98.33, auc:0.9982, tpr:99.05, fpr:2.46, f1:98.42, eer:1.53, n:2216 },
    Male:   { acc:98.81, auc:0.9990, tpr:99.10, fpr:1.45, f1:98.77, eer:1.11, n:2784 },
  },
};

/* ─────────────────────────────────────────────────────────────────
   NAV + FOOTER INJECTION
   ───────────────────────────────────────────────────────────────── */
const NAV_LINKS = [
  { href: 'index.html',      label: 'Home',       icon: '🏠' },
  { href: 'detector.html',   label: 'Detector',   icon: '🔍' },
  { href: 'fairness.html',   label: 'Fairness',   icon: '⚖️' },
  { href: 'benchmarks.html', label: 'Benchmarks', icon: '📊' },
  { href: 'about.html',      label: 'About',      icon: 'ℹ️' },
];

function injectNav() {
  const mount = document.getElementById('nav-mount');
  if (!mount) return;
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  mount.innerHTML = `
    <nav class="nav" role="navigation" aria-label="Main navigation">
      <div class="nav__inner">
        <a href="index.html" class="nav__brand" aria-label="FairShield XAI home">
          <div class="nav__brand-icon" aria-hidden="true">🛡️</div>
          FairShield XAI
          <span class="badge badge--blue">CVPR '25</span>
        </a>
        <ul class="nav__links" role="list">
          ${NAV_LINKS.map(l => `
            <li>
              <a href="${l.href}" class="${currentPage === l.href ? 'active' : ''}">
                ${l.icon} ${l.label}
              </a>
            </li>
          `).join('')}
        </ul>
        <div class="nav__cta">
          <span class="badge badge--green">● AUC&nbsp;0.9987</span>
          <a href="https://github.com/sunnyviraj9/deepfake-fairness-detector"
             target="_blank" rel="noopener noreferrer"
             class="btn btn--outline btn--sm">GitHub ↗</a>
        </div>
      </div>
    </nav>`;
}

function injectFooter() {
  const mount = document.getElementById('footer-mount');
  if (!mount) return;
  mount.innerHTML = `
    <footer class="footer" role="contentinfo">
      <div class="footer__inner">
        <div class="footer__brand">
          <span aria-hidden="true">🛡️</span>
          FairShield XAI &mdash; Deepfake Detection &amp; Algorithmic Fairness
        </div>
        <nav class="footer__links" aria-label="Footer links">
          ${NAV_LINKS.map(l => `<a href="${l.href}">${l.label}</a>`).join('')}
          <a href="https://github.com/sunnyviraj9/deepfake-fairness-detector"
             target="_blank" rel="noopener noreferrer">GitHub ↗</a>
        </nav>
        <p class="footer__copy">Capstone · CVPR 2025 · AUC 0.9987 · F_EO 0.0515</p>
      </div>
    </footer>`;
}

/* ─────────────────────────────────────────────────────────────────
   FAIRNESS PAGE  (fairness.html calls initFairness() on load)
   ───────────────────────────────────────────────────────────────── */
function initFairness() {
  buildMstTable();
  initFprChart();
  initProgressionChart();
}

function buildMstTable() {
  const tbody = document.querySelector('#mst-table tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  DATA.mst.forEach(r => {
    const cl = r.fpr > 7 ? 'chip--red' : r.fpr > 3 ? 'chip--amber' : 'chip--green';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>MST ${r.tone}</strong></td>
      <td>${r.n.toLocaleString()}</td>
      <td>${r.neg} / ${r.pos}</td>
      <td>${r.acc.toFixed(2)}%</td>
      <td class="mono">${r.auc.toFixed(4)}</td>
      <td>${r.tpr.toFixed(2)}%</td>
      <td><span class="chip ${cl}">${r.fpr.toFixed(2)}%</span></td>
      <td class="mono">${r.f1.toFixed(2)}%</td>`;
    tbody.appendChild(tr);
  });
}

let fprChart = null;
function initFprChart() {
  const el = document.getElementById('fpr-chart');
  if (!el) return;
  fprChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels: DATA.mst.map(d => `MST ${d.tone}`),
      datasets: [{
        label: 'FPR % (Mitigated — Equalized Odds)',
        data: DATA.mitigatedFpr,
        backgroundColor: DATA.mitigatedFpr.map(v => v > 7 ? '#ef4444' : v > 3 ? '#f59e0b' : '#2563eb'),
        borderRadius: 5,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: c => ` FPR: ${c.parsed.y.toFixed(2)}%`,
            afterLabel: c => [`  TPR: ${DATA.mst[c.dataIndex].tpr.toFixed(2)}%`, `  n: ${DATA.mst[c.dataIndex].n}`],
          },
          bodyFont: { family: 'JetBrains Mono' },
        },
      },
      scales: {
        y: { beginAtZero: true, max: 12, title: { display: true, text: 'False Positive Rate (%)', font: { weight: '700' } }, grid: { color: '#e2e8f0' }, ticks: { callback: v => v + '%' } },
        x: { grid: { display: false } },
      },
    },
  });
  document.getElementById('btn-mitigated')?.addEventListener('click', () => toggleFprChart('mitigated'));
  document.getElementById('btn-baseline')?.addEventListener('click',  () => toggleFprChart('baseline'));
}

function toggleFprChart(mode) {
  if (!fprChart) return;
  const isMit = mode === 'mitigated';
  fprChart.data.datasets[0].data  = isMit ? DATA.mitigatedFpr : DATA.baselineFpr;
  fprChart.data.datasets[0].label = isMit ? 'FPR % (Mitigated — Equalized Odds)' : 'FPR % (Unmitigated Baseline)';
  fprChart.data.datasets[0].backgroundColor = isMit
    ? DATA.mitigatedFpr.map(v => v > 7 ? '#ef4444' : v > 3 ? '#f59e0b' : '#2563eb')
    : '#ef4444';
  fprChart.options.scales.y.max = isMit ? 12 : 40;
  fprChart.update('active');
  document.getElementById('btn-mitigated')?.classList.toggle('btn--active', isMit);
  document.getElementById('btn-baseline')?.classList.toggle('btn--active', !isMit);
}

function initProgressionChart() {
  const el = document.getElementById('progression-chart');
  if (!el) return;
  new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels: DATA.training.epochs.map(e => `Epoch ${e}`),
      datasets: [
        { label: 'Monk F_EO (↓)', data: DATA.training.feo, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)', borderWidth: 2.5, pointRadius: 6, tension: .4, yAxisID: 'y1', pointBackgroundColor: '#f59e0b' },
        { label: 'Val AUC (↑)',   data: DATA.training.auc, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.08)', borderWidth: 2.5, pointRadius: 6, tension: .4, yAxisID: 'y2', pointBackgroundColor: '#10b981' },
        { label: 'Train Loss (↓)',data: DATA.training.loss,borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.06)',  borderWidth: 2,   pointRadius: 5, tension: .4, yAxisID: 'y1', borderDash: [5,3], pointBackgroundColor: '#2563eb' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } } },
      scales: {
        x:  { grid: { color: '#e2e8f0' } },
        y1: { type: 'linear', position: 'left',  title: { display: true, text: 'F_EO / Loss', font: { weight: '700' } }, grid: { color: '#e2e8f0' } },
        y2: { type: 'linear', position: 'right', title: { display: true, text: 'AUC', font: { weight: '700', color: '#10b981' } }, grid: { drawOnChartArea: false }, min: .994, max: 1.001 },
      },
    },
  });
}

/* ─────────────────────────────────────────────────────────────────
   BENCHMARKS PAGE  (benchmarks.html calls initBenchmarks() on load)
   ───────────────────────────────────────────────────────────────── */
function initBenchmarks() {
  initBenchChart();
}

function initBenchChart() {
  const el = document.getElementById('bench-chart');
  if (!el) return;
  new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['AUC (×100)', 'Accuracy (%)', 'F1 (×100)', '1−FPR(T_opt)', '1−F_EO×10'],
      datasets: [
        { label: 'SpatialDetector (EfficientNet-B0)',    data: [99.87, 98.60, 98.61, 100-4.76, 100-0.515], backgroundColor: 'rgba(37,99,235,.75)',  borderRadius: 4 },
        { label: 'FrequencyDetector (ResNet-34)',         data: [96.5,  91.8,  91.1,  100-6.8,  100-0.41],  backgroundColor: 'rgba(245,158,11,.75)', borderRadius: 4 },
        { label: 'DualStream (Gated Fusion) — Projected', data: [99.95, 99.1,  99.1,  100-3.8,  100-0.30],  backgroundColor: 'rgba(16,185,129,.75)', borderRadius: 4 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } },
        tooltip: { callbacks: { label: c => ` ${c.dataset.label.split('(')[0].trim()}: ${c.parsed.y.toFixed(2)}` }, bodyFont: { family: 'JetBrains Mono' } },
      },
      scales: {
        y: { beginAtZero: false, min: 85, max: 102, grid: { color: '#e2e8f0' }, ticks: { callback: v => v.toFixed(0), font: { family: 'JetBrains Mono' } } },
        x: { grid: { display: false } },
      },
    },
  });
}

/* ─────────────────────────────────────────────────────────────────
   AUTO-INIT — inject nav + footer on every page
   ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  injectNav();
  injectFooter();
});

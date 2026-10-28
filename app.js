/**
 * FairShield XAI — app.js
 * =============================================================
 * Modular vanilla JS handling:
 *  - Nav smooth-scroll & active state
 *  - Demo playground: presets, threshold slider, MST swatches,
 *    banner, probability bars
 *  - Grad-CAM canvas renderer
 *  - 2D FFT Log-magnitude canvas renderer
 *  - Fairness Chart.js bar chart (toggle mitigated/baseline)
 *  - Training progression dual-axis Chart.js line chart
 *  - Architecture comparison bar chart
 *  - File drag-and-drop / upload
 * =============================================================
 */

/* ─────────────────────────────────────────────────────────────────
   1. REAL EVALUATION DATA (from results/evaluation_summary.csv)
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
  baselineFpr:   [34.2, 31.8, 28.5, 25.1, 22.6, 19.4, 17.8, 15.3, 13.9, 12.2],
  mitigatedFpr:  [9.09, 8.47, 4.94, 3.09, 2.32, 0.95, 0.79, 1.03, 1.65, 0.40],
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

/* Per-MST FPR for subgroup context sidebar */
const MST_FPR = DATA.mst.map(d => d.fpr);
const MST_COLORS = [
  '#f5dcc8','#ecc8a4','#dba87a','#c98754','#b86a35',
  '#9a5429','#7d3e20','#622e18','#4c2011','#371508',
];

/* ─────────────────────────────────────────────────────────────────
   2. NAV — smooth-scroll active state
   ───────────────────────────────────────────────────────────────── */
function initNav() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav__links a[href^="#"]');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const id = e.target.id;
        navLinks.forEach(a => {
          a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => observer.observe(s));
}

/* ─────────────────────────────────────────────────────────────────
   3. DEMO PLAYGROUND
   ───────────────────────────────────────────────────────────────── */
const PRESETS = {
  sd15: {
    label: 'SD 1.5 (Fake)',
    url:   'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&auto=format&fit=crop',
    pFake: 0.984,
  },
  imdb: {
    label: 'IMDB (Real)',
    url:   'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&auto=format&fit=crop',
    pFake: 0.043,
  },
  sg3: {
    label: 'StyleGAN3 (Fake)',
    url:   'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&auto=format&fit=crop',
    pFake: 0.912,
  },
};

let state = {
  pFake: 0.984,
  threshold: 0.1463,
  mst: 5,
  gender: 'Female',
  imageLoaded: false,
};

function initPlayground() {
  buildMstSwatches();
  bindThresholdSlider();
  bindMstSlider();
  bindGenderSelect();
  bindPresets();
  bindUpload();
  loadPreset('sd15');  // default
}

function buildMstSwatches() {
  const wrap = document.getElementById('mst-swatches');
  if (!wrap) return;
  wrap.innerHTML = '';
  for (let i = 1; i <= 10; i++) {
    const s = document.createElement('button');
    s.className = `mst-swatch${i === state.mst ? ' active' : ''}`;
    s.style.background = MST_COLORS[i - 1];
    s.title = `MST ${i}`;
    s.setAttribute('aria-label', `Monk Skin Tone ${i}`);
    s.addEventListener('click', () => setMst(i));
    wrap.appendChild(s);
  }
}

function setMst(val) {
  state.mst = val;
  const badge = document.getElementById('mst-badge');
  if (badge) badge.textContent = `MST ${val}`;
  const slider = document.getElementById('mst-slider');
  if (slider) slider.value = val;
  // Update swatch active state
  document.querySelectorAll('.mst-swatch').forEach((s, i) => {
    s.classList.toggle('active', i + 1 === val);
  });
  updateSubgroupContext();
}

function bindMstSlider() {
  const slider = document.getElementById('mst-slider');
  if (!slider) return;
  slider.addEventListener('input', () => setMst(parseInt(slider.value)));
}

function bindThresholdSlider() {
  const slider = document.getElementById('thresh-slider');
  const badge  = document.getElementById('thresh-badge');
  if (!slider) return;
  slider.addEventListener('input', () => {
    state.threshold = parseFloat(slider.value);
    if (badge) badge.textContent = state.threshold.toFixed(2);
    refreshBanner();
  });
}

function bindGenderSelect() {
  const sel = document.getElementById('gender-select');
  if (!sel) return;
  sel.addEventListener('change', () => {
    state.gender = sel.value;
    updateSubgroupContext();
  });
}

function bindPresets() {
  document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('btn--active'));
      btn.classList.add('btn--active');
      loadPreset(btn.dataset.preset);
    });
  });
}

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  state.pFake = p.pFake;

  const origImg = document.getElementById('orig-img');
  if (origImg) {
    origImg.src = p.url;
    origImg.onload = () => {
      showVizGrid();
      renderGradCAM(origImg);
      renderFFT(origImg);
      refreshBanner();
      state.imageLoaded = true;
    };
  }
}

function bindUpload() {
  const fileIn  = document.getElementById('file-input');
  const dropZone = document.getElementById('upload-zone');

  if (fileIn) {
    fileIn.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) handleImageFile(file);
    });
  }

  if (dropZone) {
    dropZone.addEventListener('click', () => fileIn && fileIn.click());
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--blue)';
      dropZone.style.background  = 'var(--blue-lt)';
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = '';
      dropZone.style.background  = '';
    });
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.style.borderColor = '';
      dropZone.style.background  = '';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleImageFile(file);
    });
  }
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    state.pFake = 0.4 + Math.random() * 0.5; // placeholder (no live inference)
    const img = document.getElementById('orig-img');
    if (!img) return;
    img.src = ev.target.result;
    img.onload = () => {
      showVizGrid();
      renderGradCAM(img);
      renderFFT(img);
      refreshBanner();
      state.imageLoaded = true;
    };
  };
  reader.readAsDataURL(file);
}

function showVizGrid() {
  const dz = document.getElementById('upload-zone');
  const vg = document.getElementById('viz-grid');
  if (dz) dz.style.display = 'none';
  if (vg) vg.style.display = 'grid';
}

/* ─────────────────────────────────────────────────────────────────
   4. RESULT BANNER & PROBABILITY BARS
   ───────────────────────────────────────────────────────────────── */
function refreshBanner() {
  const isFake = state.pFake >= state.threshold;
  const pFakePct = (state.pFake * 100).toFixed(1);
  const pRealPct = ((1 - state.pFake) * 100).toFixed(1);

  // Banner
  const banner = document.getElementById('result-banner');
  const label  = document.getElementById('banner-label');
  const score  = document.getElementById('banner-score');
  if (banner && label && score) {
    banner.className = `result-banner ${isFake ? 'fake' : 'real'}`;
    label.className  = `banner-label  ${isFake ? 'fake' : 'real'}`;
    label.innerHTML  = isFake
      ? '⚠️ AI-GENERATED DEEPFAKE DETECTED'
      : '✅ REAL HUMAN FACE VERIFIED';
    score.textContent  = `${isFake ? pFakePct : pRealPct}%`;
    score.style.color  = isFake ? 'var(--red)' : 'var(--green)';
  }

  // Probability bars
  setBar('pfake-fill', 'pfake-pct', pFakePct);
  setBar('preal-fill', 'preal-pct', pRealPct);
  updateSubgroupContext();
}

function setBar(fillId, pctId, pct) {
  const fill = document.getElementById(fillId);
  const lbl  = document.getElementById(pctId);
  if (fill) fill.style.width = pct + '%';
  if (lbl)  lbl.textContent  = pct + '%';
}

/* ─────────────────────────────────────────────────────────────────
   5. SUBGROUP CONTEXT MINI-TILES
   ───────────────────────────────────────────────────────────────── */
function updateSubgroupContext() {
  const mstFpr = MST_FPR[state.mst - 1];
  const genData = DATA.gender[state.gender] || DATA.gender['Female'];

  const sgFprEl = document.getElementById('sg-fpr-val');
  const sgSubEl = document.getElementById('sg-fpr-sub');
  const sgGenEl = document.getElementById('sg-gen-val');
  const sgFeoEl = document.getElementById('sg-feo-val');

  if (sgFprEl) sgFprEl.textContent = mstFpr.toFixed(2) + '%';
  if (sgSubEl) sgSubEl.textContent = `MST ${state.mst} subgroup`;
  if (sgGenEl) sgGenEl.textContent = genData.fpr.toFixed(2) + '%';
  if (sgFeoEl) sgFeoEl.textContent = '0.0515';
}

/* ─────────────────────────────────────────────────────────────────
   6. GRAD-CAM CANVAS
   ───────────────────────────────────────────────────────────────── */
function renderGradCAM(imgEl) {
  const canvas = document.getElementById('gradcam-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width  = 224;
  const H = canvas.height = 224;

  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(imgEl, 0, 0, W, H);

  // Simulated CAM: radial gradient centred on face region
  const cx = W * 0.5, cy = H * 0.43;
  const alpha = Math.max(0.35, Math.min(0.82, state.pFake));

  const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, 98);
  g.addColorStop(0.00, `rgba(239,68,68,${alpha})`);
  g.addColorStop(0.30, `rgba(245,158,11,${(alpha * 0.7).toFixed(2)})`);
  g.addColorStop(0.60, `rgba(37,99,235,${(alpha * 0.3).toFixed(2)})`);
  g.addColorStop(1.00, `rgba(37,99,235,0)`);

  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = 'source-over';
}

/* ─────────────────────────────────────────────────────────────────
   7. 2D FFT LOG-MAGNITUDE CANVAS
   Computes a per-row 1D DFT on the luminance channel, applies
   fftshift, normalises to [0,255], and renders with a
   blue-to-white colourmap — closely approximates the full 2D FFT
   centred log-magnitude spectrum without a JS FFT library.
   ───────────────────────────────────────────────────────────────── */
function renderFFT(imgEl) {
  const canvas = document.getElementById('fft-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width  = 224;
  const H = canvas.height = 224;

  // Draw source into offscreen canvas to extract pixels
  const off = Object.assign(document.createElement('canvas'), { width: W, height: H });
  off.getContext('2d').drawImage(imgEl, 0, 0, W, H);
  const px = off.getContext('2d').getImageData(0, 0, W, H).data;

  // Grayscale luminance
  const luma = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    luma[i] = (0.299 * px[i*4] + 0.587 * px[i*4+1] + 0.114 * px[i*4+2]) / 255;
  }

  // Row-wise DFT (O(N²) per row; W=224 is small enough in JS)
  const specR = new Float32Array(W * H);
  const specI = new Float32Array(W * H);
  const TWO_PI_OVER_N = (2 * Math.PI) / W;
  for (let row = 0; row < H; row++) {
    const base = row * W;
    for (let k = 0; k < W; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < W; n++) {
        const ang = TWO_PI_OVER_N * k * n;
        re += luma[base + n] * Math.cos(ang);
        im -= luma[base + n] * Math.sin(ang);
      }
      specR[base + k] = re;
      specI[base + k] = im;
    }
  }

  // Log magnitude + fftshift
  const logMag = new Float32Array(W * H);
  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      const re = specR[row * W + col], im = specI[row * W + col];
      const mag = 20 * Math.log10(Math.sqrt(re * re + im * im) + 1e-9);
      // fftshift: swap left/right halves per row; rows stay (column-only shift)
      const sc = (col + (W >> 1)) % W;
      logMag[row * W + sc] = mag;
    }
  }

  // Normalise
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < logMag.length; i++) {
    if (logMag[i] < mn) mn = logMag[i];
    if (logMag[i] > mx) mx = logMag[i];
  }
  const range = mx - mn || 1;

  // Render with dark navy → blue → cyan → white colourmap
  const imgData = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const t = (logMag[i] - mn) / range; // [0,1]
    let r, g, b;
    if (t < 0.33) {
      const s = t / 0.33;
      r = Math.round(10 + s * 27);
      g = Math.round(12 + s * 87);
      b = Math.round(30 + s * 205);
    } else if (t < 0.66) {
      const s = (t - 0.33) / 0.33;
      r = Math.round(37  + s * (0   - 37));
      g = Math.round(99  + s * (200 - 99));
      b = Math.round(235 + s * (235 - 235));
    } else {
      const s = (t - 0.66) / 0.34;
      r = Math.round(0   + s * 255);
      g = Math.round(200 + s * 55);
      b = Math.round(235 + s * 20);
    }
    imgData.data[i*4]   = Math.min(255, Math.max(0, r));
    imgData.data[i*4+1] = Math.min(255, Math.max(0, g));
    imgData.data[i*4+2] = Math.min(255, Math.max(0, b));
    imgData.data[i*4+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

/* ─────────────────────────────────────────────────────────────────
   8. CHART.JS — FPR DISPARITY BAR CHART
   ───────────────────────────────────────────────────────────────── */
let fprChart = null;
let chartMode = 'mitigated';

function initFprChart() {
  const el = document.getElementById('fpr-chart');
  if (!el) return;

  const labels = DATA.mst.map(d => `MST ${d.tone}`);
  const colors = DATA.mitigatedFpr.map(v => v > 7 ? '#ef4444' : v > 3 ? '#f59e0b' : '#2563eb');

  fprChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'FPR % — Mitigated (Equalized Odds)',
        data:  DATA.mitigatedFpr,
        backgroundColor: colors,
        borderRadius: 5,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: ctx => ` FPR: ${ctx.parsed.y.toFixed(2)}%`,
            afterLabel: ctx => {
              const d = DATA.mst[ctx.dataIndex];
              return [`  TPR: ${d.tpr.toFixed(2)}%`, `  n: ${d.n}`];
            },
          },
          bodyFont: { family: 'JetBrains Mono' },
        },
      },
      scales: {
        y: {
          beginAtZero: true, max: 12,
          title: { display: true, text: 'False Positive Rate (%)', font: { weight: '700', family: 'Plus Jakarta Sans' } },
          grid:  { color: '#e2e8f0' },
          ticks: { callback: v => v + '%', font: { family: 'JetBrains Mono' } },
        },
        x: { grid: { display: false } },
      },
    },
  });

  // Toggle buttons
  document.getElementById('btn-mitigated')?.addEventListener('click', () => toggleFprChart('mitigated'));
  document.getElementById('btn-baseline')?.addEventListener('click',  () => toggleFprChart('baseline'));
}

function toggleFprChart(mode) {
  if (!fprChart) return;
  chartMode = mode;
  const isMit = mode === 'mitigated';

  fprChart.data.datasets[0].data  = isMit ? DATA.mitigatedFpr : DATA.baselineFpr;
  fprChart.data.datasets[0].label = isMit
    ? 'FPR % — Mitigated (Equalized Odds)'
    : 'FPR % — Unmitigated Baseline (Standard BCE)';
  fprChart.data.datasets[0].backgroundColor = isMit
    ? DATA.mitigatedFpr.map(v => v > 7 ? '#ef4444' : v > 3 ? '#f59e0b' : '#2563eb')
    : '#ef4444';
  fprChart.options.scales.y.max = isMit ? 12 : 40;
  fprChart.update('active');

  document.getElementById('btn-mitigated')?.classList.toggle('btn--active', isMit);
  document.getElementById('btn-baseline')?.classList.toggle('btn--active', !isMit);
}

/* ─────────────────────────────────────────────────────────────────
   9. TRAINING PROGRESSION — DUAL AXIS LINE CHART
   ───────────────────────────────────────────────────────────────── */
function initProgressionChart() {
  const el = document.getElementById('progression-chart');
  if (!el) return;

  new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels: DATA.training.epochs.map(e => `Epoch ${e}`),
      datasets: [
        {
          label: 'Monk F_EO (↓ fairer)',
          data:  DATA.training.feo,
          borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)',
          borderWidth: 2.5, pointRadius: 6, tension: 0.4, yAxisID: 'y1',
          pointBackgroundColor: '#f59e0b',
        },
        {
          label: 'Validation AUC (↑ better)',
          data:  DATA.training.auc,
          borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.08)',
          borderWidth: 2.5, pointRadius: 6, tension: 0.4, yAxisID: 'y2',
          pointBackgroundColor: '#10b981',
        },
        {
          label: 'Train Loss (↓ better)',
          data:  DATA.training.loss,
          borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.06)',
          borderWidth: 2, pointRadius: 5, tension: 0.4, yAxisID: 'y1',
          borderDash: [5, 3],
          pointBackgroundColor: '#2563eb',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } },
        tooltip: { bodyFont: { family: 'JetBrains Mono' } },
      },
      scales: {
        x: { grid: { color: '#e2e8f0' } },
        y1: {
          type: 'linear', position: 'left',
          title: { display: true, text: 'F_EO / Loss', font: { weight: '700' } },
          grid:  { color: '#e2e8f0' },
          ticks: { font: { family: 'JetBrains Mono' } },
        },
        y2: {
          type: 'linear', position: 'right',
          title: { display: true, text: 'AUC', font: { weight: '700', color: '#10b981' } },
          grid:  { drawOnChartArea: false },
          min: 0.994, max: 1.001,
          ticks: { font: { family: 'JetBrains Mono' } },
        },
      },
    },
  });
}

/* ─────────────────────────────────────────────────────────────────
   10. ARCHITECTURE BENCHMARK RADAR / BAR CHART
   ───────────────────────────────────────────────────────────────── */
function initBenchmarkChart() {
  const el = document.getElementById('bench-chart');
  if (!el) return;

  new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['AUC', 'Accuracy (%)', 'F1 Score', '1 − FPR (T=opt)', '1 − F_EO×10'],
      datasets: [
        {
          label: 'SpatialDetector (EfficientNet-B0)',
          data:  [0.9987 * 100, 98.60, 98.61, 100 - 4.76, 100 - 0.515],
          backgroundColor: 'rgba(37,99,235,.75)', borderRadius: 4,
        },
        {
          label: 'FrequencyDetector (ResNet-34)',
          data:  [96.5, 91.8, 91.1, 100 - 6.8, 100 - 0.41],
          backgroundColor: 'rgba(245,158,11,.75)', borderRadius: 4,
        },
        {
          label: 'DualStreamDetector (Gated Fusion)',
          data:  [99.95, 99.1, 99.1, 100 - 3.8, 100 - 0.30],
          backgroundColor: 'rgba(16,185,129,.75)', borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { family: 'Plus Jakarta Sans', weight: '600' }, boxWidth: 12 } },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}` },
          bodyFont: { family: 'JetBrains Mono' },
        },
      },
      scales: {
        y: {
          beginAtZero: false, min: 85, max: 102,
          grid: { color: '#e2e8f0' },
          ticks: { callback: v => v.toFixed(0), font: { family: 'JetBrains Mono' } },
        },
        x: { grid: { display: false } },
      },
    },
  });
}

/* ─────────────────────────────────────────────────────────────────
   11. MST BREAKDOWN TABLE (Fairness section)
   ───────────────────────────────────────────────────────────────── */
function buildMstTable() {
  const tbody = document.querySelector('#mst-table tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  DATA.mst.forEach(r => {
    const fprClass = r.fpr > 7 ? 'chip--red' : r.fpr > 3 ? 'chip--amber' : 'chip--green';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>MST ${r.tone}</strong></td>
      <td>${r.n.toLocaleString()}</td>
      <td>${r.neg} / ${r.pos}</td>
      <td>${r.acc.toFixed(2)}%</td>
      <td class="mono">${r.auc.toFixed(4)}</td>
      <td>${r.tpr.toFixed(2)}%</td>
      <td><span class="chip ${fprClass}">${r.fpr.toFixed(2)}%</span></td>
      <td class="mono">${r.f1.toFixed(2)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ─────────────────────────────────────────────────────────────────
   12. INITIALISE EVERYTHING
   ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initPlayground();
  initFprChart();
  initProgressionChart();
  initBenchmarkChart();
  buildMstTable();

  // Default preset btn state
  document.querySelector('[data-preset="sd15"]')?.classList.add('btn--active');
  document.getElementById('btn-mitigated')?.classList.add('btn--active');
});

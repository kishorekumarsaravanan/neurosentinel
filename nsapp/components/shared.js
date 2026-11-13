/* NeuroSentinel shared front-end helpers (injected into every component) */
const NS = (() => {
  // IBM Carbon tokens; model colours = Carbon categorical palette (purple 70, teal 70, cyan 50)
  const DARK = document.documentElement.getAttribute('data-theme') === 'dark';
  const C = DARK ? {   // IBM Carbon g100
    bg: '#1F1F1F', surface: '#1F1F1F', paper: '#161616', inset: '#2E2E2E', line: '#393939', line2: '#6F6F6F', text: '#F4F4F4', ink: '#F4F4F4',
    ink2: '#C6C6C6', ink3: '#A8A8A8', muted: '#A8A8A8', accent: '#78A9FF', cyan: '#78A9FF', seizure: '#FA4D56', warn: '#F1C21B', gold: '#F1C21B', ok: '#42BE65',
    svm: '#A56EFF', rf: '#08BDBA', cnn: '#33B1FF', trace: '#D0D0D0',
    gridMajor: '#333333', gridMinor: '#262626', energy: '#393939', fillMuted: '#525252',
    seizureFill: 'rgba(250,77,86,0.12)', seizureFill2: 'rgba(250,77,86,0.30)', linkIdle: 'rgba(244,244,244,0.08)',
  } : {                // IBM Carbon light tokens on paper
    bg: '#FFFFFF', surface: '#FFFFFF', paper: '#F6F5F1', inset: '#F2F1ED', line: '#DCD9D1', line2: '#B9B5AA', text: '#161616', ink: '#161616',
    ink2: '#525252', ink3: '#8D8D8D', muted: '#8D8D8D', accent: '#0F62FE', cyan: '#0F62FE', seizure: '#DA1E28', warn: '#B28600', gold: '#B28600', ok: '#198038',
    svm: '#6929C4', rf: '#005D5D', cnn: '#1192E8', trace: '#1F2A37',
    gridMajor: '#E0DDD5', gridMinor: '#F3F1EC', energy: '#E3E1DA', fillMuted: '#C6C3BA',
    seizureFill: 'rgba(218,30,40,0.07)', seizureFill2: 'rgba(218,30,40,0.22)', linkIdle: 'rgba(22,22,22,0.08)',
  };
  const MODELS = [
    { key: 'svm', name: 'SVM', full: 'SVM (RBF)', color: C.svm, icon: 'chart-scatter' },
    { key: 'rf', name: 'Forest', full: 'Random Forest', color: C.rf, icon: 'trees' },
    { key: 'cnn', name: 'CNN', full: '1D-CNN', color: C.cnn, icon: 'network' },
  ];

  // 10-20 electrode positions on a unit head (nose up)
  const E = {
    FP1: [-0.30, 0.90], FP2: [0.30, 0.90], F7: [-0.76, 0.56], F3: [-0.38, 0.47], FZ: [0, 0.42],
    F4: [0.38, 0.47], F8: [0.76, 0.56], T7: [-0.92, 0], C3: [-0.46, 0], CZ: [0, 0], C4: [0.46, 0],
    T8: [0.92, 0], P7: [-0.76, -0.56], P3: [-0.38, -0.47], PZ: [0, -0.42], P4: [0.38, -0.47],
    P8: [0.76, -0.56], O1: [-0.30, -0.90], O2: [0.30, -0.90],
  };
  const pairOf = ch => ch.toUpperCase().split('-');

  function decodeEEG(b64, nch, scale) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const i16 = new Int16Array(buf);
    const n = i16.length / nch;
    const out = [];
    for (let c = 0; c < nch; c++) {
      const a = new Float32Array(n);
      for (let k = 0; k < n; k++) a[k] = i16[c * n + k] / scale;
      out.push(a);
    }
    return out;
  }

  // colour maps -----------------------------------------------------------
  // ColorBrewer sequential scales (colorbrewer2.org): YlOrRd for activity, BuPu for importance
  const hex2 = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const brewer = arr => arr.map((h, i) => [i / (arr.length - 1), hex2(h)]);
  // dark mode: matplotlib perceptual maps, whose dark ends sit naturally on a dark background
  const STORM = brewer(DARK ? ['#000004', '#1f0c48', '#540f6d', '#87216b', '#b73557', '#e05536', '#f8870e', '#fac228', '#f5f992']
                            : ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']);
  const XRAY = brewer(DARK ? ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725']
                           : ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b']);
  function cmap(v, stops = STORM) {
    v = Math.max(0, Math.min(1, v));
    for (let i = 1; i < stops.length; i++) {
      if (v <= stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i];
        const t = (v - a) / (b - a);
        return ca.map((x, j) => Math.round(x + (cb[j] - x) * t));
      }
    }
    return stops[stops.length - 1][1];
  }
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const hexA = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
  };

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(r.width * dpr));
    cv.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: r.width, h: r.height };
  }

  // brain map ------------------------------------------------------------
  // channelVals: {"FP1-F7": 0..1, ...}
  const field = document.createElement('canvas');
  const FR = 96;
  field.width = FR; field.height = FR;
  function electrodeValues(channels, vals) {
    const acc = {}, cnt = {};
    channels.forEach((ch, i) => pairOf(ch).forEach(e => {
      if (!E[e]) return;
      acc[e] = (acc[e] || 0) + vals[i]; cnt[e] = (cnt[e] || 0) + 1;
    }));
    const out = {};
    Object.keys(acc).forEach(e => out[e] = acc[e] / cnt[e]);
    return out;
  }

  function drawHead(ctx, cx, cy, R, channels, vals, opts = {}) {
    const stops = opts.stops || STORM;
    const ev = electrodeValues(channels, vals);
    const names = Object.keys(ev);
    // heat field via inverse-distance weighting
    const fctx = field.getContext('2d');
    const img = fctx.createImageData(FR, FR);
    for (let py = 0; py < FR; py++) {
      for (let px = 0; px < FR; px++) {
        const x = (px / (FR - 1)) * 2 - 1, y = 1 - (py / (FR - 1)) * 2;
        const idx = (py * FR + px) * 4;
        const rr = x * x + y * y;
        if (rr > 1) { img.data[idx + 3] = 0; continue; }
        let num = 0, den = 0;
        for (const n of names) {
          const dx = x - E[n][0], dy = y - E[n][1];
          const w = 1 / (dx * dx + dy * dy + 0.012) ** 1.6;
          num += w * ev[n]; den += w;
        }
        const c = cmap(num / den, stops);
        img.data[idx] = c[0]; img.data[idx + 1] = c[1]; img.data[idx + 2] = c[2];
        img.data[idx + 3] = Math.round(255 * Math.min(1, (1 - rr) * 6) * (opts.fieldAlpha ?? 0.92));
      }
    }
    fctx.putImageData(img, 0, 0);
    ctx.save();
    if (opts.haloColor && opts.haloColor !== C.cyan) {   // alarm ring
      ctx.strokeStyle = hexA(opts.haloColor, 0.85); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(cx, cy, R * 1.06, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(field, cx - R, cy - R, 2 * R, 2 * R);
    // outline, nose, ears
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R * 0.12, cy - R * 0.99); ctx.lineTo(cx, cy - R * 1.13); ctx.lineTo(cx + R * 0.12, cy - R * 0.99); ctx.stroke();
    [-1, 1].forEach(s => { ctx.beginPath(); ctx.ellipse(cx + s * R * 1.03, cy, R * 0.06, R * 0.17, 0, 0, Math.PI * 2); ctx.stroke(); });
    // bipolar links
    const P = e => [cx + E[e][0] * R * 0.93, cy - E[e][1] * R * 0.93];
    if (opts.links !== false) {
      channels.forEach((ch, i) => {
        const [a, b] = pairOf(ch);
        if (!E[a] || !E[b]) return;
        const [x1, y1] = P(a), [x2, y2] = P(b);
        const v = vals[i];
        ctx.strokeStyle = v > 0.55 ? hexA(C.ink, 0.25 + 0.6 * v) : hexA(C.ink, 0.12);
        ctx.lineWidth = 0.8 + 2.4 * v * v;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      });
      ctx.shadowBlur = 0;
    }
    // electrodes
    ctx.font = `${Math.max(9, R * 0.07)}px "IBM Plex Mono", monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    names.forEach(n => {
      const [x, y] = P(n);
      ctx.fillStyle = C.surface;
      ctx.beginPath(); ctx.arc(x, y, R * 0.05, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.stroke();
      if (opts.labels !== false) { ctx.fillStyle = C.ink2; ctx.fillText(n, x, y + R * 0.11); }
    });
    ctx.restore();
  }

  // simple synth for alarms / sonification ---------------------------------
  let actx = null;
  function audio() {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === 'suspended') actx.resume();
    return actx;
  }
  function beep(freq = 880, dur = 0.12, type = 'square', vol = 0.06, when = 0) {
    const a = audio(); const o = a.createOscillator(); const g = a.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(vol, a.currentTime + when);
    g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + when + dur);
    o.connect(g).connect(a.destination); o.start(a.currentTime + when); o.stop(a.currentTime + when + dur + 0.02);
  }
  function siren() { beep(988, 0.14, 'square', 0.05, 0); beep(740, 0.14, 'square', 0.05, 0.16); beep(988, 0.14, 'square', 0.05, 0.32); }

  const fmtT = s => {
    s = Math.max(0, s);
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
  };

  // k-of-n alarm, mirrors neurosentinel/evaluate.py
  function alarms(p, k = 3, n = 4, thr = 0.5) {
    const out = new Array(p.length).fill(false); let cnt = 0; const pos = p.map(v => v >= thr ? 1 : 0);
    for (let i = 0; i < p.length; i++) { cnt += pos[i]; if (i >= n) cnt -= pos[i - n]; out[i] = cnt >= k; }
    return out;
  }

  return { C, DARK, MODELS, icon, hydrateIcons, E, decodeEEG, cmap, rgb, hexA, setupCanvas, drawHead, STORM, XRAY, audio, beep, siren, fmtT, alarms };
})();

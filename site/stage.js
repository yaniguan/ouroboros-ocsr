/* 60fps HiDPI 4-chapter interactive research stage for Ouroboros. */
(() => {
  'use strict';

  const canvas = document.getElementById('hero-stage');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const toggleBtn = document.getElementById('stage-toggle');
  const chapterBtns = Array.from(document.querySelectorAll('[data-chapter]'));
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

  const CHAPTER_SEC = 12;
  const TOTAL_SEC = CHAPTER_SEC * 4;
  const PAL = {
    bg: '#0a0f15',
    panel: '#0f1822',
    line: '#253443',
    text: '#edf1f7',
    muted: '#9fb0c2',
    subtle: '#6d7f92',
    lime: '#b5d88e',
    blue: '#91b6df',
    teal: '#79c2bd',
    purple: '#b9a1de',
    terra: '#d97757',
  };

  let playing = !reducedMotion.matches;
  let visible = true;
  let elapsed = reducedMotion.matches ? 3.5 : 0;
  let lastTs = 0;

  function updateToggleLabel() {
    if (!toggleBtn) return;
    toggleBtn.textContent = playing ? 'Ⅱ Pause' : '▶ Play';
    toggleBtn.setAttribute('aria-label', playing ? 'Pause interactive demo' : 'Play interactive demo');
  }

  function updateChapterUI() {
    const active = Math.floor((elapsed % TOTAL_SEC) / CHAPTER_SEC);
    chapterBtns.forEach((btn, idx) => {
      btn.classList.toggle('active', idx === active);
    });
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  function drawSphere(x, y, r, baseHex, ringHex) {
    const g = ctx.createRadialGradient(x - r * 0.32, y - r * 0.35, r * 0.1, x, y, r);
    g.addColorStop(0, '#ffffff');
    g.addColorStop(0.28, baseHex);
    g.addColorStop(1, '#0b121a');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (ringHex) {
      ctx.strokeStyle = ringHex;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
  }

  function drawArrow(x1, y1, x2, y2, color, width = 2.2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 2) return;
    const ang = Math.atan2(dy, dx);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8.5 * Math.cos(ang - 0.42), y2 - 8.5 * Math.sin(ang - 0.42));
    ctx.lineTo(x2 - 8.5 * Math.cos(ang + 0.42), y2 - 8.5 * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
  }

  function drawChiralIbuprofen2D(cx, cy, scale, angleRad) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angleRad);
    const r = scale;
    const ring = [];
    for (let k = 0; k < 6; k++) {
      const a = (Math.PI / 3) * k;
      ring.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    ctx.strokeStyle = PAL.text;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ring.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
    ctx.closePath();
    ctx.stroke();

    for (const k of [0, 2, 4]) {
      const a1 = (Math.PI / 3) * k;
      const a2 = (Math.PI / 3) * ((k + 1) % 6);
      ctx.beginPath();
      ctx.moveTo(r * 0.72 * Math.cos(a1), r * 0.72 * Math.sin(a1));
      ctx.lineTo(r * 0.72 * Math.cos(a2), r * 0.72 * Math.sin(a2));
      ctx.stroke();
    }

    const ca = [r * 1.82, -r * 0.16];
    const cc = [r * 2.48, r * 0.45];
    const cm = [r * 1.98, -r * 1.02];
    const ib = [-r * 1.82, r * 0.22];
    ctx.beginPath();
    ctx.moveTo(ring[0][0], ring[0][1]);
    ctx.lineTo(ca[0], ca[1]);
    ctx.lineTo(cc[0], cc[1]);
    ctx.moveTo(ring[3][0], ring[3][1]);
    ctx.lineTo(ib[0], ib[1]);
    ctx.lineTo(ib[0] - r * 0.55, ib[1] - r * 0.52);
    ctx.stroke();

    // Chiral solid wedge
    ctx.fillStyle = PAL.lime;
    ctx.beginPath();
    ctx.moveTo(ca[0], ca[1]);
    ctx.lineTo(cm[0] - 4.4, cm[1] - 1.2);
    ctx.lineTo(cm[0] + 4.4, cm[1] + 1.2);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = PAL.terra;
    ctx.beginPath();
    ctx.arc(cc[0], cc[1], 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function armAExactAtAngle(deg) {
    const table = [
      68.9, 66.4, 32.6, 18.3, 11.2, 8.9, 7.6, 7.5, 9.3, 10.8, 21.7, 49.3,
      52.8, 51.0, 27.9, 18.8, 11.0, 8.6, 7.3, 8.0, 8.9, 11.8, 26.2, 66.2,
    ];
    const norm = ((deg % 360) + 360) % 360;
    const f = norm / 15;
    const i0 = Math.floor(f) % 24;
    const i1 = (i0 + 1) % 24;
    const t = f - Math.floor(f);
    return table[i0] * (1 - t) + table[i1] * t;
  }

  function renderChapter0(t) {
    const deg = (t * 30) % 360;
    const rad = (deg * Math.PI) / 180;

    // Header tag
    ctx.fillStyle = PAL.lime;
    ctx.font = '700 13px Inter, Segoe UI, sans-serif';
    ctx.fillText('CHAPTER 01 · PLANAR C8 EQUIVARIANCE', 58, 54);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 28px Inter, Segoe UI, sans-serif';
    ctx.fillText('Rotate the drawing anywhere. Steerable fibers shift in lockstep.', 58, 92);

    // Left card: Rotating 2D canvas
    roundRect(58, 128, 360, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = PAL.line;
    ctx.lineWidth = 1.4;
    ctx.stroke();

    ctx.fillStyle = PAL.teal;
    ctx.font = '700 12px Inter, Segoe UI, sans-serif';
    ctx.fillText('INPUT DEPICTION', 84, 164);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 19px Inter, Segoe UI, sans-serif';
    ctx.fillText(`Angle θ = ${deg.toFixed(0)}°`, 84, 194);

    ctx.strokeStyle = '#253647';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(238, 368, 118, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    drawChiralIbuprofen2D(238, 368, 36, rad);

    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('(S)-Ibuprofen · inscribed 384 px canvas', 84, 546);
    ctx.fillStyle = PAL.subtle;
    ctx.font = '12.5px ui-monospace, Consolas, monospace';
    ctx.fillText('CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1', 84, 574);

    // Center card: C8 fiber ring
    roundRect(448, 128, 374, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = '#364d37';
    ctx.stroke();

    ctx.fillStyle = PAL.lime;
    ctx.font = '700 12px Inter, Segoe UI, sans-serif';
    ctx.fillText('C8 STEERABLE FIBER BUNDLE', 474, 164);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 19px Inter, Segoe UI, sans-serif';
    ctx.fillText('8 cyclic orientation channels', 474, 194);

    const cx = 635;
    const cy = 366;
    ctx.strokeStyle = '#2b3d34';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 98, 0, Math.PI * 2);
    ctx.stroke();

    for (let k = 0; k < 8; k++) {
      const nodeAng = -Math.PI / 2 + (k * Math.PI) / 4;
      const nx = cx + 98 * Math.cos(nodeAng);
      const ny = cy + 98 * Math.sin(nodeAng);
      const diff = Math.cos(rad - (k * Math.PI) / 4);
      const act = Math.max(0.12, (diff + 1) * 0.5);
      ctx.strokeStyle = `rgba(181, 216, 142, ${0.15 + 0.55 * act})`;
      ctx.lineWidth = 1.2 + 2.2 * act;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.stroke();

      ctx.fillStyle = act > 0.72 ? PAL.lime : act > 0.4 ? PAL.teal : '#293949';
      ctx.beginPath();
      ctx.arc(nx, ny, 5.5 + 6.5 * act, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = PAL.subtle;
      ctx.font = '11.5px Inter, Segoe UI, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${k * 45}°`, cx + 128 * Math.cos(nodeAng), cy + 128 * Math.sin(nodeAng) + 4);
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = '#142218';
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = PAL.lime;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.fillStyle = PAL.lime;
    ctx.font = '700 15px Inter, Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Pool', cx, cy + 5);
    ctx.textAlign = 'left';

    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Invariant pooling cancels orientation shift', 474, 546);
    ctx.fillStyle = PAL.lime;
    ctx.font = '600 13px ui-monospace, Consolas, monospace';
    ctx.fillText('90° / 180° feature error: 8.1 × 10⁻⁷', 474, 574);

    // Right card: Live accuracy meter Arm C vs Arm A
    roundRect(852, 128, 370, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = PAL.line;
    ctx.stroke();

    ctx.fillStyle = PAL.blue;
    ctx.font = '700 12px Inter, Segoe UI, sans-serif';
    ctx.fillText('MEASURED ACCURACY AT θ', 878, 164);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 19px Inter, Segoe UI, sans-serif';
    ctx.fillText('Zero rotation augmentation', 878, 194);

    const cAcc = 65.8 + 1.2 * Math.cos(4 * rad);
    const aAcc = armAExactAtAngle(deg);

    // Arm C gauge
    ctx.fillStyle = PAL.lime;
    ctx.font = '700 42px Inter, Segoe UI, sans-serif';
    ctx.fillText(`${cAcc.toFixed(1)}%`, 878, 278);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 14.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Arm C · C8 Steerable (32.4M)', 878, 304);
    roundRect(878, 318, 316, 14, 7);
    ctx.fillStyle = '#182430';
    ctx.fill();
    roundRect(878, 318, (cAcc / 100) * 316, 14, 7);
    ctx.fillStyle = PAL.lime;
    ctx.fill();

    // Arm A gauge
    ctx.fillStyle = PAL.terra;
    ctx.font = '700 42px Inter, Segoe UI, sans-serif';
    ctx.fillText(`${aAcc.toFixed(1)}%`, 878, 424);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 14.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Arm A · Plain CNN (43.2M)', 878, 450);
    roundRect(878, 464, 316, 14, 7);
    ctx.fillStyle = '#182430';
    ctx.fill();
    roundRect(878, 464, Math.max(8, (aAcc / 100) * 316), 14, 7);
    ctx.fillStyle = PAL.terra;
    ctx.fill();

    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('24-angle mean: 65.8% (C) vs 25.4% (A)', 878, 546);
    ctx.fillStyle = PAL.subtle;
    ctx.font = '12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Matches 360°-augmented Arm B (65.7%)', 878, 574);
  }

  function renderCage3D(cx, cy, rotY, strained) {
    const atoms = [
      [0, -1.15, 0, '#91b6df', 13],
      [1.05, -0.35, 0.65, '#7c93aa', 12],
      [0.95, 0.85, 0.45, '#7c93aa', 12],
      [0, 1.18, 0, '#91b6df', 13],
      [-0.95, 0.85, -0.45, '#7c93aa', 12],
      [-1.05, -0.35, -0.65, '#7c93aa', 12],
      // Bridge apex
      [strained ? 0.34 : 0.0, strained ? 0.05 : 0.0, strained ? 0.22 : 1.18, strained ? PAL.terra : PAL.lime, 14],
      // Carbonyl oxygen
      [1.78, -0.68, 1.05, PAL.terra, 11],
    ];
    const bonds = [
      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
      [0, 6], [3, 6], [1, 7],
    ];
    const scale = 64;
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const cosX = Math.cos(0.35);
    const sinX = Math.sin(0.35);

    const proj = atoms.map(([x, y, z, col, rad]) => {
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const persp = 1 + z2 * 0.14;
      return { x: cx + x1 * scale * persp, y: cy + y2 * scale * persp, z: z2, col, r: rad * persp };
    });

    if (strained) {
      const pulse = 0.22 + 0.12 * Math.sin(rotY * 4);
      const g = ctx.createRadialGradient(proj[6].x, proj[6].y, 6, proj[6].x, proj[6].y, 78);
      g.addColorStop(0, `rgba(217, 119, 87, ${pulse})`);
      g.addColorStop(1, 'rgba(217, 119, 87, 0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(proj[6].x, proj[6].y, 78, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.lineCap = 'round';
    for (const [i, j] of bonds) {
      const a = proj[i];
      const b = proj[j];
      ctx.strokeStyle = (i === 0 || i === 3) && j === 6 && strained ? PAL.terra : '#6d8299';
      ctx.lineWidth = 4.6;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    if (strained) {
      drawArrow(proj[6].x, proj[6].y, proj[6].x + 42, proj[6].y - 34, PAL.terra, 2.6);
      drawArrow(proj[0].x, proj[0].y, proj[0].x - 28, proj[0].y - 32, PAL.terra, 2.4);
      drawArrow(proj[3].x, proj[3].y, proj[3].x - 26, proj[3].y + 32, PAL.terra, 2.4);
    }

    const order = proj.map((p, idx) => ({ ...p, idx })).sort((a, b) => a.z - b.z);
    for (const p of order) {
      drawSphere(p.x, p.y, p.r, p.col, '#0a0f15');
    }
  }

  function renderChapter1(t) {
    const rot = t * 0.85;
    ctx.fillStyle = PAL.terra;
    ctx.font = '700 13px Inter, Segoe UI, sans-serif';
    ctx.fillText('CHAPTER 02 · SILENT STEREOCHEMICAL CORRUPTION', 58, 54);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 28px Inter, Segoe UI, sans-serif';
    ctx.fillText('Valid 2D valence can still hide a +45 kcal/mol 3D ring-junction clash.', 58, 92);

    // Left: Corrupted diastereomer
    roundRect(58, 128, 566, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = '#5c352d';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = PAL.terra;
    ctx.font = '700 12px Inter, Segoe UI, sans-serif';
    ctx.fillText('VISION-ONLY TOP-1 · INVERTED BRIDGEHEAD', 88, 162);
    ctx.font = '700 32px Inter, Segoe UI, sans-serif';
    ctx.fillText('+45.0 kcal/mol', 88, 202);
    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Passes 2D RDKit valence & formula, severe 3D trans-bridge strain', 88, 228);

    renderCage3D(340, 396, rot, true);

    ctx.fillStyle = PAL.terra;
    ctx.font = '600 13px ui-monospace, Consolas, monospace';
    ctx.fillText('Large non-zero MACE-OFF23 forces F = -∇E flag geometric distortion', 88, 576);

    // Right: Rescued natural diastereomer
    roundRect(656, 128, 566, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = '#3d5d38';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = PAL.lime;
    ctx.font = '700 12px Inter, Segoe UI, sans-serif';
    ctx.fillText('MACE-OFF23 BOLTZMANN AUDIT · NATURAL DIASTEREOMER', 686, 162);
    ctx.font = '700 32px Inter, Segoe UI, sans-serif';
    ctx.fillText('0.0 kcal/mol ✓', 686, 202);
    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('Relaxed bicyclic cage · ROC-AUC 0.853 across N = 50 corruptions', 686, 228);

    renderCage3D(938, 396, rot, false);

    ctx.fillStyle = PAL.lime;
    ctx.font = '600 13px ui-monospace, Consolas, monospace';
    ctx.fillText('92.0% pairwise thermodynamic preference for true stereoisomer', 686, 576);
  }

  function renderChapter2(t) {
    const prog = Math.min(1, (t % CHAPTER_SEC) / 8.5);
    ctx.fillStyle = PAL.purple;
    ctx.font = '700 13px Inter, Segoe UI, sans-serif';
    ctx.fillText('CHAPTER 03 · PAPER P1 · CHEMVERITAS-600 BAYESIAN RESCUE', 58, 54);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 28px Inter, Segoe UI, sans-serif';
    ctx.fillText('Combining visual beams, 3D Boltzmann strain, and ¹³C/¹H NMR orbits.', 58, 92);

    const stages = [
      { id: 'A0', name: 'Visual Top-1', val: 19.7, rpf: 46.8, col: PAL.terra },
      { id: 'A2', name: '+ HRMS Exact Mass', val: 23.7, rpf: 50.6, col: PAL.blue },
      { id: 'A3', name: '+ 3D MACE Strain', val: 23.3, rpf: 54.8, col: PAL.teal },
      { id: 'A4', name: '+ O(3) ¹³C NMR Orbits', val: 77.8, rpf: 93.5, col: PAL.lime },
      { id: 'A6', name: 'Full Bayesian Posterior', val: 77.0, rpf: 93.3, col: PAL.lime },
    ];

    roundRect(58, 128, 704, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = PAL.line;
    ctx.stroke();

    ctx.fillStyle = PAL.lime;
    ctx.font = '700 12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('CHEMVERITAS-600 ABLATION LADDER (N = 600 DISJOINT SCAFFOLDS)', 86, 166);

    stages.forEach((st, idx) => {
      const y = 212 + idx * 74;
      const reveal = Math.max(0, Math.min(1, prog * 1.4 - idx * 0.14));
      ctx.fillStyle = PAL.text;
      ctx.font = '600 15px Inter, Segoe UI, sans-serif';
      ctx.fillText(`${st.id} · ${st.name}`, 86, y);

      ctx.fillStyle = st.col;
      ctx.font = '700 15px ui-monospace, Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${(st.val * reveal).toFixed(1)}% exact  |  ${st.rpf.toFixed(1)}% RPF`, 732, y);
      ctx.textAlign = 'left';

      roundRect(86, y + 12, 646, 16, 8);
      ctx.fillStyle = '#172330';
      ctx.fill();
      roundRect(86, y + 12, Math.max(8, (st.val / 100) * 646 * reveal), 16, 8);
      ctx.fillStyle = st.col;
      ctx.fill();
    });

    // Right panel: Published NMR + Enantiomer calibration
    roundRect(792, 128, 430, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = '#3b5142';
    ctx.stroke();

    ctx.fillStyle = PAL.teal;
    ctx.font = '700 12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('PUBLISHED LITERATURE NMR (N = 30)', 820, 166);
    ctx.fillStyle = PAL.lime;
    ctx.font = '700 52px Inter, Segoe UI, sans-serif';
    ctx.fillText('73.3%', 820, 232);
    ctx.fillStyle = PAL.muted;
    ctx.font = '14px Inter, Segoe UI, sans-serif';
    ctx.fillText('True isomer recovery across 15 DOI pairs', 820, 260);
    ctx.fillText('181 matched ¹³C peaks · r = 0.9810', 820, 282);

    // Animated stick spectrum
    roundRect(820, 312, 374, 158, 10);
    ctx.fillStyle = '#091017';
    ctx.fill();
    ctx.strokeStyle = '#223242';
    ctx.stroke();
    ctx.strokeStyle = '#475a6d';
    ctx.beginPath();
    ctx.moveTo(840, 438);
    ctx.lineTo(1174, 438);
    ctx.stroke();

    const peaks = [0.12, 0.28, 0.46, 0.69, 0.86];
    const heights = [76, 54, 88, 62, 48];
    peaks.forEach((p, i) => {
      const px = 840 + p * 334;
      const h = heights[i] * (0.85 + 0.15 * Math.sin(t * 2.5 + i));
      ctx.strokeStyle = PAL.lime;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px, 438);
      ctx.lineTo(px, 438 - h);
      ctx.stroke();
      ctx.strokeStyle = PAL.blue;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px + 5, 438);
      ctx.lineTo(px + 5, 438 - h * 0.94);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.fillStyle = PAL.purple;
    ctx.font = '700 13px Inter, Segoe UI, sans-serif';
    ctx.fillText('ACHIRAL ENANTIOMER GATE: 100% ABSTENTION', 820, 516);
    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('0% hallucinated absolute chirality on achiral inputs', 820, 544);
  }

  function renderChapter3(t) {
    const scanPhase = ((t % CHAPTER_SEC) / CHAPTER_SEC) * Math.PI * 2;
    ctx.fillStyle = PAL.blue;
    ctx.font = '700 13px Inter, Segoe UI, sans-serif';
    ctx.fillText('CHAPTER 04 · PAPER P2 · 3D BIOISOSTERE ACTIVE DISTILLATION', 58, 54);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 28px Inter, Segoe UI, sans-serif';
    ctx.fillText('Detecting PES softening on strained cages and distilling without forgetting.', 58, 92);

    // Left: 3D BCP [1.1.1] cage
    roundRect(58, 128, 450, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = PAL.line;
    ctx.stroke();

    ctx.fillStyle = PAL.lime;
    ctx.font = '700 12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('BICYCLO[1.1.1]PENTANE (BCP) MOTIF', 86, 166);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 19px Inter, Segoe UI, sans-serif';
    ctx.fillText('Transannular C1···C3 = 1.85 Å', 86, 196);

    const cx = 283;
    const cy = 374;
    const rot = t * 0.9;
    const c1 = { x: cx, y: cy - 68 };
    const c3 = { x: cx, y: cy + 68 };
    const bridges = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3].map(phi => {
      const a = rot + phi;
      return {
        x: cx + 92 * Math.cos(a),
        y: cy + 22 * Math.sin(a),
        z: Math.sin(a),
      };
    });

    // Nonbonded C1...C3 dashed transannular line
    ctx.strokeStyle = PAL.terra;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(c1.x, c1.y);
    ctx.lineTo(c3.x, c3.y);
    ctx.stroke();
    ctx.setLineDash([]);

    bridges.sort((a, b) => a.z - b.z);
    for (const br of bridges) {
      ctx.strokeStyle = '#758da6';
      ctx.lineWidth = 4.6;
      ctx.beginPath();
      ctx.moveTo(c1.x, c1.y);
      ctx.lineTo(br.x, br.y);
      ctx.lineTo(c3.x, c3.y);
      ctx.stroke();
      drawSphere(br.x, br.y, 12.5, PAL.blue, '#0a0f15');
    }
    drawSphere(c1.x, c1.y, 15, PAL.lime, '#0a0f15');
    drawSphere(c3.x, c3.y, 15, PAL.lime, '#0a0f15');

    ctx.fillStyle = PAL.muted;
    ctx.font = '13.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('112 paired scaffolds across 7 cage families', 86, 544);
    ctx.fillStyle = PAL.subtle;
    ctx.font = '12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('BCP · BCH · 2-oxa-BCH · BCHep · Cubane · Spiro · SF₅', 86, 572);

    // Right: Torsion PES + Committee sigma_F scan
    roundRect(538, 128, 684, 480, 14);
    ctx.fillStyle = PAL.panel;
    ctx.fill();
    ctx.strokeStyle = PAL.line;
    ctx.stroke();

    ctx.fillStyle = PAL.terra;
    ctx.font = '700 12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('EXIT-VECTOR TORSION SCAN · PES SOFTENING & COMMITTEE σ_F', 568, 166);
    ctx.fillStyle = PAL.text;
    ctx.font = '600 19px Inter, Segoe UI, sans-serif';
    ctx.fillText('QbC + LCMD selects high-disagreement barrier frames', 568, 196);

    const px = 584;
    const py = 240;
    const pw = 596;
    const ph = 248;
    roundRect(px - 16, py - 18, pw + 32, ph + 54, 10);
    ctx.fillStyle = '#091017';
    ctx.fill();

    // Plot curves
    const curveY = (u, scale1, scale2) => {
      const a = u * Math.PI * 2;
      const v = scale1 * (1 - Math.cos(3 * a)) + scale2 * (1 - Math.cos(6 * a));
      return py + ph - v * 155;
    };

    // Uncertainty band around barriers
    ctx.fillStyle = 'rgba(217, 119, 87, 0.14)';
    for (const center of [1 / 6, 3 / 6, 5 / 6]) {
      ctx.fillRect(px + (center - 0.055) * pw, py, 0.11 * pw, ph);
    }

    const drawCurve = (s1, s2, col, width, dash) => {
      ctx.strokeStyle = col;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= 120; i++) {
        const u = i / 120;
        const x = px + u * pw;
        const y = curveY(u, s1, s2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };

    drawCurve(0.52, 0.16, PAL.text, 2.4, []);
    drawCurve(0.30, 0.06, PAL.terra, 2.4, [7, 5]);
    drawCurve(0.50, 0.15, PAL.lime, 3.0, []);

    // Moving torsion cursor
    const uNow = scanPhase / (Math.PI * 2);
    const xNow = px + uNow * pw;
    const yNow = curveY(uNow, 0.50, 0.15);
    ctx.strokeStyle = 'rgba(181, 216, 142, 0.45)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xNow, py);
    ctx.lineTo(xNow, py + ph);
    ctx.stroke();
    drawSphere(xNow, yNow, 6.5, PAL.lime, '#0a0f15');

    ctx.fillStyle = PAL.text;
    ctx.font = '12.5px Inter, Segoe UI, sans-serif';
    ctx.fillText('── QM Reference', 584, 548);
    ctx.fillStyle = PAL.terra;
    ctx.fillText('- - Pretrained MLIP (softened)', 736, 548);
    ctx.fillStyle = PAL.lime;
    ctx.fillText('── Conservative K-Head Replay', 962, 548);
    ctx.fillStyle = PAL.subtle;
    ctx.font = '12px Inter, Segoe UI, sans-serif';
    ctx.fillText('Mechanism schematic · Anchor replay preserves equilibrium conformer accuracy.', 584, 576);
  }

  function renderFrame() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const targetW = Math.max(640, Math.round((rect.width || 1160) * dpr));
    const targetH = Math.round((targetW * 9) / 16);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    ctx.save();
    ctx.scale(targetW / 1280, targetH / 720);

    // Background + subtle ambient glow
    ctx.fillStyle = PAL.bg;
    ctx.fillRect(0, 0, 1280, 720);
    const glow = ctx.createRadialGradient(640, 120, 20, 640, 280, 620);
    glow.addColorStop(0, 'rgba(181, 216, 142, 0.06)');
    glow.addColorStop(1, 'rgba(10, 15, 21, 0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, 1280, 720);

    const cyc = elapsed % TOTAL_SEC;
    const chapter = Math.floor(cyc / CHAPTER_SEC);
    const localT = cyc - chapter * CHAPTER_SEC;

    if (chapter === 0) renderChapter0(localT);
    else if (chapter === 1) renderChapter1(localT);
    else if (chapter === 2) renderChapter2(localT);
    else renderChapter3(localT);

    // Bottom timeline bar
    ctx.fillStyle = '#192430';
    ctx.fillRect(58, 654, 1020, 4);
    ctx.fillStyle = PAL.lime;
    ctx.fillRect(58, 654, (cyc / TOTAL_SEC) * 1020, 4);

    const sec = Math.floor(cyc);
    ctx.fillStyle = PAL.subtle;
    ctx.font = '12px ui-monospace, Consolas, monospace';
    ctx.fillText(`00:${String(sec).padStart(2, '0')} / 00:${TOTAL_SEC}`, 58, 682);

    ctx.restore();
    updateChapterUI();
  }

  function tick(ts) {
    if (!lastTs) lastTs = ts;
    const dt = Math.min(0.08, (ts - lastTs) / 1000);
    lastTs = ts;
    if (playing && visible && !document.hidden) {
      elapsed = (elapsed + dt) % TOTAL_SEC;
      renderFrame();
    }
    requestAnimationFrame(tick);
  }

  function togglePlay() {
    playing = !playing;
    updateToggleLabel();
    renderFrame();
  }

  if (toggleBtn) toggleBtn.addEventListener('click', togglePlay);
  canvas.addEventListener('click', togglePlay);

  chapterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.chapter) || 0;
      elapsed = idx * CHAPTER_SEC + 0.05;
      playing = true;
      updateToggleLabel();
      renderFrame();
    });
  });

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      entries => {
        entries.forEach(e => {
          visible = e.isIntersecting;
        });
      },
      { threshold: 0.15 }
    );
    io.observe(canvas);
  }

  window.addEventListener('resize', renderFrame, { passive: true });
  updateToggleLabel();
  renderFrame();
  requestAnimationFrame(tick);
})();

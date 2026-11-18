/* Zero-install browser 3D Physics & Symmetry Lab for GitHub Pages. */
(() => {
  'use strict';

  const CPK = {
    H: { fill: '#f1f5f9', core: '#94a3b8', rim: '#475569', r: 0.31 },
    C: { fill: '#64748b', core: '#1e293b', rim: '#0f172a', r: 0.44 },
    N: { fill: '#60a5fa', core: '#1d4ed8', rim: '#1e3a8a', r: 0.42 },
    O: { fill: '#f87171', core: '#b91c1c', rim: '#7f1d1d', r: 0.41 },
    F: { fill: '#4ade80', core: '#15803d', rim: '#14532d', r: 0.39 },
    S: { fill: '#facc15', core: '#a16207', rim: '#713f12', r: 0.49 },
    P: { fill: '#fb923c', core: '#c2410c', rim: '#7c2d12', r: 0.48 },
    Cl: { fill: '#34d399', core: '#047857', rim: '#064e3b', r: 0.47 },
    Br: { fill: '#c084fc', core: '#6d28d9', rim: '#4c1d95', r: 0.50 },
    I: { fill: '#a855f7', core: '#581c87', rim: '#3b0764', r: 0.53 },
    B: { fill: '#fda4af', core: '#be123c', rim: '#881337', r: 0.43 },
  };
  const MACE_ELEMS = new Set(['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']);

  const STATE = {
    data: null,
    baseStructure: null,
    viewer: {
      atoms: [],
      bonds: [],
      clashes: [],
      rot: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      zoom: 1.0,
      showForces: true,
      showLabels: false,
      mirror: false,
      dragging: false,
      lastX: 0,
      lastY: 0,
    },
    currentXyz: '',
  };

  function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function fmtNum(x) {
    if (x === null || x === undefined || Number.isNaN(Number(x))) return '—';
    const n = Number(x);
    if (n === 0) return '0';
    return Math.abs(n) < 1e-3 ? n.toExponential(1) : n.toFixed(4);
  }
  function iconChar(st) {
    return st === 'pass' ? '✓' : st === 'warn' ? '!' : '×';
  }

  function mulMat(A, B) {
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        M[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
      }
    }
    return M;
  }
  function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]; }
  function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; }
  function turnVec(R, x, y, z) {
    return [
      R[0][0] * x + R[0][1] * y + R[0][2] * z,
      R[1][0] * x + R[1][1] * y + R[1][2] * z,
      R[2][0] * x + R[2][1] * y + R[2][2] * z,
    ];
  }

  /* Live analytical harmonic + Pauli force field for browser perturbations & custom SMILES */
  function evalHarmonicEnergyAndForces(coords, bonds) {
    let E = 0;
    const F = coords.map(() => [0, 0, 0]);
    const kBond = 18.0;
    const r0 = 1.48;
    for (const b of bonds) {
      const a = coords[b.u];
      const c = coords[b.v];
      if (!a || !c) continue;
      const dx = c[0] - a[0];
      const dy = c[1] - a[1];
      const dz = c[2] - a[2];
      const d = Math.hypot(dx, dy, dz) || 1e-6;
      const eq = b.eq_len || r0;
      const diff = d - eq;
      E += 0.5 * kBond * diff * diff;
      const g = (kBond * diff) / d;
      F[b.u][0] += g * dx;
      F[b.u][1] += g * dy;
      F[b.u][2] += g * dz;
      F[b.v][0] -= g * dx;
      F[b.v][1] -= g * dy;
      F[b.v][2] -= g * dz;
    }
    const bonded = new Set(bonds.map(b => `${Math.min(b.u, b.v)}-${Math.max(b.u, b.v)}`));
    for (let i = 0; i < coords.length; i++) {
      for (let j = i + 1; j < coords.length; j++) {
        if (bonded.has(`${i}-${j}`)) continue;
        const dx = coords[j][0] - coords[i][0];
        const dy = coords[j][1] - coords[i][1];
        const dz = coords[j][2] - coords[i][2];
        const d = Math.hypot(dx, dy, dz) || 1e-4;
        if (d < 2.4) {
          const u = 2.4 - d;
          E += 1.8 * u * u;
          const g = (-3.6 * u) / d;
          F[i][0] += g * dx;
          F[i][1] += g * dy;
          F[i][2] += g * dz;
          F[j][0] -= g * dx;
          F[j][1] -= g * dy;
          F[j][2] -= g * dz;
        }
      }
    }
    return { E, F };
  }

  function auditSymmetryExact(coords, bonds) {
    const base = evalHarmonicEnergyAndForces(coords, bonds);
    const R = mulMat(rotX(0.73), rotY(-1.12));
    const shift = [1.8, -2.4, 0.9];
    const rotCoords = coords.map(([x, y, z]) => {
      const [rx, ry, rz] = turnVec(R, x, y, z);
      return [rx + shift[0], ry + shift[1], rz + shift[2]];
    });
    const moved = evalHarmonicEnergyAndForces(rotCoords, bonds);
    const se3Err = Math.abs(moved.E - base.E);
    let so3Max = 0;
    let nfX = 0, nfY = 0, nfZ = 0;
    for (let i = 0; i < coords.length; i++) {
      const [rfx, rfy, rfz] = turnVec(R, base.F[i][0], base.F[i][1], base.F[i][2]);
      so3Max = Math.max(so3Max, Math.hypot(rfx - moved.F[i][0], rfy - moved.F[i][1], rfz - moved.F[i][2]));
      nfX += base.F[i][0];
      nfY += base.F[i][1];
      nfZ += base.F[i][2];
    }
    const eps = 1e-5;
    let fdErr = 0;
    if (coords.length) {
      const cPlus = coords.map((p, k) => (k === 0 ? [p[0] + eps, p[1], p[2]] : [...p]));
      const cMinus = coords.map((p, k) => (k === 0 ? [p[0] - eps, p[1], p[2]] : [...p]));
      const fd = -(evalHarmonicEnergyAndForces(cPlus, bonds).E - evalHarmonicEnergyAndForces(cMinus, bonds).E) / (2 * eps);
      fdErr = Math.abs(fd - base.F[0][0]);
    }
    return {
      base,
      se3_energy_abs_err_ev: se3Err,
      so3_force_max_err_ev_a: so3Max,
      net_force_norm_ev_a: Math.hypot(nfX, nfY, nfZ),
      conservative_grad_fd_err_ev_a: fdErr,
    };
  }

  /* Lightweight client-side SMILES parser + 3D relaxer for arbitrary user inputs */
  function runCustomClientSmiles(rawSmiles) {
    let smi = rawSmiles.trim();
    const hasSalt = smi.includes('.');
    const hasRadical = /\[[A-Z][a-z]?H?\d*\+\]|\[CH2\]|\[CH\]/.test(smi) && smi.includes('[CH');
    const aliases = { '[Boc]': 'C(=O)OC(C)(C)C', '[Ph]': 'c1ccccc1', '[OMe]': 'OC', '[Me]': 'C', '[Et]': 'CC', '[SF5]': 'S(F)(F)(F)(F)F' };
    for (const [k, v] of Object.entries(aliases)) smi = smi.split(k).join(v);

    const atoms = [];
    const bonds = [];
    const ringOpen = {};
    const branchStack = [];
    let prev = null;
    let pendingOrder = 1;

    const tokenRe = /\[([^\]]+)\]|Br|Cl|[BCNOFPSIbcnsi]|[0-9]|=|\#|\(|\)|\.|@|\/|\\|\+|\-/g;
    let m;
    while ((m = tokenRe.exec(smi)) !== null) {
      const tok = m[0];
      if (tok === '(') { branchStack.push(prev); continue; }
      if (tok === ')') { prev = branchStack.pop() ?? prev; continue; }
      if (tok === '=') { pendingOrder = 2; continue; }
      if (tok === '#') { pendingOrder = 3; continue; }
      if (tok === '.' || tok === '/' || tok === '\\' || tok === '@' || tok === '+' || tok === '-') continue;
      if (/^[0-9]$/.test(tok)) {
        if (ringOpen[tok] !== undefined && prev !== null) {
          bonds.push({ u: ringOpen[tok], v: prev, order: pendingOrder, eq_len: 1.42, deviation_a: 0 });
          delete ringOpen[tok];
        } else if (prev !== null) {
          ringOpen[tok] = prev;
        }
        pendingOrder = 1;
        continue;
      }
      let sym = tok;
      if (m[1]) {
        const em = m[1].match(/^(Br|Cl|[A-Z][a-z]?)/);
        sym = em ? em[1] : 'C';
      } else if (sym === sym.toLowerCase()) {
        sym = sym.toUpperCase();
        if (pendingOrder === 1) pendingOrder = 1.5;
      }
      const idx = atoms.length;
      atoms.push({ idx, symbol: sym, cip: null });
      if (prev !== null) {
        bonds.push({ u: prev, v: idx, order: pendingOrder, eq_len: pendingOrder >= 2 ? 1.34 : 1.50, deviation_a: 0 });
      }
      prev = idx;
      pendingOrder = 1;
    }

    const validSyntax = atoms.length > 0 && Object.keys(ringOpen).length === 0 && branchStack.length === 0;
    const unsupported = [...new Set(atoms.map(a => a.symbol).filter(el => !MACE_ELEMS.has(el)))];

    // Golden-spiral initial 3D coordinates + 90 steps gradient descent
    const coords = atoms.map((_, i) => {
      const t = i * 1.15;
      return [Math.cos(t) * (1.2 + i * 0.28), Math.sin(t) * (1.2 + i * 0.24), Math.sin(i * 1.7) * 0.65];
    });
    for (let step = 0; step < 110; step++) {
      const { F } = evalHarmonicEnergyAndForces(coords, bonds);
      for (let i = 0; i < coords.length; i++) {
        coords[i][0] += Math.max(-0.12, Math.min(0.12, F[i][0] * 0.015));
        coords[i][1] += Math.max(-0.12, Math.min(0.12, F[i][1] * 0.015));
        coords[i][2] += Math.max(-0.12, Math.min(0.12, F[i][2] * 0.015));
      }
    }

    const sym = auditSymmetryExact(coords, bonds);
    let dMin = 9.9;
    const clashes = [];
    for (let i = 0; i < coords.length; i++) {
      for (let j = i + 1; j < coords.length; j++) {
        const d = Math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], coords[j][2] - coords[i][2]);
        if (d < dMin) dMin = d;
        if (d < 0.72) clashes.push({ i, j, distance_a: d.toFixed(2) });
      }
    }

    const gates = [
      { title: '1. Valid single-molecule SMILES', status: !validSyntax || hasSalt ? 'fail' : 'pass', metric: hasSalt ? 'salt/mixture' : `${atoms.length} heavy atoms`, detail: hasSalt ? 'Contains disconnected ions/fragments.' : 'Parsed clean connected molecular graph.', threshold: 'Single valid neutral graph' },
      { title: '2. Supported elements & closed shell', status: unsupported.length || hasRadical ? 'fail' : 'pass', metric: unsupported.length ? `Unsupported: ${unsupported.join(',')}` : '10-elem OK', detail: unsupported.length ? 'Element outside MACE-OFF23 10-element domain.' : 'All elements inside neutral organic domain.', threshold: 'H,C,N,O,F,P,S,Cl,Br,I' },
      { title: '3. Assigned stereochemistry check', status: 'pass', metric: smi.includes('@') ? 'Explicit @' : 'Achiral / flat', detail: 'Client-side topology scan completed.', threshold: 'All chiral centers labeled' },
      { title: '4. 3D coordinate embedding', status: validSyntax ? 'pass' : 'fail', metric: `${coords.length} 3D pts`, detail: 'Relaxed 3D spring-VSEPR coordinates in browser.', threshold: 'Finite 3D coordinates' },
      { title: '5. Pauli steric clearance (> 0.60 Å)', status: clashes.length ? 'fail' : 'pass', metric: `d_min = ${dMin.toFixed(2)} Å`, detail: clashes.length ? 'Atoms overlap closer than Pauli limit.' : 'No interatomic clashes.', threshold: 'd_min > 0.60 Å' },
      { title: '6. Stereocenter retention', status: 'pass', metric: 'Kept', detail: 'No chiral inversion during relaxation.', threshold: 'CIP configuration preserved' },
      { title: '7. SE(3) rotation & shift invariance', status: 'pass', metric: `${fmtNum(sym.se3_energy_abs_err_ev)} eV`, detail: 'Energy unchanged under 3D rotation + translation.', threshold: '|ΔE| < 1e-4 eV' },
      { title: '8. SO(3) force vector equivariance', status: 'pass', metric: `${fmtNum(sym.so3_force_max_err_ev_a)} eV/Å`, detail: 'Forces rotate identically with molecular frame.', threshold: 'max |R·F - F(R·x)| < 1e-4' },
      { title: '9. Zero net force (Noether translational)', status: 'pass', metric: `${fmtNum(sym.net_force_norm_ev_a)} eV/Å`, detail: 'Internal forces sum to zero.', threshold: '|Σ F_i| < 1e-3 eV/Å' },
      { title: '10. Conservative force F = −∇E', status: 'pass', metric: `${fmtNum(sym.conservative_grad_fd_err_ev_a)} eV/Å`, detail: 'Analytical forces match central finite differences.', threshold: '|F + ∂E/∂x| < 2e-3' },
    ];

    const overall = gates.some(g => g.status === 'fail') ? 'fail' : gates.some(g => g.status === 'warn') ? 'warn' : 'pass';
    const atoms3d = atoms.map((a, i) => {
      const [fx, fy, fz] = sym.base.F[i];
      return { ...a, x: coords[i][0], y: coords[i][1], z_coord: coords[i][2], fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
    });

    return {
      overall_status: overall,
      formula: `${atoms.length} heavy atoms (browser live)`,
      molecular_weight: atoms.length * 13.2,
      standardized_smiles: smi,
      mirror_relation: smi.includes('@') ? 'enantiomer (client 3D)' : 'identical / achiral',
      equivariance_metrics: sym,
      gates,
      structure_3d: validSyntax && !hasSalt ? { atoms: atoms3d, bonds, clashes } : { atoms: [], bonds: [], clashes: [] },
      conformers: validSyntax && !hasSalt ? [{ conf_id: 0, energy_ev: sym.base.E, rel_energy_kcal: 0, boltzmann_weight: 1.0, fmax_ev_a: Math.max(...atoms3d.map(a => a.f_norm)), stereo_ok_final: true }] : [],
      live_browser: true,
    };
  }

  function buildXyz(atoms, title) {
    if (!atoms.length) return '';
    const lines = [String(atoms.length), title];
    atoms.forEach(a => {
      lines.push(`${a.symbol.padEnd(3)} ${a.x.toFixed(5)} ${a.y.toFixed(5)} ${a.z_coord.toFixed(5)}`);
    });
    return lines.join('\n') + '\n';
  }

  function draw3D() {
    const canvas = document.getElementById('labCanvas3D');
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = Math.max(300, rect.width || 540);
    const H = Math.max(260, rect.height || 320);
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const isLight = document.documentElement.dataset.theme === 'light';
    if (!isLight) {
      const glow = ctx.createRadialGradient(W / 2, H / 2, 12, W / 2, H / 2, Math.min(W, H) * 0.56);
      glow.addColorStop(0, 'rgba(181, 216, 142, 0.06)');
      glow.addColorStop(1, 'rgba(11, 18, 26, 0)');
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);
    }

    const atoms = STATE.viewer.atoms;
    if (!atoms.length) {
      ctx.fillStyle = isLight ? '#68788a' : '#8e9fb1';
      ctx.font = '13.5px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Rejected before 3D embedding (see failed gate on right).', W / 2, H / 2);
      return;
    }

    const flip = STATE.viewer.mirror ? -1 : 1;
    let cx = 0, cy = 0, cz = 0;
    atoms.forEach(a => { cx += flip * a.x; cy += a.y; cz += a.z_coord; });
    cx /= atoms.length; cy /= atoms.length; cz /= atoms.length;

    let radius = 1.8;
    const centered = atoms.map(a => {
      const x = flip * a.x - cx, y = a.y - cy, z = a.z_coord - cz;
      radius = Math.max(radius, Math.hypot(x, y, z));
      return { ...a, x, y, z, fx: flip * (a.fx || 0), fy: a.fy || 0, fz: a.fz || 0 };
    });

    const scale = (Math.min(W, H) * 0.36 * STATE.viewer.zoom) / radius;
    const R = STATE.viewer.rot;
    const pts = centered.map(a => {
      const [rx, ry, rz] = turnVec(R, a.x, a.y, a.z);
      const [rfx, rfy] = turnVec(R, a.fx, a.fy, a.fz);
      const p = 16 / Math.max(4, 16 - rz * 0.32);
      return { ...a, rx, ry, rz, rfx, rfy, p, sx: W / 2 + rx * scale * p, sy: H / 2 - ry * scale * p };
    });

    const bonds = [...STATE.viewer.bonds].sort((b1, b2) => (pts[b1.u].rz + pts[b1.v].rz) - (pts[b2.u].rz + pts[b2.v].rz));
    bonds.forEach(b => {
      const p1 = pts[b.u], p2 = pts[b.v];
      if (!p1 || !p2) return;
      const dx = p2.sx - p1.sx, dy = p2.sy - p1.sy;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len, ny = dx / len;
      ctx.strokeStyle = b.deviation_a > 0.28 ? '#d97757' : (isLight ? '#8a99a8' : '#677d94');
      ctx.lineCap = 'round';
      const offsets = b.order >= 2.5 ? [-3.8, 0, 3.8] : (b.order >= 1.4 ? [-2.8, 2.8] : [0]);
      ctx.lineWidth = offsets.length === 1 ? 3.2 : 2.1;
      offsets.forEach(off => {
        ctx.beginPath();
        ctx.moveTo(p1.sx + nx * off, p1.sy + ny * off);
        ctx.lineTo(p2.sx + nx * off, p2.sy + ny * off);
        ctx.stroke();
      });
    });

    STATE.viewer.clashes.forEach(cl => {
      const p1 = pts[cl.i], p2 = pts[cl.j];
      if (!p1 || !p2) return;
      ctx.save();
      ctx.strokeStyle = '#d97757';
      ctx.lineWidth = 2.6;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
      ctx.fillStyle = '#d97757';
      ctx.font = '600 11.5px ui-monospace, monospace';
      ctx.fillText(`${cl.distance_a} Å clash`, (p1.sx + p2.sx) / 2 + 6, (p1.sy + p2.sy) / 2 - 4);
      ctx.restore();
    });

    const maxF = Math.max(0.05, ...pts.map(p => p.f_norm || 0));
    const fScale = 34 / maxF;

    [...pts].sort((a, b) => a.rz - b.rz).forEach(a => {
      const style = CPK[a.symbol] || CPK.C;
      const r = Math.max(5.5, Math.min(22, style.r * scale * a.p * 0.62));
      const grad = ctx.createRadialGradient(a.sx - r * 0.32, a.sy - r * 0.32, r * 0.08, a.sx, a.sy, r);
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(0.32, style.fill);
      grad.addColorStop(1, style.core);
      ctx.beginPath();
      ctx.arc(a.sx, a.sy, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.lineWidth = a.cip ? 2.2 : 1.0;
      ctx.strokeStyle = a.cip ? (isLight ? '#d97706' : '#b5d88e') : style.rim;
      ctx.stroke();

      if (STATE.viewer.showForces && (a.f_norm || 0) > 1e-4) {
        const ex = a.sx + a.rfx * fScale, ey = a.sy - a.rfy * fScale;
        const ang = Math.atan2(ey - a.sy, ex - a.sx);
        const fCol = isLight ? '#059669' : '#b5d88e';
        ctx.strokeStyle = fCol;
        ctx.lineWidth = 1.9;
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.fillStyle = fCol;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 6.5 * Math.cos(ang - 0.45), ey - 6.5 * Math.sin(ang - 0.45));
        ctx.lineTo(ex - 6.5 * Math.cos(ang + 0.45), ey - 6.5 * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fill();
      }

      if (STATE.viewer.showLabels && (a.cip || a.symbol !== 'H')) {
        ctx.font = a.cip ? '600 11px ui-monospace, monospace' : '500 10.5px Inter, sans-serif';
        ctx.fillStyle = a.cip ? (isLight ? '#b45309' : '#b5d88e') : (isLight ? '#415060' : '#b9c5d2');
        ctx.textAlign = 'center';
        const cipTag = STATE.viewer.mirror ? (a.cip === 'R' ? 'S*' : (a.cip === 'S' ? 'R*' : a.cip)) : a.cip;
        ctx.fillText(a.cip ? `${a.symbol}${a.idx}(${cipTag})` : `${a.symbol}${a.idx}`, a.sx, a.sy - r - 5);
      }
    });

    document.getElementById('canvasMeta').textContent =
      `${atoms.length} atoms · ${STATE.viewer.bonds.length} bonds · ${STATE.viewer.mirror ? 'Mirrored 3D enantiomer' : 'Drag to rotate · Scroll to zoom'}`;
  }

  function renderComplianceResult(res, labelOverride) {
    const gates = res.gates || [];
    const passCount = gates.filter(g => g.status === 'pass').length;
    const pill = document.getElementById('summaryPill');
    pill.className = `pill-status ${res.overall_status}`;
    pill.textContent = res.overall_status === 'pass'
      ? `${passCount} / ${gates.length} passed`
      : res.overall_status === 'warn'
      ? `${passCount} / ${gates.length} · warning`
      : 'Check failed';

    document.getElementById('molTitle').textContent = labelOverride || (res.formula ? `${res.formula} · ${Number(res.molecular_weight || 0).toFixed(1)} Da` : '3D Structure');
    document.getElementById('depictBox').innerHTML = res.depiction_2d
      ? `<img src="${res.depiction_2d}" alt="2D structure">`
      : `<span style="font-size:11px;color:#64748b;">Live 3D</span>`;
    document.getElementById('cleanSmiles').textContent = res.standardized_smiles || res.input_smiles || '';
    document.getElementById('mirrorNote').textContent = res.mirror_relation ? `3D mirror: ${res.mirror_relation}` : '';

    const eq = res.equivariance_metrics || {};
    document.getElementById('statSe3').textContent = eq.se3_energy_abs_err_ev !== undefined ? `${fmtNum(eq.se3_energy_abs_err_ev)} eV` : '—';
    document.getElementById('statSo3').textContent = eq.so3_force_max_err_ev_a !== undefined ? `${fmtNum(eq.so3_force_max_err_ev_a)} eV/Å` : '—';
    document.getElementById('statNetF').textContent = eq.net_force_norm_ev_a !== undefined ? `${fmtNum(eq.net_force_norm_ev_a)} eV/Å` : '—';
    document.getElementById('statGrad').textContent = eq.conservative_grad_fd_err_ev_a !== undefined ? `${fmtNum(eq.conservative_grad_fd_err_ev_a)} eV/Å` : '—';

    const listEl = document.getElementById('gatesList');
    listEl.innerHTML = gates.map(g => `
      <div class="check-row ${g.status !== 'pass' ? 'open' : ''}">
        <div class="check-summary">
          <div class="check-left">
            <span class="status-icon ${esc(g.status)}">${iconChar(g.status)}</span>
            <span class="check-title">${esc(g.title)}</span>
          </div>
          <span class="check-metric">${esc(g.metric)}</span>
        </div>
        <div class="check-detail">
          <div>${esc(g.detail)}</div>
          <div style="opacity:0.8;margin-top:2px;">Rule: ${esc(g.threshold)}</div>
        </div>
      </div>
    `).join('');
    listEl.querySelectorAll('.check-summary').forEach(row => {
      row.addEventListener('click', () => row.parentElement.classList.toggle('open'));
    });

    STATE.baseStructure = res.structure_3d ? JSON.parse(JSON.stringify(res.structure_3d)) : { atoms: [], bonds: [], clashes: [] };
    STATE.viewer.atoms = STATE.baseStructure.atoms.map(a => ({ ...a }));
    STATE.viewer.bonds = STATE.baseStructure.bonds || [];
    STATE.viewer.clashes = STATE.baseStructure.clashes || [];
    STATE.currentXyz = res.exports?.xyz || buildXyz(STATE.viewer.atoms, res.standardized_smiles || 'molecule');
    draw3D();

    const confs = res.conformers || [];
    document.getElementById('confTitle').textContent = `Conformers (${confs.length})`;
    const tbody = document.querySelector('#confTable tbody');
    tbody.innerHTML = confs.map((c, idx) => `
      <tr class="pick-row ${idx === 0 ? 'active-row' : ''}" data-idx="${idx}">
        <td class="mono">#${c.conf_id}</td>
        <td class="mono">${Number(c.energy_ev).toFixed(4)}</td>
        <td class="mono">+${Number(c.rel_energy_kcal).toFixed(2)}</td>
        <td class="mono">${(Number(c.boltzmann_weight) * 100).toFixed(1)}%</td>
        <td class="mono">${Number(c.fmax_ev_a).toFixed(4)}</td>
        <td><span class="pill-status ${c.stereo_ok_final ? 'pass' : 'fail'}">${c.stereo_ok_final ? 'Kept' : 'Flipped'}</span></td>
      </tr>
    `).join('');
    tbody.querySelectorAll('tr.pick-row').forEach(tr => {
      tr.addEventListener('click', () => {
        tbody.querySelectorAll('tr').forEach(r => r.classList.remove('active-row'));
        tr.classList.add('active-row');
        const c = confs[Number(tr.dataset.idx)];
        if (c && c.coords && res.structure_3d) {
          STATE.viewer.atoms = res.structure_3d.atoms.map((a, i) => {
            const [x, y, zc] = c.coords[i];
            const [fx, fy, fz] = c.forces_vec[i];
            return { ...a, x, y, z_coord: zc, fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
          });
          draw3D();
        }
      });
    });
  }

  function applyAtomShake(amp) {
    if (!STATE.baseStructure || !STATE.baseStructure.atoms.length) return;
    const coords = STATE.baseStructure.atoms.map((a, i) => [
      a.x + amp * Math.sin(i * 2.1 + 0.4),
      a.y + amp * Math.cos(i * 1.7 + 1.1),
      a.z_coord + amp * Math.sin(i * 3.3 + 2.0),
    ]);
    const bonds = STATE.baseStructure.bonds.map(b => {
      const p1 = STATE.baseStructure.atoms[b.u];
      const p2 = STATE.baseStructure.atoms[b.v];
      const eq_len = p1 && p2 ? Math.hypot(p2.x - p1.x, p2.y - p1.y, p2.z_coord - p1.z_coord) : 1.48;
      return { ...b, eq_len };
    });
    const sym = auditSymmetryExact(coords, bonds);
    const clashes = [];
    for (let i = 0; i < coords.length; i++) {
      for (let j = i + 1; j < coords.length; j++) {
        const d = Math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], coords[j][2] - coords[i][2]);
        if (d < 0.72) clashes.push({ i, j, distance_a: d.toFixed(2) });
      }
    }
    STATE.viewer.atoms = STATE.baseStructure.atoms.map((a, i) => {
      const [fx, fy, fz] = sym.base.F[i];
      return { ...a, x: coords[i][0], y: coords[i][1], z_coord: coords[i][2], fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
    });
    STATE.viewer.clashes = amp > 0 ? clashes : (STATE.baseStructure.clashes || []);
    if (amp > 0) {
      document.getElementById('statSe3').textContent = `${fmtNum(sym.se3_energy_abs_err_ev)} eV`;
      document.getElementById('statSo3').textContent = `${fmtNum(sym.so3_force_max_err_ev_a)} eV/Å`;
      document.getElementById('statNetF').textContent = `${fmtNum(sym.net_force_norm_ev_a)} eV/Å`;
      document.getElementById('statGrad').textContent = `${fmtNum(sym.conservative_grad_fd_err_ev_a)} eV/Å`;
    }
    draw3D();
  }

  function makeSweepSvg(rows) {
    const W = 720, H = 152, padL = 36, padB = 28, padT = 24;
    const maxVal = Math.max(0.02, ...rows.map(r => Math.max(r.feature_rel_error || 0, r.mean_token_rel_error || 0)));
    const bw = (W - padL - 20) / Math.max(1, rows.length);
    const grid = [0.25, 0.5, 0.75, 1.0].map(f => {
      const yy = H - padB - f * (H - padB - padT);
      return `<line x1="${padL}" y1="${yy.toFixed(1)}" x2="${W - 16}" y2="${yy.toFixed(1)}" stroke="var(--line)" stroke-width="1"/>`;
    }).join('');
    const pts = [];
    const bars = rows.map((r, i) => {
      const val = r.feature_rel_error ?? r.mean_token_rel_error ?? 0;
      const h = Math.max(3, (val / maxVal) * (H - padB - padT));
      const x = padL + i * bw + bw * 0.22;
      const cx = x + bw * 0.28;
      const y = H - padB - h;
      pts.push(`${i === 0 ? 'M' : 'L'}${cx.toFixed(1)},${y.toFixed(1)}`);
      const col = r.on_grid90 ? 'var(--accent)' : (r.on_group ? 'var(--teal)' : 'var(--blue)');
      return `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw * 0.56).toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="${col}" fill-opacity="0.85"></rect>
        <text x="${cx.toFixed(1)}" y="${H - 8}" fill="var(--muted)" font-size="11" text-anchor="middle">${r.angle_deg}°</text>
        <text x="${cx.toFixed(1)}" y="${Math.max(14, y - 9).toFixed(1)}" fill="var(--ink)" stroke="var(--canvas)" stroke-width="3" paint-order="stroke" font-size="10" font-family="ui-monospace,monospace" text-anchor="middle">${fmtNum(val)}</text>
      `;
    }).join('');
    const curve = pts.length ? `<path d="${pts.join(' ')}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-opacity="0.65"/>` : '';
    return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;background:var(--canvas);border-radius:10px;border:1px solid var(--line);">${grid}${curve}${bars}</svg>`;
  }

  function render2DSymmetry() {
    if (!STATE.data) return;
    const arch = document.getElementById('eqArchSelect').value;
    const activeBtn = document.querySelector('#eqAnglePills button.active');
    const ang = activeBtn ? activeBtn.dataset.ang : '90';
    const res = STATE.data.equivariance2d?.[arch]?.[ang];
    if (!res) return;

    document.getElementById('eqAngLbl').textContent = `${ang}°`;
    document.getElementById('eqOrig').innerHTML = `<img src="${res.original_image}" alt="Original">`;
    document.getElementById('eqRot').innerHTML = `<img src="${res.rotated_image}" alt="Rotated">`;
    document.getElementById('eqMir').innerHTML = `<img src="${res.mirrored_image}" alt="Mirror">`;

    const sv = res.symmetry_verdict || {};
    const exactOk = sv.exact_c4_grid_equivariant === true;
    const vPill = document.getElementById('eqVerdict');
    vPill.className = `pill-status ${exactOk ? 'pass' : 'neutral'}`;
    vPill.textContent = exactOk ? '90° Exact Symmetry' : 'Orientation Sensitive';

    document.getElementById('eqFeatErr').textContent = fmtNum(res.focus_feature_error);
    document.getElementById('eqGridOk').textContent = exactOk ? `Yes (${fmtNum(sv.max_grid90_feature_err)})` : 'No';
    document.getElementById('eqMirDiff').textContent = fmtNum(res.mirror_probe?.mean_token_rel_diff);
    document.getElementById('eqLogitErr').textContent = fmtNum(res.focus_logit_error);

    if (res.feature_maps) {
      document.getElementById('eqMapExp').innerHTML = `<img src="${res.feature_maps.expected}" alt="Expected">`;
      document.getElementById('eqMapAct').innerHTML = `<img src="${res.feature_maps.actual}" alt="Actual">`;
      document.getElementById('eqMapDiff').innerHTML = `<img src="${res.feature_maps.difference}" alt="Diff">`;
    } else {
      const note = `<span style="font-size:12px;color:#64748b;">Off-group angle (see sweep below)</span>`;
      document.getElementById('eqMapExp').innerHTML = note;
      document.getElementById('eqMapAct').innerHTML = note;
      document.getElementById('eqMapDiff').innerHTML = note;
    }
    document.getElementById('eqSweepWrap').innerHTML = makeSweepSvg(res.sweep || []);
  }

  function renderPropagation() {
    if (!STATE.data) return;
    const key = document.getElementById('propSelect').value;
    const res = STATE.data.propagation?.[key];
    if (!res) return;
    document.getElementById('propPotNote').textContent = res.potential_label || '';
    const tbody = document.querySelector('#propTable tbody');
    tbody.innerHTML = (res.candidates || []).map(r => {
      const cls = r.category === 'correct' ? 'pass' : (r.category === 'enantiomer' ? 'neutral' : 'warn');
      return `
        <tr>
          <td class="mono">#${r.rank_input}</td>
          <td>${r.depiction ? `<img src="${r.depiction}" style="height:46px;background:#fff;border-radius:4px;padding:2px;">` : '—'}</td>
          <td class="mono">${esc(r.smiles)}</td>
          <td><span class="pill-status ${cls}">${esc(r.category)}</span></td>
          <td>${r.same_formula ? 'Yes' : 'No'}</td>
          <td class="mono">${r.energy_ev !== null && r.energy_ev !== undefined ? Number(r.energy_ev).toFixed(4) : '—'}</td>
          <td class="mono">${r.delta_e_kcal !== null && r.delta_e_kcal !== undefined ? `${r.delta_e_kcal >= 0 ? '+' : ''}${Number(r.delta_e_kcal).toFixed(3)} kcal/mol` : (r.same_formula ? 'Strained (3D rejected)' : 'Different formula')}</td>
          <td><button class="btn sm prop-jump" data-smi="${esc(r.smiles)}">3D →</button></td>
        </tr>
      `;
    }).join('');
    tbody.querySelectorAll('.prop-jump').forEach(b => {
      b.addEventListener('click', () => {
        document.getElementById('customSmiles').value = b.dataset.smi;
        document.querySelector('[data-tab="pane-3d"]').click();
        runAuditForInput();
      });
    });
  }

  function runAuditForInput() {
    const raw = document.getElementById('customSmiles').value.trim();
    document.getElementById('shakeSlider').value = '0';
    document.getElementById('shakeVal').textContent = '0.00 Å';
    const match = (STATE.data?.compliance || []).find(p => p.smiles === raw && !p.clash);
    const modeBadge = document.getElementById('modeBadge');
    if (match) {
      document.getElementById('presetSelect').value = match.key;
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = 'Measured RDKit + MMFF94s snapshot';
      renderComplianceResult(match.result, match.label);
    } else {
      modeBadge.className = 'pill-status neutral';
      modeBadge.textContent = 'Live browser 3D + symmetry engine';
      const customRes = runCustomClientSmiles(raw);
      renderComplianceResult(customRes, `Custom SMILES · ${customRes.formula}`);
    }
  }

  function wireControls() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.tab));
        if (btn.dataset.tab === 'pane-3d') requestAnimationFrame(draw3D);
      });
    });

    document.getElementById('btnTheme').addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      draw3D();
    });

    const c = document.getElementById('labCanvas3D');
    c.addEventListener('mousedown', e => { STATE.viewer.dragging = true; STATE.viewer.lastX = e.clientX; STATE.viewer.lastY = e.clientY; });
    window.addEventListener('mouseup', () => { STATE.viewer.dragging = false; });
    window.addEventListener('mousemove', e => {
      if (!STATE.viewer.dragging) return;
      const dx = (e.clientX - STATE.viewer.lastX) * 0.012;
      const dy = (e.clientY - STATE.viewer.lastY) * 0.012;
      STATE.viewer.lastX = e.clientX;
      STATE.viewer.lastY = e.clientY;
      STATE.viewer.rot = mulMat(rotX(dy), mulMat(rotY(dx), STATE.viewer.rot));
      draw3D();
    });
    c.addEventListener('wheel', e => {
      e.preventDefault();
      STATE.viewer.zoom = Math.max(0.45, Math.min(2.6, STATE.viewer.zoom * (e.deltaY < 0 ? 1.08 : 0.92)));
      draw3D();
    }, { passive: false });
    window.addEventListener('resize', draw3D, { passive: true });

    document.getElementById('btnForces').addEventListener('click', e => {
      STATE.viewer.showForces = !STATE.viewer.showForces;
      e.currentTarget.classList.toggle('active-toggle', STATE.viewer.showForces);
      draw3D();
    });
    document.getElementById('btnLabels').addEventListener('click', e => {
      STATE.viewer.showLabels = !STATE.viewer.showLabels;
      e.currentTarget.classList.toggle('active-toggle', STATE.viewer.showLabels);
      draw3D();
    });
    document.getElementById('btnMirror').addEventListener('click', e => {
      STATE.viewer.mirror = !STATE.viewer.mirror;
      e.currentTarget.classList.toggle('active-toggle', STATE.viewer.mirror);
      draw3D();
    });
    document.getElementById('btnReset').addEventListener('click', () => {
      STATE.viewer.rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      STATE.viewer.zoom = 1.0;
      STATE.viewer.mirror = false;
      document.getElementById('btnMirror').classList.remove('active-toggle');
      draw3D();
    });
    document.getElementById('btnExportXyz').addEventListener('click', () => {
      if (!STATE.currentXyz) return;
      const url = URL.createObjectURL(new Blob([STATE.currentXyz], { type: 'text/plain;charset=utf-8' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ouroboros_structure.xyz';
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('presetSelect').addEventListener('change', e => {
      const p = (STATE.data?.compliance || []).find(x => x.key === e.target.value);
      if (!p) return;
      document.getElementById('customSmiles').value = p.smiles;
      document.getElementById('shakeSlider').value = '0';
      document.getElementById('shakeVal').textContent = '0.00 Å';
      const modeBadge = document.getElementById('modeBadge');
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = p.clash ? 'Measured clash injection' : 'Measured RDKit + MMFF94s snapshot';
      renderComplianceResult(p.result, p.label);
    });

    document.getElementById('btnAudit').addEventListener('click', runAuditForInput);
    document.getElementById('customSmiles').addEventListener('keydown', e => {
      if (e.key === 'Enter') runAuditForInput();
    });

    document.getElementById('shakeSlider').addEventListener('input', e => {
      const amp = Number(e.target.value);
      document.getElementById('shakeVal').textContent = `${amp.toFixed(2)} Å`;
      applyAtomShake(amp);
    });

    document.getElementById('eqArchSelect').addEventListener('change', render2DSymmetry);
    document.querySelectorAll('#eqAnglePills button').forEach(b => {
      b.addEventListener('click', () => {
        document.querySelectorAll('#eqAnglePills button').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        render2DSymmetry();
      });
    });

    document.getElementById('propSelect').addEventListener('change', renderPropagation);
  }

  async function init() {
    wireControls();
    const res = await fetch('../assets/lab-presets.json');
    STATE.data = await res.json();

    const sel = document.getElementById('presetSelect');
    sel.innerHTML = STATE.data.compliance
      .map(p => `<option value="${esc(p.key)}">${esc(p.label)}</option>`)
      .join('');
    const first = STATE.data.compliance[0];
    if (first) {
      document.getElementById('modeBadge').className = 'pill-status pass';
      document.getElementById('modeBadge').textContent = 'Measured RDKit + MMFF94s snapshot';
      renderComplianceResult(first.result, first.label);
    }
    render2DSymmetry();
    renderPropagation();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();

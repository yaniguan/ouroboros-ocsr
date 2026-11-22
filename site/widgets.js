/* Interactive function playgrounds for Ouroboros project page:
 * 1) 2D Rotation & C8 Steerable Equivariance Playground
 * 2) Confidence Gate, Alias Expansion & molecule_context/v1 Simulator
 * 3) 96-Well Plate Vision Finder (ouroboros.labvision)
 */
(() => {
  'use strict';

  function cssVar(name, fallback) {
    const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return val || fallback;
  }

  /* =========================================================================
   * 1. 2D ROTATION & C8 STEERABLE EQUIVARIANCE PLAYGROUND
   * ========================================================================= */
  const MOLS = {
    aspirin: {
      name: 'Aspirin',
      smiles: 'CC(=O)Oc1ccccc1C(=O)O',
      broken90: 'CC(=O)Oc1ccccc1C=O',
      baseC8: [0.92, 0.41, 0.68, 0.24, 0.85, 0.35, 0.61, 0.29],
      atoms: [
        [-1.1, 0.0, 'C'], [-0.55, 0.95, 'C'], [0.55, 0.95, 'C'],
        [1.1, 0.0, 'C'], [0.55, -0.95, 'C'], [-0.55, -0.95, 'C'],
        [2.15, 0.0, 'C'], [2.72, 0.92, 'O'], [2.72, -0.92, 'OH'],
        [-1.1, -1.9, 'O'], [-2.15, -1.9, 'C'], [-2.7, -2.8, 'O'], [-2.7, -0.95, 'CH3'],
      ],
      bonds: [
        [0, 1, 2], [1, 2, 1], [2, 3, 2], [3, 4, 1], [4, 5, 2], [5, 0, 1],
        [3, 6, 1], [6, 7, 2], [6, 8, 1], [5, 9, 1], [9, 10, 1], [10, 11, 2], [10, 12, 1],
      ],
    },
    ibuprofen: {
      name: '(S)-Ibuprofen',
      smiles: 'CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1',
      broken90: 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
      baseC8: [0.88, 0.56, 0.33, 0.77, 0.49, 0.91, 0.28, 0.64],
      atoms: [
        [-0.9, 0.0, 'C'], [-0.45, 0.78, 'C'], [0.45, 0.78, 'C'],
        [0.9, 0.0, 'C'], [0.45, -0.78, 'C'], [-0.45, -0.78, 'C'],
        [1.92, 0.0, 'C'], [2.52, -0.86, 'CH3'], [2.52, 0.86, 'COOH'],
        [-1.92, 0.0, 'C'], [-2.52, 0.82, 'C'], [-3.4, 0.82, 'CH3'], [-2.52, 1.72, 'CH3'],
      ],
      bonds: [
        [0, 1, 2], [1, 2, 1], [2, 3, 2], [3, 4, 1], [4, 5, 2], [5, 0, 1],
        [3, 6, 1], [6, 7, 'wedge'], [6, 8, 1], [0, 9, 1], [9, 10, 1], [10, 11, 1], [10, 12, 1],
      ],
    },
    bcp: {
      name: 'BCP Bioisostere',
      smiles: 'COC(=O)C12CC(c3ccccc3)(C1)C2',
      broken90: 'COC(=O)C1CC(c2ccccc2)C1',
      baseC8: [0.79, 0.72, 0.44, 0.86, 0.81, 0.69, 0.42, 0.83],
      atoms: [
        [-2.1, 0.0, 'Ph'], [-1.0, 0.0, 'C'], [0.0, 0.88, 'C'],
        [0.0, -0.88, 'C'], [0.18, 0.18, 'C'], [1.05, 0.0, 'C'],
        [2.12, 0.0, 'CO2Me'],
      ],
      bonds: [
        [0, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1],
        [2, 5, 1], [3, 5, 1], [4, 5, 1], [5, 6, 1],
      ],
    },
  };

  function mountRotationPlayground() {
    const canvas = document.getElementById('rot-canvas');
    const molSelect = document.getElementById('rot-mol');
    const angleInput = document.getElementById('rot-angle');
    const angleLabel = document.getElementById('rot-angle-val');
    const cropCheck = document.getElementById('rot-ink-crop');
    const errOut = document.getElementById('rot-feat-err');
    const armCOut = document.getElementById('rot-arm-c-smi');
    const armAOut = document.getElementById('rot-arm-a-smi');
    const armAStatus = document.getElementById('rot-arm-a-badge');
    const barsHost = document.getElementById('rot-c8-bars');
    if (!canvas || !molSelect || !angleInput) return;

    const ctx = canvas.getContext('2d');

    function draw() {
      const mol = MOLS[molSelect.value] || MOLS.aspirin;
      const deg = Number(angleInput.value) || 0;
      const rad = (deg * Math.PI) / 180;
      const inkCrop = cropCheck ? cropCheck.checked : true;
      if (angleLabel) angleLabel.textContent = `${deg}°`;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth || 340;
      const h = canvas.clientHeight || 280;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const bg = cssVar('--surface-alt', '#f8fafc');
      const ink = cssVar('--text', '#0f172a');
      const muted = cssVar('--muted', '#64748b');
      const line = cssVar('--line', '#cbd5e1');
      const teal = cssVar('--accent', '#0d9488');

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(w, h) * 0.41;

      // Inscribed circle guide (rotation never clips content)
      ctx.save();
      ctx.strokeStyle = line;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      const scale = (inkCrop ? 0.27 : 0.21) * radius;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rad);

      if (inkCrop) {
        ctx.strokeStyle = teal;
        ctx.globalAlpha = 0.28;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(-3.55 * scale, -2.95 * scale, 7.1 * scale, 5.9 * scale);
        ctx.globalAlpha = 1;
      }

      // Draw bonds
      ctx.strokeStyle = ink;
      ctx.fillStyle = ink;
      ctx.lineCap = 'round';
      for (const [i, j, order] of mol.bonds) {
        const [x1, y1] = mol.atoms[i];
        const [x2, y2] = mol.atoms[j];
        const ax = x1 * scale;
        const ay = y1 * scale;
        const bx = x2 * scale;
        const by = y2 * scale;
        if (order === 'wedge') {
          const dx = bx - ax;
          const dy = by - ay;
          const len = Math.hypot(dx, dy) || 1;
          const nx = (-dy / len) * 4.5;
          const ny = (dx / len) * 4.5;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx + nx, by + ny);
          ctx.lineTo(bx - nx, by - ny);
          ctx.closePath();
          ctx.fill();
        } else if (order === 2) {
          const dx = bx - ax;
          const dy = by - ay;
          const len = Math.hypot(dx, dy) || 1;
          const nx = (-dy / len) * 2.4;
          const ny = (dx / len) * 2.4;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.moveTo(ax + nx, ay + ny);
          ctx.lineTo(bx + nx, by + ny);
          ctx.moveTo(ax - nx, ay - ny);
          ctx.lineTo(bx - nx, by - ny);
          ctx.stroke();
        } else {
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }
      }

      // Draw heteroatom / group labels upright
      ctx.font = '600 11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const [x, y, sym] of mol.atoms) {
        if (sym === 'C') continue;
        const px = x * scale;
        const py = y * scale;
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(-rad);
        const tw = ctx.measureText(sym).width + 6;
        ctx.fillStyle = bg;
        ctx.fillRect(-tw / 2, -8, tw, 16);
        ctx.fillStyle = sym.includes('O') ? '#dc2626' : teal;
        ctx.fillText(sym, 0, 0.5);
        ctx.restore();
      }
      ctx.restore();

      ctx.fillStyle = muted;
      ctx.font = '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(`384px inscribed canvas${inkCrop ? ' · --ink-crop on' : ''}`, 12, h - 10);

      // Update C8 cyclic-shift bars and feature error (defined on C8 45° multiples, matching RotationLab)
      const step45 = deg / 45;
      let errStr = '— (defined on 45° C8 grid)';
      if (deg % 360 === 0) errStr = '3.1 × 10⁻¹⁶ (exact)';
      else if (deg % 90 === 0) errStr = '8.1 × 10⁻⁷ (exact C4 grid)';
      else if (deg % 45 === 0) errStr = '2.4 × 10⁻³ (C8 steerable grid)';
      if (errOut) errOut.textContent = errStr;

      if (armCOut) armCOut.textContent = mol.smiles;
      const offUpright = Math.min(deg, 360 - deg) > 32;
      if (armAOut) armAOut.textContent = offUpright ? mol.broken90 : mol.smiles;
      if (armAStatus) {
        armAStatus.textContent = offUpright ? 'stereo/ring drift' : 'exact at 0°';
        armAStatus.className = offUpright ? 'pill-badge warn' : 'pill-badge ok';
      }

      if (barsHost) {
        const shift = ((Math.round(step45) % 8) + 8) % 8;
        barsHost.replaceChildren();
        for (let k = 0; k < 8; k++) {
          const val = mol.baseC8[(k - shift + 8) % 8];
          const col = document.createElement('div');
          col.className = 'c8-bar-col';
          const fill = document.createElement('div');
          fill.className = 'c8-bar-fill';
          fill.style.height = `${Math.round(val * 100)}%`;
          const lbl = document.createElement('span');
          lbl.textContent = `${k * 45}°`;
          col.append(fill, lbl);
          barsHost.append(col);
        }
      }
    }

    document.querySelectorAll('[data-rot-preset]').forEach(btn => {
      btn.addEventListener('click', () => {
        angleInput.value = btn.dataset.rotPreset;
        draw();
      });
    });
    molSelect.addEventListener('change', draw);
    angleInput.addEventListener('input', draw);
    if (cropCheck) cropCheck.addEventListener('change', draw);
    window.addEventListener('ouroboros-theme-change', draw);
    if ('ResizeObserver' in window) new ResizeObserver(() => requestAnimationFrame(draw)).observe(canvas);
    draw();
  }

  /* =========================================================================
   * 2. CONFIDENCE GATE, ALIAS EXPANSION & MOLECULE_CONTEXT/V1 SIMULATOR
   * ========================================================================= */
  const SAMPLE_READS = [
    {
      id: 'UOB-0142',
      subset: 'UOB',
      smiles: 'CC(=O)Nc1ccc(O)cc1',
      expanded: 'CC(=O)Nc1ccc(O)cc1',
      formula: 'C8H9NO2',
      margin: 0.74,
      disagree: 0.0,
      hasAlias: false,
      unassignedStereo: false,
      valid: true,
      correct: true,
    },
    {
      id: 'ACS-0038',
      subset: 'ACS',
      smiles: '[Boc]N1CCC(CC1)c2ccccc2',
      expanded: 'CC(C)(C)OC(=O)N1CCC(CC1)c2ccccc2',
      formula: 'C16H23NO2',
      margin: 0.46,
      disagree: 0.0,
      hasAlias: true,
      genericR: false,
      unassignedStereo: false,
      valid: true,
      correct: true,
    },
    {
      id: 'USPTO-0219',
      subset: 'USPTO',
      smiles: 'C[C@H](N)C(=O)Nc1ccc(F)cc1',
      expanded: 'C[C@H](N)C(=O)Nc1ccc(F)cc1',
      formula: 'C9H11FN2O',
      margin: 0.34,
      disagree: 0.25,
      hasAlias: false,
      unassignedStereo: false,
      valid: true,
      correct: true,
    },
    {
      id: 'ACS-0071',
      subset: 'ACS',
      smiles: '[OMe]c1ccc(cc1)C(=O)O',
      expanded: 'COc1ccc(C(=O)O)cc1',
      formula: 'C8H8O3',
      margin: 0.52,
      disagree: 0.0,
      hasAlias: true,
      genericR: false,
      unassignedStereo: false,
      valid: true,
      correct: true,
    },
    {
      id: 'ACS-0089',
      subset: 'ACS',
      smiles: '[R1]c1nc2ccccc2s1',
      expanded: '[R1]c1nc2ccccc2s1',
      formula: 'C7H4NS*',
      margin: 0.41,
      disagree: 0.0,
      hasAlias: true,
      genericR: true,
      unassignedStereo: false,
      valid: true,
      correct: true,
    },
    {
      id: 'USPTO-0404',
      subset: 'USPTO',
      smiles: 'CC(O)C(=O)Oc1ccccc1',
      expanded: 'CC(O)C(=O)Oc1ccccc1',
      formula: 'C9H10O3',
      margin: 0.29,
      disagree: 0.25,
      hasAlias: false,
      unassignedStereo: true,
      valid: true,
      correct: false,
    },
    {
      id: 'USPTO-0512',
      subset: 'USPTO',
      smiles: 'Cc1cc(F)ccc1-c1nc(N[C@H](C)CO)nc2c1ccc(=O)n2C',
      expanded: 'Cc1cc(F)ccc1-c1nc(N[C@H](C)CO)nc2c1ccc(=O)n2C',
      formula: 'C18H19FN4O2',
      margin: 0.08,
      disagree: 0.5,
      hasAlias: false,
      unassignedStereo: false,
      valid: true,
      correct: false,
    },
    {
      id: 'HD-0019',
      subset: 'DECIMER_HDM',
      smiles: 'C1=CC(=O)C(Cl)=',
      expanded: '',
      formula: '—',
      margin: 0.0,
      disagree: 1.0,
      hasAlias: false,
      unassignedStereo: false,
      valid: false,
      correct: false,
    },
  ];

  function mountGateSimulator() {
    const marginIn = document.getElementById('gate-margin');
    const marginVal = document.getElementById('gate-margin-val');
    const disIn = document.getElementById('gate-disagree');
    const disVal = document.getElementById('gate-disagree-val');
    const aliasCheck = document.getElementById('gate-expand-aliases');
    const tbody = document.getElementById('gate-table-body');
    const promptPre = document.getElementById('gate-prompt-preview');
    const statsOut = document.getElementById('gate-stats-summary');
    if (!marginIn || !disIn || !tbody || !promptPre) return;

    function update() {
      const minMargin = Number(marginIn.value);
      const maxDis = Number(disIn.value);
      const expand = aliasCheck ? aliasCheck.checked : true;
      if (marginVal) marginVal.textContent = minMargin.toFixed(2);
      if (disVal) disVal.textContent = maxDis.toFixed(2);

      tbody.replaceChildren();
      const promptLines = ['Molecules read from figures (OCSR). Trust only status=ok.'];
      let nOk = 0;
      let nOkCorrect = 0;
      let nReview = 0;

      for (const row of SAMPLE_READS) {
        let status = 'ok';
        const reasons = [];
        const smiShown = expand && row.expanded ? row.expanded : row.smiles;

        if (!row.valid) {
          status = 'invalid';
          reasons.push('unparsable');
        } else {
          if (row.margin < minMargin) reasons.push('low_margin');
          if (row.disagree > maxDis) reasons.push('source_disagreement');
          if (reasons.length) {
            status = 'rejected';
          } else {
            if (row.genericR || (row.hasAlias && !expand)) reasons.push('pseudo_atoms');
            if (row.unassignedStereo) reasons.push('unassigned_stereo');
            if (reasons.length) status = 'review';
          }
        }

        if (status === 'ok') {
          nOk++;
          if (row.correct) nOkCorrect++;
          promptLines.push(`- ${row.id}: ok | SMILES ${smiShown} | formula ${row.formula}`);
        } else if (status === 'review') {
          nReview++;
          promptLines.push(
            `- ${row.id}: review (${reasons.join(', ')}) | ask a human (confirm with a human before use)`
          );
        } else {
          promptLines.push(`- ${row.id}: ${status} (${reasons.join(', ')}) | structure unknown`);
        }

        const tr = document.createElement('tr');
        const badgeCls =
          status === 'ok' ? 'ok' : status === 'review' ? 'warn' : 'bad';
        tr.innerHTML =
          `<td class="mono">${row.id}</td>` +
          `<td class="mono smi-cell">${row.valid ? smiShown : '<em>invalid SMILES</em>'}</td>` +
          `<td>${row.margin.toFixed(2)}</td>` +
          `<td>${row.disagree.toFixed(2)}</td>` +
          `<td><span class="pill-badge ${badgeCls}">${status}</span></td>`;
        tbody.append(tr);
      }

      promptPre.textContent = promptLines.join('\n');
      if (statsOut) {
        const prec = nOk > 0 ? `${Math.round((100 * nOkCorrect) / nOk)}%` : '—';
        statsOut.textContent =
          `status=ok: ${nOk}/${SAMPLE_READS.length} (${Math.round((100 * nOk) / SAMPLE_READS.length)}% sent to reasoner, ${prec} exact) · review: ${nReview}`;
      }
    }

    marginIn.addEventListener('input', update);
    disIn.addEventListener('input', update);
    if (aliasCheck) aliasCheck.addEventListener('change', update);
    update();
  }

  /* =========================================================================
   * 3. 96-WELL PLATE SYMMETRY FINDER (OUROBOROS.LABVISION)
   * ========================================================================= */
  const RINGS = [6, 12, 18, 24, 36]; // sum = 96

  function wellLabel(idx1, kind, order) {
    const k = idx1 - 1;
    if (kind === 'circ') {
      let rem = k;
      for (let r = 0; r < RINGS.length; r++) {
        if (rem < RINGS[r]) {
          return `R${r}S${String(rem).padStart(2, '0')}`;
        }
        rem -= RINGS[r];
      }
      return `R4S35`;
    }
    if (order === 'COL_MAJOR') {
      const col = Math.floor(k / 8) + 1;
      const row = k % 8;
      return `${String.fromCharCode(65 + row)}${col}`;
    }
    const row = Math.floor(k / 12);
    const col = (k % 12) + 1;
    return `${String.fromCharCode(65 + row)}${col}`;
  }

  function mountWellFinder() {
    const canvas = document.getElementById('well-canvas');
    const kindSel = document.getElementById('well-kind');
    const orderSel = document.getElementById('well-order');
    const wellIn = document.getElementById('well-index');
    const wellVal = document.getElementById('well-index-val');
    const rotIn = document.getElementById('well-rot');
    const rotVal = document.getElementById('well-rot-val');
    const noiseIn = document.getElementById('well-noise');
    const noiseVal = document.getElementById('well-noise-val');
    const readout = document.getElementById('well-readout');
    if (!canvas || !kindSel || !orderSel || !wellIn || !rotIn) return;

    const ctx = canvas.getContext('2d');
    let hitPoints = [];

    function syncOrderOptions() {
      const isCirc = kindSel.value === 'circ';
      for (const opt of orderSel.options) {
        if (opt.value === 'RING_MAJOR') opt.disabled = !isCirc;
        else opt.disabled = isCirc;
      }
      if (isCirc && orderSel.value !== 'RING_MAJOR') orderSel.value = 'RING_MAJOR';
      if (!isCirc && orderSel.value === 'RING_MAJOR') orderSel.value = 'ROW_MAJOR';
    }

    function draw() {
      syncOrderOptions();
      const kind = kindSel.value;
      const order = orderSel.value;
      const target = Math.max(1, Math.min(96, Number(wellIn.value) || 65));
      const deg = Number(rotIn.value) || 0;
      const noise = noiseIn ? Number(noiseIn.value) || 0 : 0;
      const rad = (deg * Math.PI) / 180;

      if (wellVal) wellVal.textContent = `#${target}`;
      if (rotVal) rotVal.textContent = `${deg}°`;
      if (noiseVal) noiseVal.textContent = `${noise.toFixed(1)} px`;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth || 360;
      const h = canvas.clientHeight || 280;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const bg = cssVar('--surface-alt', '#f8fafc');
      const card = cssVar('--surface', '#ffffff');
      const ink = cssVar('--text', '#0f172a');
      const muted = cssVar('--muted', '#64748b');
      const line = cssVar('--line', '#cbd5e1');
      const teal = cssVar('--accent', '#0d9488');
      const amber = '#d97706';

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const marginSigma = Math.max(0.4, 9.4 - noise * 3.6);
      const flagged = marginSigma < 3.0;
      hitPoints = [];

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rad);

      if (kind === 'rect') {
        const pw = Math.min(w, h) * 0.78;
        const ph = pw * 0.68;
        const cut = 14;
        // Plate body with A1 top-left chamfer
        ctx.fillStyle = card;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(-pw / 2 + cut, -ph / 2);
        ctx.lineTo(pw / 2, -ph / 2);
        ctx.lineTo(pw / 2, ph / 2);
        ctx.lineTo(-pw / 2, ph / 2);
        ctx.lineTo(-pw / 2, -ph / 2 + cut);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // A1 orientation fiducial dot
        ctx.fillStyle = teal;
        ctx.beginPath();
        ctx.arc(-pw / 2 + 13, -ph / 2 + 13, 3.3, 0, Math.PI * 2);
        ctx.fill();

        const gx0 = -pw * 0.41;
        const gy0 = -ph * 0.37;
        const dx = (pw * 0.82) / 11;
        const dy = (ph * 0.74) / 7;

        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 12; c++) {
            const idx1 = order === 'COL_MAJOR' ? c * 8 + r + 1 : r * 12 + c + 1;
            const wx = gx0 + c * dx;
            const wy = gy0 + r * dy;
            const isTarget = idx1 === target;

            ctx.beginPath();
            ctx.arc(wx, wy, isTarget ? 6.8 : 4.7, 0, Math.PI * 2);
            ctx.fillStyle = isTarget ? (flagged ? amber : teal) : bg;
            ctx.fill();
            ctx.lineWidth = isTarget ? 2.2 : 1.1;
            ctx.strokeStyle = isTarget ? ink : line;
            ctx.stroke();

            // Record screen-space coordinates for click picking
            const sx = cx + wx * Math.cos(rad) - wy * Math.sin(rad);
            const sy = cy + wx * Math.sin(rad) + wy * Math.cos(rad);
            hitPoints.push({ idx1, sx, sy });
          }
        }
      } else {
        const pr = Math.min(w, h) * 0.41;
        ctx.fillStyle = card;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(0, 0, pr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Rim notch at top (-y, breaks 60° rotational symmetry)
        ctx.fillStyle = teal;
        ctx.beginPath();
        ctx.moveTo(0, -pr - 2);
        ctx.lineTo(-5.5, -pr + 9);
        ctx.lineTo(5.5, -pr + 9);
        ctx.closePath();
        ctx.fill();

        // Chiral dot at notch + 20° clockwise on rim (rho = 0.94)
        const dotAng = (20 * Math.PI) / 180;
        ctx.fillStyle = '#4f46e5';
        ctx.beginPath();
        ctx.arc(pr * 0.92 * Math.sin(dotAng), -pr * 0.92 * Math.cos(dotAng), 3.1, 0, Math.PI * 2);
        ctx.fill();

        let running = 1;
        for (let ri = 0; ri < RINGS.length; ri++) {
          const slots = RINGS[ri];
          const rr = pr * (0.25 + ri * 0.15);
          for (let s = 0; s < slots; s++) {
            const a = (2 * Math.PI * s) / slots;
            const wx = rr * Math.sin(a);
            const wy = -rr * Math.cos(a);
            const idx1 = running++;
            const isTarget = idx1 === target;

            ctx.beginPath();
            ctx.arc(wx, wy, isTarget ? 6.2 : 3.9, 0, Math.PI * 2);
            ctx.fillStyle = isTarget ? (flagged ? amber : teal) : bg;
            ctx.fill();
            ctx.lineWidth = isTarget ? 2.0 : 1.0;
            ctx.strokeStyle = isTarget ? ink : line;
            ctx.stroke();

            const sx = cx + wx * Math.cos(rad) - wy * Math.sin(rad);
            const sy = cy + wx * Math.sin(rad) + wy * Math.cos(rad);
            hitPoints.push({ idx1, sx, sy });
          }
        }
      }
      ctx.restore();

      ctx.fillStyle = muted;
      ctx.font = '500 11px Inter, system-ui, sans-serif';
      ctx.fillText('Click any well on the plate to inspect it', 12, h - 10);

      if (readout) {
        const code = wellLabel(target, kind, order);
        const lat = kind === 'rect' ? '2.6 ms' : '9.0 ms';
        const margin = `${marginSigma.toFixed(1)} σ`;
        readout.innerHTML =
          `<strong>Well #${target} → <code>${code}</code></strong> (${order}) · ` +
          `symmetry margin <code>${margin}</code> (min 3.0 σ) · median CPU <code>${lat}</code> · ` +
          (flagged
            ? `<span class="pill-badge warn">ambiguous (flagged, never guessed)</span>`
            : `<span class="pill-badge ok">decoded cleanly</span>`);
      }
    }

    canvas.addEventListener('click', ev => {
      const rect = canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      let best = null;
      let bestD = 22;
      for (const pt of hitPoints) {
        const d = Math.hypot(pt.sx - x, pt.sy - y);
        if (d < bestD) {
          bestD = d;
          best = pt;
        }
      }
      if (best) {
        wellIn.value = String(best.idx1);
        draw();
      }
    });

    kindSel.addEventListener('change', draw);
    orderSel.addEventListener('change', draw);
    wellIn.addEventListener('input', draw);
    rotIn.addEventListener('input', draw);
    if (noiseIn) noiseIn.addEventListener('input', draw);
    window.addEventListener('ouroboros-theme-change', draw);
    if ('ResizeObserver' in window) new ResizeObserver(() => requestAnimationFrame(draw)).observe(canvas);
    draw();
  }

  function initAll() {
    mountRotationPlayground();
    mountGateSimulator();
    mountWellFinder();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll, { once: true });
  } else {
    initAll();
  }
})();

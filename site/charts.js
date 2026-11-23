/* Theme-aware interactive SVG benchmark charts for Ouroboros. */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const scriptURL = document.currentScript ? document.currentScript.src : document.baseURI;
  const dataURL = new URL('assets/benchmark-data.json', scriptURL);

  function palette() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    return dark
      ? {
          bg: '#111923',
          plot: '#0d141d',
          text: '#edf2f7',
          muted: '#9fb0c2',
          grid: '#22303e',
          axis: '#6d8094',
          whisker: '#edf2f7',
          cArmC: '#2dd4bf',
          cArmB: '#60a5fa',
          cArmA: '#fb923c',
        }
      : {
          bg: '#ffffff',
          plot: '#f8fafc',
          text: '#0f172a',
          muted: '#475569',
          grid: '#e2e8f0',
          axis: '#94a3b8',
          whisker: '#334155',
          cArmC: '#0d9488',
          cArmB: '#2563eb',
          cArmA: '#ea580c',
        };
  }

  function el(name, attrs = {}, text) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function mountCoreExplorer(data) {
    const host = document.getElementById('core-chart');
    const select = document.getElementById('core-view-select');
    const ciInput = document.getElementById('core-show-ci');
    const ciWrap = document.getElementById('core-ci-wrap');
    const legend = document.getElementById('core-chart-legend');
    const caption = document.getElementById('core-chart-caption');
    const tip = document.getElementById('core-chart-tooltip');
    if (!host || !select || !legend || !caption) return;

    function renderLegend(items) {
      legend.replaceChildren();
      for (const item of items) {
        const span = document.createElement('span');
        span.className = 'legend-item';
        const sw = el('svg', { viewBox: '0 0 38 12', width: 38, height: 12, 'aria-hidden': 'true' });
        sw.append(
          el('line', {
            x1: 2,
            y1: 6,
            x2: 36,
            y2: 6,
            stroke: item.color,
            'stroke-width': item.width || 3.5,
            'stroke-linecap': 'round',
            'stroke-dasharray': item.dash && item.dash.length ? item.dash.join(' ') : 'none',
          })
        );
        const txt = document.createElement('span');
        txt.textContent = item.name;
        span.append(sw, txt);
        legend.append(span);
      }
    }

    function modelColor(id, fallback, pal) {
      if (id === 'C') return pal.cArmC;
      if (id === 'B') return pal.cArmB;
      if (id === 'A') return pal.cArmA;
      return fallback;
    }

    function renderRotationSweep(metricKey, yLabel, yMax, yMin, desc, sweepOverride) {
      const pal = palette();
      const showCI = metricKey === 'exact' && ciInput && ciInput.checked;
      if (ciWrap) ciWrap.style.display = metricKey === 'exact' ? 'inline-flex' : 'none';
      caption.textContent = desc;

      const sweep = sweepOverride || data.rotation_sweep;
      const maxAng = sweep.angles[sweep.angles.length - 1] || 345;
      const models = sweep.models.map(m => ({ ...m, color: modelColor(m.id, m.color, pal) }));
      renderLegend(models);

      const width = host.getBoundingClientRect().width || 1000;
      const fs = width < 560 ? 20 : 15;
      const plot = { x: 84, y: 22, w: 888, h: 374 };
      const sx = ang => plot.x + (ang / maxAng) * plot.w;
      const sy = val => plot.y + plot.h - ((val - yMin) / (yMax - yMin)) * plot.h;

      const svg = el('svg', { viewBox: '0 0 1000 468', role: 'img', 'aria-label': desc });
      svg.append(el('rect', { width: 1000, height: 468, rx: 10, fill: pal.bg }));
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, rx: 6, fill: pal.plot }));

      const yStep = (yMax - yMin) <= 60 ? 10 : 20;
      for (let yv = yMin; yv <= yMax; yv += yStep) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: pal.grid, 'stroke-width': 1 }));
        svg.append(el('text', { x: plot.x - 12, y: yy, fill: pal.muted, 'font-size': fs, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`));
      }
      const xTicks = maxAng === 315 ? [0, 45, 90, 135, 180, 225, 270, 315] : [0, 45, 90, 135, 180, 225, 270, 315, 345];
      for (const xv of xTicks) {
        const xx = sx(xv);
        svg.append(el('line', { x1: xx, y1: plot.y + plot.h, x2: xx, y2: plot.y + plot.h + 5, stroke: pal.axis }));
        svg.append(el('text', { x: xx, y: plot.y + plot.h + 24, fill: pal.muted, 'font-size': fs, 'text-anchor': 'middle' }, `${xv}°`));
      }

      const ordered = [...models].reverse();
      if (showCI) {
        for (const m of ordered) {
          const upper = sweep.angles.map((a, i) => `${i === 0 ? 'M' : 'L'}${sx(a).toFixed(1)},${sy(m.exact_ci[i][1]).toFixed(1)}`).join(' ');
          const lower = [...sweep.angles]
            .map((a, i) => ({ a, v: m.exact_ci[i][0] }))
            .reverse()
            .map(p => `L${sx(p.a).toFixed(1)},${sy(p.v).toFixed(1)}`)
            .join(' ');
          svg.append(el('path', { d: `${upper} ${lower} Z`, fill: m.color, 'fill-opacity': m.id === 'C' ? 0.16 : 0.09 }));
        }
      }

      for (const m of ordered) {
        const vals = m[metricKey];
        const d = sweep.angles.map((a, i) => `${i === 0 ? 'M' : 'L'}${sx(a).toFixed(1)},${sy(vals[i]).toFixed(1)}`).join(' ');
        svg.append(
          el('path', {
            d,
            fill: 'none',
            stroke: m.color,
            'stroke-width': m.width,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round',
            'stroke-dasharray': m.dash.length ? m.dash.join(' ') : 'none',
          })
        );
        for (let i = 0; i < sweep.angles.length; i++) {
          svg.append(el('circle', { cx: sx(sweep.angles[i]), cy: sy(vals[i]), r: m.id === 'C' ? 3.3 : 2.5, fill: m.color }));
        }
      }

      svg.append(el('path', { d: `M${plot.x},${plot.y}V${plot.y + plot.h}H${plot.x + plot.w}`, fill: 'none', stroke: pal.axis, 'stroke-width': 1.2 }));
      svg.append(el('text', { x: plot.x + plot.w / 2, y: 454, fill: pal.muted, 'font-size': fs, 'text-anchor': 'middle' }, 'Test image rotation angle (degrees)'));
      svg.append(el('text', { transform: `translate(22 ${plot.y + plot.h / 2}) rotate(-90)`, fill: pal.muted, 'font-size': fs, 'text-anchor': 'middle' }, yLabel));

      const vLine = el('line', { x1: plot.x, y1: plot.y, x2: plot.x, y2: plot.y + plot.h, stroke: pal.cArmC, 'stroke-width': 1.2, 'stroke-dasharray': '4 4', opacity: 0 });
      svg.append(vLine);

      const overlay = el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: 'transparent' });
      overlay.addEventListener('mousemove', ev => {
        if (!tip) return;
        const box = svg.getBoundingClientRect();
        const svgX = ((ev.clientX - box.left) / box.width) * 1000;
        const nPts = Math.max(1, sweep.angles.length - 1);
        const idx = Math.max(0, Math.min(nPts, Math.round(((svgX - plot.x) / plot.w) * nPts)));
        const ang = sweep.angles[idx];
        const xPos = sx(ang);
        vLine.setAttribute('x1', String(xPos));
        vLine.setAttribute('x2', String(xPos));
        vLine.setAttribute('opacity', '0.75');
        tip.innerHTML =
          `<strong>Angle ${ang}°</strong><br>` +
          models.map(m => `<span style="color:${m.color}">●</span> ${m.id}: <strong>${m[metricKey][idx].toFixed(1)}%</strong>`).join('<br>');
        tip.hidden = false;
        const relLeft = (xPos / 1000) * box.width;
        tip.style.left = `${Math.min(box.width - 180, Math.max(8, relLeft + 14))}px`;
        tip.style.top = '24px';
      });
      overlay.addEventListener('mouseleave', () => {
        vLine.setAttribute('opacity', '0');
        if (tip) tip.hidden = true;
      });
      svg.append(overlay);
      host.replaceChildren(svg);
    }

    function renderFrozenVsDev() {
      const pal = palette();
      const showCI = ciInput ? ciInput.checked : true;
      if (ciWrap) ciWrap.style.display = 'inline-flex';
      if (tip) tip.hidden = true;

      const cmp = data.frozen_vs_dev;
      if (!cmp) return renderRealSubsets();
      const series = [
        { key: 'frozen', name: 'frozen-test-v2 (n = 300, untouched single-shot)', color: pal.cArmC },
        { key: 'dev', name: 'subset100 (n = 298, dev-test)', color: pal.cArmB },
        { key: 'handdrawn', name: 'DECIMER Hand-drawn (n = 200, zero HD training)', color: pal.cArmA },
      ];
      renderLegend(series.map(s => ({ name: s.name, color: s.color, width: 5, dash: [] })));
      caption.textContent =
        'Comparison of the A + C ensemble (beam 5 + pad TTA, locked strictly on PMC val at 44.6%) across the untouched 300-image frozen-test-v2 holdout, the 298-image subset100 dev-test set, and the 200-image DECIMER Hand-drawn benchmark.';

      const svg = el('svg', { viewBox: '0 0 1000 450', role: 'img', 'aria-label': 'Frozen holdout vs dev-test exact match' });
      svg.append(el('rect', { width: 1000, height: 450, rx: 10, fill: pal.bg }));
      const plot = { x: 210, y: 22, w: 746, h: 374 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, rx: 6, fill: pal.plot }));

      const sx = val => plot.x + (val / 100) * plot.w;
      for (const xv of [0, 20, 40, 60, 80, 100]) {
        const xx = sx(xv);
        svg.append(el('line', { x1: xx, y1: plot.y, x2: xx, y2: plot.y + plot.h, stroke: pal.grid }));
        svg.append(el('text', { x: xx, y: plot.y + plot.h + 24, fill: pal.muted, 'font-size': 14.5, 'text-anchor': 'middle' }, `${xv}%`));
      }

      const groups = cmp.groups;
      const groupH = plot.h / groups.length;
      groups.forEach((g, gi) => {
        const gy = plot.y + gi * groupH;
        if (gi > 0) svg.append(el('line', { x1: plot.x, y1: gy, x2: plot.x + plot.w, y2: gy, stroke: pal.grid }));
        svg.append(el('text', { x: plot.x - 14, y: gy + groupH / 2, fill: pal.text, 'font-size': 14.5, 'font-weight': 600, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, g.label));

        const present = series.filter(s => g[s.key] !== undefined && g[s.key] !== null);
        const barH = 14;
        const totalH = present.length * barH + (present.length - 1) * 5;
        const y0 = gy + (groupH - totalH) / 2;
        present.forEach((s, si) => {
          const item = g[s.key];
          const by = y0 + si * (barH + 5);
          const bw = Math.max(2, sx(item.exact) - plot.x);
          svg.append(el('rect', { x: plot.x, y: by, width: bw, height: barH, rx: 3, fill: s.color }));
          if (showCI && item.ci) {
            const xLo = sx(item.ci[0]);
            const xHi = sx(item.ci[1]);
            const cy = by + barH / 2;
            svg.append(el('line', { x1: xLo, y1: cy, x2: xHi, y2: cy, stroke: pal.whisker, 'stroke-width': 1.4 }));
          }
          const lx = (showCI && item.ci ? sx(item.ci[1]) : sx(item.exact)) + 8;
          svg.append(el('text', { x: lx, y: by + barH - 2, fill: pal.muted, 'font-size': 12.5, 'font-weight': 600 }, `${item.exact.toFixed(1)}%`));
        });
      });

      svg.append(el('text', { x: plot.x + plot.w / 2, y: 438, fill: pal.muted, 'font-size': 14.5, 'text-anchor': 'middle' }, 'Exact match with full stereochemistry (%)'));
      host.replaceChildren(svg);
    }

    function renderRealSubsets() {
      const pal = palette();
      const showCI = ciInput ? ciInput.checked : true;
      if (ciWrap) ciWrap.style.display = 'inline-flex';
      if (tip) tip.hidden = true;

      const rb = data.real_benchmarks;
      const paletteCycle = [pal.cArmC, '#0891b2', pal.cArmB, '#6366f1', pal.cArmA];
      const models = rb.models.map((m, idx) => ({ ...m, color: paletteCycle[idx % paletteCycle.length] }));
      renderLegend(models.map(m => ({ name: m.name, color: m.color, width: 4, dash: [] })));
      caption.textContent =
        'Upright real document dev-test set (298 images: ACS 100, UOB 100, USPTO 98). Exact match with full stereochemistry; whiskers show 1,000-resample bootstrap 95% CIs.';

      const svg = el('svg', { viewBox: '0 0 1000 490', role: 'img', 'aria-label': 'Real document dev-test OCSR accuracy' });
      svg.append(el('rect', { width: 1000, height: 490, rx: 10, fill: pal.bg }));
      const plot = { x: 214, y: 20, w: 742, h: 410 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, rx: 6, fill: pal.plot }));

      const sx = val => plot.x + (val / 100) * plot.w;
      for (const xv of [0, 20, 40, 60, 80, 100]) {
        const xx = sx(xv);
        svg.append(el('line', { x1: xx, y1: plot.y, x2: xx, y2: plot.y + plot.h, stroke: pal.grid }));
        svg.append(el('text', { x: xx, y: plot.y + plot.h + 24, fill: pal.muted, 'font-size': 14.5, 'text-anchor': 'middle' }, `${xv}%`));
      }

      const groupH = plot.h / rb.subsets.length;
      const barH = 13.5;
      rb.subsets.forEach((sub, gi) => {
        const gy = plot.y + gi * groupH;
        if (gi > 0) svg.append(el('line', { x1: plot.x, y1: gy, x2: plot.x + plot.w, y2: gy, stroke: pal.grid }));
        svg.append(el('text', { x: plot.x - 14, y: gy + groupH / 2, fill: pal.text, 'font-size': 14.5, 'font-weight': 600, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, rb.subset_labels[sub]));
        models.forEach((m, mi) => {
          const by = gy + 11 + mi * (barH + 4);
          const val = m.scores[sub];
          const ci = m.cis[sub];
          const bw = Math.max(2, sx(val) - plot.x);
          svg.append(el('rect', { x: plot.x, y: by, width: bw, height: barH, rx: 3, fill: m.color }));
          if (showCI && ci) {
            const xLo = sx(ci[0]);
            const xHi = sx(ci[1]);
            const cy = by + barH / 2;
            svg.append(el('line', { x1: xLo, y1: cy, x2: xHi, y2: cy, stroke: pal.whisker, 'stroke-width': 1.3 }));
          }
          const lx = (showCI && ci ? sx(ci[1]) : sx(val)) + 7;
          svg.append(el('text', { x: lx, y: by + barH - 2, fill: pal.muted, 'font-size': 12 }, `${val.toFixed(1)}%`));
        });
      });

      svg.append(el('text', { x: plot.x + plot.w / 2, y: 476, fill: pal.muted, 'font-size': 14.5, 'text-anchor': 'middle' }, 'Exact match with stereochemistry (%)'));
      host.replaceChildren(svg);
    }

    function render() {
      const v = select.value;
      if (v === 'rot_exact') {
        renderRotationSweep(
          'exact',
          'Exact match (%)',
          80,
          0,
          '24-angle synthetic rotation sweep (n = 2,000 molecules per angle, 15° steps). Arm C (65.8% mean) matches 360°-augmented Arm B (65.7%) with 25% fewer parameters and zero rotated training images.'
        );
      } else if (v === 'rot_real_uob' && data.rotated_real_uob) {
        renderRotationSweep(
          'exact',
          'Real UOB exact match (%)',
          60,
          0,
          'Rotated real UOB figures (n = 100 scanned structures, 45° steps). Pure-PyTorch C8 steerable C-pure holds 39.0% across 0°, 90°, 180°, and 270° (37.5% 8-angle mean, zero rotated training images), beating both plain CNN A (1.0% at 90°/270°) and 360°-augmented CNN B (33.3% on 90°/180°/270°).',
          data.rotated_real_uob
        );
      } else if (v === 'frozen_vs_dev') {
        renderFrozenVsDev();
      } else if (v === 'rot_stereo') {
        renderRotationSweep(
          'stereo',
          'Stereo-subset exact (%)',
          50,
          0,
          '24-angle stereo-subset exact match (n = 800 chiral or E/Z molecules per angle). Arm C holds ~37.5% across all orientations while plain CNN Arm A drops below 4% at 90° and 270°.'
        );
      } else if (v === 'rot_tanimoto') {
        renderRotationSweep(
          'tanimoto',
          'Mean Morgan Tanimoto (%)',
          100,
          35,
          '24-angle mean Morgan fingerprint similarity (r = 2, 2048 bits). Both Arm C and Arm B remain above 92% across every angle.'
        );
      } else {
        renderRealSubsets();
      }
    }

    select.addEventListener('change', render);
    if (ciInput) ciInput.addEventListener('change', render);
    window.addEventListener('ouroboros-theme-change', render);
    render();
    if ('ResizeObserver' in window) new ResizeObserver(() => requestAnimationFrame(render)).observe(host);
  }

  function mountP1Explorer(data) {
    const host = document.getElementById('p1-chart');
    const select = document.getElementById('p1-view-select');
    const legend = document.getElementById('p1-chart-legend');
    const caption = document.getElementById('p1-chart-caption');
    if (!host || !select || !legend || !caption) return;

    const p1 = data.p1_chemveritas;

    function setLegend(entries) {
      legend.replaceChildren();
      for (const e of entries) {
        const span = document.createElement('span');
        span.className = 'legend-item';
        const dot = document.createElement('span');
        dot.style.cssText = `width:12px;height:12px;border-radius:3px;background:${e.color};display:inline-block;`;
        const txt = document.createElement('span');
        txt.textContent = e.name;
        span.append(dot, txt);
        legend.append(span);
      }
    }

    function renderAblationLadder() {
      const pal = palette();
      setLegend([
        { name: 'T2 Stereoisomer Exact Match (n = 600, in-model)', color: pal.cArmC },
        { name: 'Veritas-RPF Reasoning Path Fidelity', color: pal.cArmB },
        { name: 'T3 Strain Audit Accuracy', color: '#6366f1' },
      ]);
      caption.textContent =
        'ChemVeritas-600 in-model ablation ladder (A0 → A6, N = 600 connectivity-disjoint scaffolds, synthetic noise-free NMR + ±1.6 ppm HRMS jitter). Adding O(3) ¹³C symmetry orbits lifts diastereomer selection from 19.7% (rank prior) to 77.8% (77.0% full loop).';

      const svg = el('svg', { viewBox: '0 0 1000 436', role: 'img', 'aria-label': 'ChemVeritas-600 ablation results' });
      svg.append(el('rect', { width: 1000, height: 436, rx: 10, fill: pal.bg }));
      const plot = { x: 76, y: 22, w: 888, h: 338 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, rx: 6, fill: pal.plot }));

      const sy = v => plot.y + plot.h - (v / 100) * plot.h;
      for (const yv of [0, 20, 40, 60, 80, 100]) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: pal.grid }));
        svg.append(el('text', { x: plot.x - 12, y: yy, fill: pal.muted, 'font-size': 14, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`));
      }

      const colW = plot.w / p1.ablations.length;
      p1.ablations.forEach((ab, i) => {
        const cx = plot.x + (i + 0.5) * colW;
        const bw = 23;
        const vals = [
          { v: ab.t2_exact, ci: ab.t2_ci, c: pal.cArmC },
          { v: ab.rpf, ci: null, c: pal.cArmB },
          { v: ab.t3_audit, ci: null, c: '#6366f1' },
        ];
        vals.forEach((bar, bi) => {
          const bx = cx - 38 + bi * (bw + 4);
          const by = sy(bar.v);
          svg.append(el('rect', { x: bx, y: by, width: bw, height: plot.y + plot.h - by, rx: 3, fill: bar.c }));
          if (bar.ci) {
            const mid = bx + bw / 2;
            svg.append(el('line', { x1: mid, y1: sy(bar.ci[1]), x2: mid, y2: sy(bar.ci[0]), stroke: pal.whisker, 'stroke-width': 1.4 }));
          }
        });
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 22, fill: pal.text, 'font-size': 13.5, 'font-weight': 700, 'text-anchor': 'middle' }, ab.id));
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 41, fill: pal.muted, 'font-size': 11.5, 'text-anchor': 'middle' }, ab.label));
      });
      host.replaceChildren(svg);
    }

    function renderClosedLoopViews() {
      const pal = palette();
      setLegend([
        { name: 'Ouroboros-AC Visual Top-1 (A0)', color: pal.cArmA },
        { name: 'Ouroboros-AC + Bayesian Rescue (A6)', color: pal.cArmC },
      ]);
      caption.textContent =
        'Closed-loop coupling of pretrained Ouroboros-AC beams (beam = 5, n = 100 hard polycyclic structures per synthetic view V0–V3) with HRMS + 3D strain + ¹³C/¹H NMR disambiguation.';

      const svg = el('svg', { viewBox: '0 0 1000 416', role: 'img', 'aria-label': 'Closed-loop neural OCSR across views V0 to V3' });
      svg.append(el('rect', { width: 1000, height: 416, rx: 10, fill: pal.bg }));
      const plot = { x: 84, y: 24, w: 876, h: 320 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, rx: 6, fill: pal.plot }));

      const sy = v => plot.y + plot.h - (v / 25) * plot.h;
      for (const yv of [0, 5, 10, 15, 20, 25]) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: pal.grid }));
        svg.append(el('text', { x: plot.x - 12, y: yy, fill: pal.muted, 'font-size': 14, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`));
      }

      const colW = plot.w / p1.closed_loop_views.length;
      p1.closed_loop_views.forEach((row, i) => {
        const cx = plot.x + (i + 0.5) * colW;
        const b1 = sy(row.visual);
        const b2 = sy(row.bayesian);
        svg.append(el('rect', { x: cx - 42, y: b1, width: 36, height: Math.max(3, plot.y + plot.h - b1), rx: 4, fill: pal.cArmA }));
        svg.append(el('rect', { x: cx + 6, y: b2, width: 36, height: plot.y + plot.h - b2, rx: 4, fill: pal.cArmC }));
        const yLo = sy(row.ci[0]);
        const yHi = sy(row.ci[1]);
        svg.append(el('line', { x1: cx + 24, y1: yHi, x2: cx + 24, y2: yLo, stroke: pal.whisker, 'stroke-width': 1.5 }));
        svg.append(el('text', { x: cx - 24, y: b1 - 8, fill: pal.cArmA, 'font-size': 12.5, 'font-weight': 700, 'text-anchor': 'middle' }, `${row.visual.toFixed(1)}%`));
        svg.append(el('text', { x: cx + 24, y: yHi - 8, fill: pal.cArmC, 'font-size': 12.5, 'font-weight': 700, 'text-anchor': 'middle' }, `${row.bayesian.toFixed(1)}%`));
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 26, fill: pal.text, 'font-size': 13.5, 'font-weight': 600, 'text-anchor': 'middle' }, row.view));
      });
      host.replaceChildren(svg);
    }

    function render() {
      if (select.value === 'views') renderClosedLoopViews();
      else renderAblationLadder();
    }

    select.addEventListener('change', render);
    window.addEventListener('ouroboros-theme-change', render);
    render();
  }

  async function init() {
    try {
      const res = await fetch(dataURL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      mountCoreExplorer(data);
      mountP1Explorer(data);
    } catch (err) {
      console.error('Ouroboros chart load error:', err);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();

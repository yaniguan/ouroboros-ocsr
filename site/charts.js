/* Dependency-free interactive SVG benchmark charts for Ouroboros. */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const scriptURL = document.currentScript ? document.currentScript.src : document.baseURI;
  const dataURL = new URL('assets/benchmark-data.json', scriptURL);

  const COLORS = {
    bg: '#0a0f15',
    plot: '#0e161f',
    text: '#edf1f7',
    muted: '#aebbc9',
    grid: '#24313f',
    axis: '#6d8094',
  };

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
        const sw = el('svg', { viewBox: '0 0 42 12', width: 42, height: 12, 'aria-hidden': 'true' });
        sw.append(
          el('line', {
            x1: 2,
            y1: 6,
            x2: 40,
            y2: 6,
            stroke: item.color,
            'stroke-width': item.width || 3,
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

    function renderRotationSweep(metricKey, yLabel, yMax, yMin, desc) {
      const showCI = metricKey === 'exact' && ciInput && ciInput.checked;
      if (ciWrap) ciWrap.style.display = metricKey === 'exact' ? 'inline-flex' : 'none';
      caption.textContent = desc;

      const sweep = data.rotation_sweep;
      renderLegend(sweep.models);

      const width = host.getBoundingClientRect().width || 1000;
      const compact = width < 560;
      const fs = compact ? 22 : 16;
      const plot = { x: 88, y: 26, w: 884, h: 386 };
      const sx = ang => plot.x + (ang / 345) * plot.w;
      const sy = val => plot.y + plot.h - ((val - yMin) / (yMax - yMin)) * plot.h;

      const svg = el('svg', {
        viewBox: '0 0 1000 490',
        role: 'img',
        'aria-label': desc,
      });
      svg.append(el('rect', { width: 1000, height: 490, fill: COLORS.bg }));
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: COLORS.plot }));

      const yStep = (yMax - yMin) <= 60 ? 10 : 20;
      for (let yv = yMin; yv <= yMax; yv += yStep) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: COLORS.grid, 'stroke-width': 1 }));
        svg.append(
          el('text', { x: plot.x - 14, y: yy, fill: COLORS.muted, 'font-size': fs, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`)
        );
      }
      for (const xv of [0, 45, 90, 135, 180, 225, 270, 315, 345]) {
        const xx = sx(xv);
        svg.append(el('line', { x1: xx, y1: plot.y + plot.h, x2: xx, y2: plot.y + plot.h + 6, stroke: COLORS.axis }));
        svg.append(
          el('text', { x: xx, y: plot.y + plot.h + 26, fill: COLORS.muted, 'font-size': fs, 'text-anchor': 'middle' }, `${xv}°`)
        );
      }

      const ordered = [...sweep.models].reverse();
      if (showCI) {
        for (const m of ordered) {
          const upper = sweep.angles.map((a, i) => `${i === 0 ? 'M' : 'L'}${sx(a).toFixed(1)},${sy(m.exact_ci[i][1]).toFixed(1)}`).join(' ');
          const lower = [...sweep.angles]
            .map((a, i) => ({ a, v: m.exact_ci[i][0] }))
            .reverse()
            .map(p => `L${sx(p.a).toFixed(1)},${sy(p.v).toFixed(1)}`)
            .join(' ');
          svg.append(
            el('path', {
              d: `${upper} ${lower} Z`,
              fill: m.color,
              'fill-opacity': m.id === 'C' ? 0.18 : 0.1,
              stroke: 'none',
            })
          );
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
          svg.append(el('circle', { cx: sx(sweep.angles[i]), cy: sy(vals[i]), r: m.id === 'C' ? 3.4 : 2.6, fill: m.color }));
        }
      }

      svg.append(
        el('path', {
          d: `M${plot.x},${plot.y}V${plot.y + plot.h}H${plot.x + plot.w}`,
          fill: 'none',
          stroke: COLORS.axis,
          'stroke-width': 1.2,
        })
      );
      svg.append(
        el('text', { x: plot.x + plot.w / 2, y: 474, fill: COLORS.muted, 'font-size': fs, 'text-anchor': 'middle' }, 'Test image rotation angle (degrees)')
      );
      svg.append(
        el(
          'text',
          {
            transform: `translate(24 ${plot.y + plot.h / 2}) rotate(-90)`,
            fill: COLORS.muted,
            'font-size': fs,
            'text-anchor': 'middle',
          },
          yLabel
        )
      );

      // Interactive hover crosshair
      const vLine = el('line', {
        x1: plot.x,
        y1: plot.y,
        x2: plot.x,
        y2: plot.y + plot.h,
        stroke: '#b5d88e',
        'stroke-width': 1.2,
        'stroke-dasharray': '4 4',
        opacity: 0,
      });
      svg.append(vLine);

      const overlay = el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: 'transparent' });
      overlay.addEventListener('mousemove', ev => {
        if (!tip) return;
        const box = svg.getBoundingClientRect();
        const svgX = ((ev.clientX - box.left) / box.width) * 1000;
        const idx = Math.max(0, Math.min(23, Math.round(((svgX - plot.x) / plot.w) * 23)));
        const ang = sweep.angles[idx];
        const xPos = sx(ang);
        vLine.setAttribute('x1', String(xPos));
        vLine.setAttribute('x2', String(xPos));
        vLine.setAttribute('opacity', '0.7');
        tip.innerHTML =
          `<strong>Angle ${ang}°</strong><br>` +
          sweep.models
            .map(m => `<span style="color:${m.color}">●</span> ${m.id}: <strong>${m[metricKey][idx].toFixed(1)}%</strong>`)
            .join('<br>');
        tip.hidden = false;
        const relLeft = ((xPos / 1000) * box.width);
        tip.style.left = `${Math.min(box.width - 175, Math.max(8, relLeft + 14))}px`;
        tip.style.top = '28px';
      });
      overlay.addEventListener('mouseleave', () => {
        vLine.setAttribute('opacity', '0');
        if (tip) tip.hidden = true;
      });
      svg.append(overlay);

      host.replaceChildren(svg);
    }

    function renderRealSubsets() {
      const showCI = ciInput ? ciInput.checked : true;
      if (ciWrap) ciWrap.style.display = 'inline-flex';
      if (tip) tip.hidden = true;

      const rb = data.real_benchmarks;
      renderLegend(rb.models.map(m => ({ name: m.name, color: m.color, width: 4, dash: [] })));
      caption.textContent =
        'Upright real document dev-test set (298 images: ACS 100, UOB 100, USPTO 98). Exact match with full stereochemistry; whiskers show 1,000-resample bootstrap 95% CIs. Sources: benchmarks/finetune-v2-oct30/ and benchmarks/real-subset100-rgroups/.';

      const svg = el('svg', {
        viewBox: '0 0 1000 500',
        role: 'img',
        'aria-label': 'Real document OCSR accuracy across ALL, UOB, ACS, and USPTO benchmarks',
      });
      svg.append(el('rect', { width: 1000, height: 500, fill: COLORS.bg }));
      const plot = { x: 214, y: 24, w: 742, h: 416 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: COLORS.plot }));

      const sx = val => plot.x + (val / 100) * plot.w;
      for (const xv of [0, 20, 40, 60, 80, 100]) {
        const xx = sx(xv);
        svg.append(el('line', { x1: xx, y1: plot.y, x2: xx, y2: plot.y + plot.h, stroke: COLORS.grid }));
        svg.append(el('text', { x: xx, y: plot.y + plot.h + 26, fill: COLORS.muted, 'font-size': 15, 'text-anchor': 'middle' }, `${xv}%`));
      }

      const groupH = plot.h / rb.subsets.length;
      const barH = 14;
      rb.subsets.forEach((sub, gi) => {
        const gy = plot.y + gi * groupH;
        if (gi > 0) {
          svg.append(el('line', { x1: plot.x, y1: gy, x2: plot.x + plot.w, y2: gy, stroke: COLORS.grid }));
        }
        svg.append(
          el(
            'text',
            { x: plot.x - 16, y: gy + groupH / 2, fill: COLORS.text, 'font-size': 15, 'font-weight': 600, 'text-anchor': 'end', 'dominant-baseline': 'middle' },
            rb.subset_labels[sub]
          )
        );
        rb.models.forEach((m, mi) => {
          const by = gy + 12 + mi * (barH + 4);
          const val = m.scores[sub];
          const ci = m.cis[sub];
          const bw = Math.max(2, sx(val) - plot.x);
          svg.append(el('rect', { x: plot.x, y: by, width: bw, height: barH, rx: 3, fill: m.color }));
          if (showCI && ci) {
            const xLo = sx(ci[0]);
            const xHi = sx(ci[1]);
            const cy = by + barH / 2;
            svg.append(el('line', { x1: xLo, y1: cy, x2: xHi, y2: cy, stroke: '#edf1f7', 'stroke-width': 1.4, opacity: 0.78 }));
            svg.append(el('line', { x1: xLo, y1: cy - 3.5, x2: xLo, y2: cy + 3.5, stroke: '#edf1f7', 'stroke-width': 1.4, opacity: 0.78 }));
            svg.append(el('line', { x1: xHi, y1: cy - 3.5, x2: xHi, y2: cy + 3.5, stroke: '#edf1f7', 'stroke-width': 1.4, opacity: 0.78 }));
          }
          const labelX = (showCI && ci ? sx(ci[1]) : sx(val)) + 8;
          svg.append(
            el('text', { x: labelX, y: by + barH - 2, fill: COLORS.muted, 'font-size': 12.5 }, `${val.toFixed(1)}%`)
          );
        });
      });

      svg.append(
        el('text', { x: plot.x + plot.w / 2, y: 486, fill: COLORS.muted, 'font-size': 15, 'text-anchor': 'middle' }, 'Exact match with stereochemistry (%)')
      );
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
          '24-angle synthetic rotation sweep (n = 2,000 fixed molecules per angle, 15° steps). Arm C (65.8% mean) matches 360°-augmented Arm B (65.7%) with 25% fewer parameters and zero rotated training images. Source: benchmarks/v0.1.0-alpha/{A,B,C}-summary.json.'
        );
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
      setLegend([
        { name: 'T2 Stereoisomer Exact Match (n = 600)', color: '#b5d88e' },
        { name: 'Veritas-RPF Reasoning Path Fidelity', color: '#91b6df' },
        { name: 'T3 Silent Strain Audit Accuracy', color: '#b9a1de' },
      ]);
      caption.textContent =
        'ChemVeritas-600 ablation ladder (A0 → A6, N = 600 connectivity-disjoint scaffolds). Adding O(3) ¹³C symmetry orbits and shift matching lifts stereoisomer recovery from 19.7% to 77.8%, while full joint inference reaches 77.0% exact, 93.3% Veritas-RPF, and 74.8% T3 strain-audit accuracy. Source: branch p1-phys-spec-ocsr-bench (tag p1-v1.0).';

      const svg = el('svg', { viewBox: '0 0 1000 450', role: 'img', 'aria-label': 'ChemVeritas-600 ablation results' });
      svg.append(el('rect', { width: 1000, height: 450, fill: COLORS.bg }));
      const plot = { x: 76, y: 26, w: 888, h: 344 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: COLORS.plot }));

      const sy = v => plot.y + plot.h - (v / 100) * plot.h;
      for (const yv of [0, 20, 40, 60, 80, 100]) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: COLORS.grid }));
        svg.append(el('text', { x: plot.x - 12, y: yy, fill: COLORS.muted, 'font-size': 14.5, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`));
      }

      const colW = plot.w / p1.ablations.length;
      p1.ablations.forEach((ab, i) => {
        const gx = plot.x + i * colW;
        const cx = gx + colW / 2;
        const bw = 24;
        const vals = [
          { v: ab.t2_exact, ci: ab.t2_ci, c: '#b5d88e' },
          { v: ab.rpf, ci: null, c: '#91b6df' },
          { v: ab.t3_audit, ci: null, c: '#b9a1de' },
        ];
        vals.forEach((bar, bi) => {
          const bx = cx - 40 + bi * (bw + 4);
          const by = sy(bar.v);
          svg.append(el('rect', { x: bx, y: by, width: bw, height: plot.y + plot.h - by, rx: 3, fill: bar.c }));
          if (bar.ci) {
            const yLo = sy(bar.ci[0]);
            const yHi = sy(bar.ci[1]);
            const mid = bx + bw / 2;
            svg.append(el('line', { x1: mid, y1: yHi, x2: mid, y2: yLo, stroke: '#edf1f7', 'stroke-width': 1.5 }));
          }
        });
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 24, fill: COLORS.text, 'font-size': 14, 'font-weight': 600, 'text-anchor': 'middle' }, ab.id));
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 44, fill: COLORS.muted, 'font-size': 11.5, 'text-anchor': 'middle' }, ab.label));
      });
      host.replaceChildren(svg);
    }

    function renderClosedLoopViews() {
      setLegend([
        { name: 'Ouroboros-AC Visual Top-1 (A0)', color: '#d97757' },
        { name: 'Ouroboros-AC + Full Bayesian Rescue (A6)', color: '#b5d88e' },
      ]);
      caption.textContent =
        'End-to-end closed-loop evaluation coupling pretrained Ouroboros-AC beams (beam = 5, n = 100 hard polycyclic structures per view) to MACE-OFF23 + HRMS + ¹³C/¹H NMR Bayesian disambiguation across four segregated depiction regimes (V0–V3).';

      const svg = el('svg', { viewBox: '0 0 1000 430', role: 'img', 'aria-label': 'Closed-loop neural OCSR across views V0 to V3' });
      svg.append(el('rect', { width: 1000, height: 430, fill: COLORS.bg }));
      const plot = { x: 84, y: 28, w: 876, h: 326 };
      svg.append(el('rect', { x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: COLORS.plot }));

      const sy = v => plot.y + plot.h - (v / 25) * plot.h;
      for (const yv of [0, 5, 10, 15, 20, 25]) {
        const yy = sy(yv);
        svg.append(el('line', { x1: plot.x, y1: yy, x2: plot.x + plot.w, y2: yy, stroke: COLORS.grid }));
        svg.append(el('text', { x: plot.x - 12, y: yy, fill: COLORS.muted, 'font-size': 14.5, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, `${yv}%`));
      }

      const colW = plot.w / p1.closed_loop_views.length;
      p1.closed_loop_views.forEach((row, i) => {
        const cx = plot.x + (i + 0.5) * colW;
        const b1 = sy(row.visual);
        const b2 = sy(row.bayesian);
        svg.append(el('rect', { x: cx - 42, y: b1, width: 36, height: Math.max(3, plot.y + plot.h - b1), rx: 4, fill: '#d97757' }));
        svg.append(el('rect', { x: cx + 6, y: b2, width: 36, height: plot.y + plot.h - b2, rx: 4, fill: '#b5d88e' }));
        const yLo = sy(row.ci[0]);
        const yHi = sy(row.ci[1]);
        svg.append(el('line', { x1: cx + 24, y1: yHi, x2: cx + 24, y2: yLo, stroke: '#edf1f7', 'stroke-width': 1.6 }));
        svg.append(el('text', { x: cx - 24, y: b1 - 8, fill: '#d97757', 'font-size': 13, 'font-weight': 600, 'text-anchor': 'middle' }, `${row.visual.toFixed(1)}%`));
        svg.append(el('text', { x: cx + 24, y: yHi - 8, fill: '#b5d88e', 'font-size': 13, 'font-weight': 700, 'text-anchor': 'middle' }, `${row.bayesian.toFixed(1)}%`));
        svg.append(el('text', { x: cx, y: plot.y + plot.h + 28, fill: COLORS.text, 'font-size': 14, 'font-weight': 600, 'text-anchor': 'middle' }, row.view));
      });
      host.replaceChildren(svg);
    }

    function render() {
      if (select.value === 'views') renderClosedLoopViews();
      else renderAblationLadder();
    }

    select.addEventListener('change', render);
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

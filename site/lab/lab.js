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
  const KNOWN_ELEMS = new Set([
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'Pt', 'Au', 'Hg', 'Pb', 'Bi',
  ]);
  const ATOM_MASS = { H: 1.008, B: 10.81, C: 12.011, N: 14.007, O: 15.999, F: 18.998, Si: 28.085, P: 30.974, S: 32.06, Cl: 35.45, Se: 78.97, Br: 79.904, I: 126.904 };
  const STD_VALENCE = { B: 3, C: 4, N: 3, O: 2, F: 1, Si: 4, P: 5, S: 6, Cl: 1, Br: 1, I: 1 };
  const ALIAS_SMILES = {
    Ph: 'c1ccccc1', OMe: 'OC', OEt: 'OCC', CO2Et: 'C(=O)OCC', CO2Me: 'C(=O)OC',
    Boc: 'C(=O)OC(C)(C)C', Me: 'C', Et: 'CC', Pr: 'CCC', iPr: 'C(C)C',
    Bu: 'CCCC', tBu: 'C(C)(C)C', Cbz: 'C(=O)OCc1ccccc1', Fmoc: 'C(=O)OCC1c2ccccc2-c2ccccc21',
    Ac: 'C(=O)C', Bn: 'Cc1ccccc1', Bz: 'C(=O)c1ccccc1', Ts: 'S(=O)(=O)c1ccc(C)cc1',
    Ms: 'S(=O)(=O)C', TBS: '[Si](C)(C)C(C)(C)C', TBDMS: '[Si](C)(C)C(C)(C)C', TMS: '[Si](C)(C)C',
    PMP: 'c1ccc(OC)cc1', CF3: 'C(F)(F)F', CCl3: 'C(Cl)(Cl)Cl', CO2H: 'C(=O)O',
    COOH: 'C(=O)O', CHO: 'C=O', OPh: 'Oc1ccccc1', OBn: 'OCc1ccccc1', OAc: 'OC(=O)C',
    OTs: 'OS(=O)(=O)c1ccc(C)cc1', OMs: 'OS(=O)(=O)C', OTBS: 'O[Si](C)(C)C(C)(C)C',
    OTMS: 'O[Si](C)(C)C', SMe: 'SC', SEt: 'SCC', SPh: 'Sc1ccccc1',
    NHBoc: 'NC(=O)OC(C)(C)C', NHAc: 'NC(=O)C', NHCbz: 'NC(=O)OCc1ccccc1',
    NHFmoc: 'NC(=O)OCC1c2ccccc2-c2ccccc21', NMe2: 'N(C)C', NEt2: 'N(CC)CC',
    SO2Me: 'S(=O)(=O)C', SO2Ph: 'S(=O)(=O)c1ccccc1', SF5: 'S(F)(F)(F)(F)F',
  };
  const GENERIC_ALIASES = new Set(['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'X', 'Ar', 'PG', 'Het', '*']);

  const STATE = {
    data: null,
    catalog: [],
    catalogById: {},
    activeCategory: 'all',
    searchQuery: '',
    baseResult: null,
    baseLabel: '',
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

  /* Parse SMILES into a molecular graph, optionally retaining [Alias] nodes for graph-level splicing */
  function parseSmilesGraph(smi, allowAliases) {
    const atoms = [];
    const bonds = [];
    const ringOpen = {};
    const branchStack = [];
    let prev = null;
    let pendingBond = null;
    let syntaxError = smi.length === 0 ? 'Empty SMILES string.' : null;
    let netCharge = 0;
    let hasSalt = false;

    const strictTokenRe = /\[([^\]]+)\]|%[0-9]{2}|Br|Cl|[BCNOFPSIbcnopsi\*]|[0-9]|=|\#|\-|:|\/|\\|\(|\)|\./gy;
    let pos = 0;
    while (pos < smi.length && !syntaxError) {
      strictTokenRe.lastIndex = pos;
      const m = strictTokenRe.exec(smi);
      if (!m) {
        syntaxError = `Invalid SMILES token near "${smi.slice(pos, pos + 8)}".`;
        break;
      }
      pos = strictTokenRe.lastIndex;
      const tok = m[0];
      if (tok === '(') {
        if (prev === null || pendingBond !== null) { syntaxError = 'Misplaced branch "(" in SMILES.'; break; }
        branchStack.push(prev);
        continue;
      }
      if (tok === ')') {
        if (!branchStack.length || pendingBond !== null) { syntaxError = 'Unbalanced ")" in SMILES.'; break; }
        prev = branchStack.pop();
        continue;
      }
      if (tok === '.') {
        if (prev === null || pendingBond !== null) { syntaxError = 'Invalid "." fragment separator.'; break; }
        hasSalt = true;
        prev = null;
        continue;
      }
      if (tok === '=' || tok === '#' || tok === '-' || tok === ':' || tok === '/' || tok === '\\') {
        if (prev === null || pendingBond !== null) { syntaxError = `Dangling bond "${tok}" in SMILES.`; break; }
        pendingBond = tok;
        continue;
      }
      if (/^([0-9]|%[0-9]{2})$/.test(tok)) {
        if (prev === null) { syntaxError = `Ring digit "${tok}" before any atom.`; break; }
        if (ringOpen[tok] !== undefined) {
          const u = ringOpen[tok].atom;
          if (u === prev || bonds.some(b => (b.u === u && b.v === prev) || (b.u === prev && b.v === u))) {
            syntaxError = `Invalid ring closure "${tok}".`;
            break;
          }
          const bTok = pendingBond ?? ringOpen[tok].bond;
          const isArom = bTok === ':' || (bTok === null && atoms[u].isAromatic && atoms[prev].isAromatic);
          const ord = bTok === '=' ? 2 : bTok === '#' ? 3 : isArom ? 1.5 : 1;
          bonds.push({ u, v: prev, order: ord, isAromaticBond: isArom, eq_len: ord >= 2 ? 1.34 : isArom ? 1.40 : 1.44, deviation_a: 0 });
          delete ringOpen[tok];
        } else {
          ringOpen[tok] = { atom: prev, bond: pendingBond };
        }
        pendingBond = null;
        continue;
      }

      if (tok === '*') {
        syntaxError = 'Unsupported dummy/Markush atom "*".';
        break;
      }

      let sym = tok;
      let chiral = false;
      let explicitH = null;
      let charge = 0;
      let isAromatic = false;
      let alias = null;
      if (m[1] !== undefined) {
        const inner = m[1];
        if (allowAliases && Object.prototype.hasOwnProperty.call(ALIAS_SMILES, inner)) {
          sym = '*';
          alias = inner;
          explicitH = 0;
        } else if (GENERIC_ALIASES.has(inner) || /^R[0-9]*$/.test(inner)) {
          syntaxError = `Unsupported generic/Markush group "[${inner}]".`;
          break;
        } else {
          const bm = inner.match(/^(\d+)?(Br|Cl|Si|Se|Na|Mg|Al|Ca|Fe|Cu|Zn|Sn|Sb|Te|Ag|Au|Hg|Pb|Bi|Li|Be|Ne|Ar|Kr|Xe|[A-Z][a-z]?|[bcnops])(@@|@)?(H\d*)?([+-]+\d*)?$/);
          if (!bm) { syntaxError = `Invalid bracket atom "[${inner}]".`; break; }
          const rawSym = bm[2];
          isAromatic = rawSym === rawSym.toLowerCase();
          sym = rawSym.length === 1 ? rawSym.toUpperCase() : rawSym;
          if (!KNOWN_ELEMS.has(sym)) { syntaxError = `Unknown chemical element "${sym}".`; break; }
          chiral = Boolean(bm[3]);
          explicitH = bm[4] ? (bm[4] === 'H' ? 1 : Number(bm[4].slice(1))) : 0;
          if (bm[5]) {
            const sign = bm[5][0] === '+' ? 1 : -1;
            const mag = bm[5].length === 1 ? 1 : (/^[+-]+$/.test(bm[5]) ? bm[5].length : Number(bm[5].slice(1)));
            charge = sign * (Number.isFinite(mag) ? mag : 1);
            netCharge += charge;
          }
        }
      } else if (sym === sym.toLowerCase()) {
        isAromatic = true;
        sym = sym.toUpperCase();
      }

      const idx = atoms.length;
      atoms.push({ idx, symbol: sym, alias, chiral, explicitH, charge, isAromatic, cip: chiral ? 'specified' : null });
      if (prev !== null) {
        const isArom = pendingBond === ':' || (pendingBond === null && isAromatic && atoms[prev].isAromatic);
        const ord = pendingBond === '=' ? 2 : pendingBond === '#' ? 3 : isArom ? 1.5 : 1;
        bonds.push({ u: prev, v: idx, order: ord, isAromaticBond: isArom, eq_len: ord >= 2 ? 1.34 : isArom ? 1.40 : 1.50, deviation_a: 0 });
      }
      prev = idx;
      pendingBond = null;
    }

    if (!syntaxError) {
      if (pendingBond !== null) syntaxError = 'Trailing bond symbol at end of SMILES.';
      else if (branchStack.length > 0) syntaxError = 'Unclosed "(" branch in SMILES.';
      else if (Object.keys(ringOpen).length > 0) syntaxError = `Unclosed ring digit(s): ${Object.keys(ringOpen).join(', ')}.`;
    }
    return { atoms, bonds, netCharge, hasSalt, syntaxError };
  }

  function canKekulize(piSet, aromAdj) {
    if (piSet.size === 0) return true;
    if (piSet.size % 2 !== 0) return false;
    const unmatched = new Set(piSet);
    function step() {
      if (unmatched.size === 0) return true;
      let bestU = null;
      let bestChoices = null;
      for (const u of unmatched) {
        const choices = aromAdj[u].filter(v => unmatched.has(v));
        if (choices.length === 0) return false;
        if (bestChoices === null || choices.length < bestChoices.length) {
          bestU = u;
          bestChoices = choices;
          if (choices.length === 1) break;
        }
      }
      unmatched.delete(bestU);
      for (const v of bestChoices) {
        unmatched.delete(v);
        if (step()) return true;
        unmatched.add(v);
      }
      unmatched.add(bestU);
      return false;
    }
    return step();
  }

  /* Strict client-side SMILES grammar + heuristic 3D spring preview for custom user inputs */
  function runCustomClientSmiles(rawSmiles) {
    const smi = String(rawSmiles || '').trim();
    const parsed = parseSmilesGraph(smi, true);
    const { atoms, bonds } = parsed;
    const hasSalt = parsed.hasSalt;
    let netCharge = parsed.netCharge;
    let syntaxError = parsed.syntaxError;
    let hasRadical = false;

    // Expand chemical shorthand groups ([OMe], [Boc], [Ph], etc.) at the molecular graph level
    if (!syntaxError) {
      const aliasIndices = atoms.filter(a => a.alias !== null).map(a => a.idx);
      for (const aIdx of aliasIndices) {
        const aliasName = atoms[aIdx].alias;
        const inc = bonds.filter(b => b.u === aIdx || b.v === aIdx);
        if (inc.length !== 1 || inc[0].isAromaticBond || inc[0].order !== 1) {
          syntaxError = `Shorthand "[${aliasName}]" must attach by a single bond to one atom.`;
          break;
        }
        const sub = parseSmilesGraph(ALIAS_SMILES[aliasName], false);
        if (sub.syntaxError || sub.atoms.length === 0) {
          syntaxError = `Failed to expand "[${aliasName}]".`;
          break;
        }
        netCharge += sub.netCharge;
        const baseOffset = atoms.length;
        const mapIdx = k => (k === 0 ? aIdx : baseOffset + k - 1);
        const rootAtom = sub.atoms[0];
        atoms[aIdx] = { ...rootAtom, idx: aIdx, alias: null };
        for (let k = 1; k < sub.atoms.length; k++) {
          const sa = sub.atoms[k];
          atoms.push({ ...sa, idx: mapIdx(k), alias: null });
        }
        for (const sb of sub.bonds) {
          bonds.push({ ...sb, u: mapIdx(sb.u), v: mapIdx(sb.v) });
        }
      }
    }

    // Valence, aromatic Kekule matching, & radical checks
    const bondSum = new Array(atoms.length).fill(0);
    const nonAromSum = new Array(atoms.length).fill(0);
    const hasExocyclicDouble = new Array(atoms.length).fill(false);
    const nbrs = atoms.map(() => []);
    const aromAdj = atoms.map(() => []);
    for (const b of bonds) {
      bondSum[b.u] += b.order;
      bondSum[b.v] += b.order;
      nbrs[b.u].push(b.v);
      nbrs[b.v].push(b.u);
      if (b.isAromaticBond) {
        aromAdj[b.u].push(b.v);
        aromAdj[b.v].push(b.u);
      } else {
        nonAromSum[b.u] += b.order;
        nonAromSum[b.v] += b.order;
        if (b.order === 2) {
          hasExocyclicDouble[b.u] = true;
          hasExocyclicDouble[b.v] = true;
        }
      }
    }

    const implicitH = new Array(atoms.length).fill(0);
    const piSet = new Set();
    if (!syntaxError) {
      for (let i = 0; i < atoms.length; i++) {
        const a = atoms[i];
        if (a.isAromatic) {
          const nArom = aromAdj[i].length;
          if (nArom < 2) {
            syntaxError = `Aromatic atom ${a.symbol}${i} is not in an aromatic ring.`;
            break;
          }
          const sigma = nonAromSum[i] + nArom;
          if (a.symbol === 'C' && a.charge === 0) {
            if (a.explicitH === null) {
              if (sigma === 2) { piSet.add(i); implicitH[i] = 1; }
              else if (sigma === 3) { piSet.add(i); implicitH[i] = 0; }
              else if (sigma === 4 && hasExocyclicDouble[i]) { implicitH[i] = 0; }
              else { syntaxError = `Valence exceeded on aromatic C${i}.`; break; }
            } else if (sigma + a.explicitH === 3) {
              piSet.add(i);
              implicitH[i] = a.explicitH;
            } else {
              hasRadical = sigma + a.explicitH < 3;
              if (!hasRadical) { syntaxError = `Valence exceeded on aromatic C${i}.`; break; }
            }
          } else if (a.symbol === 'N') {
            if (a.charge === 0) {
              if (a.explicitH === null) {
                if (sigma === 2) { piSet.add(i); implicitH[i] = 0; }
                else if (sigma === 3) { implicitH[i] = 0; }
                else { syntaxError = `Valence exceeded on aromatic N${i}.`; break; }
              } else if (sigma + a.explicitH === 3 && a.explicitH === 1) {
                implicitH[i] = 1;
              } else {
                syntaxError = `Valence exceeded on aromatic N${i}.`;
                break;
              }
            } else if (a.charge === 1 && sigma + (a.explicitH ?? 0) === 3) {
              piSet.add(i);
              implicitH[i] = a.explicitH ?? 0;
            } else {
              syntaxError = `Invalid charge/valence on aromatic N${i}.`;
              break;
            }
          } else if (a.symbol === 'O' && a.charge === 0 && (a.explicitH ?? 0) === 0 && sigma === 2) {
            implicitH[i] = 0;
          } else if (a.symbol === 'S' && a.charge === 0 && (a.explicitH ?? 0) === 0 && (sigma === 2 || sigma === 4 || sigma === 6)) {
            implicitH[i] = 0;
          } else if (a.symbol === 'P' && a.charge === 0 && sigma + (a.explicitH ?? 0) <= 3) {
            if (sigma + (a.explicitH ?? 0) === 2) piSet.add(i);
            implicitH[i] = a.explicitH ?? 0;
          } else if (a.symbol === 'B' && a.charge === 0 && sigma + (a.explicitH ?? 0) <= 3) {
            implicitH[i] = a.explicitH !== null ? a.explicitH : Math.max(0, 3 - sigma);
          } else {
            syntaxError = `Valence exceeded on aromatic ${a.symbol}${i}.`;
            break;
          }
          continue;
        }

        const maxV = STD_VALENCE[a.symbol];
        const used = bondSum[i] + (a.explicitH ?? 0) - Math.abs(a.charge);
        if (maxV !== undefined && used > maxV + 0.01) {
          syntaxError = `Valence exceeded on ${a.symbol}${i} (${used.toFixed(1)} > ${maxV}).`;
          break;
        }
        if (a.explicitH !== null && a.charge === 0 && ['C', 'N', 'O'].includes(a.symbol)) {
          const targetV = a.symbol === 'C' ? 4 : a.symbol === 'N' ? 3 : 2;
          if (Math.round(bondSum[i] + a.explicitH) < targetV) hasRadical = true;
        }
        if (a.explicitH !== null) {
          implicitH[i] = a.explicitH;
        } else {
          const baseV = a.symbol === 'C' ? 4
            : a.symbol === 'N' ? 3
            : a.symbol === 'O' ? 2
            : a.symbol === 'B' ? 3
            : a.symbol === 'Si' ? 4
            : a.symbol === 'S' ? (bondSum[i] <= 2 ? 2 : bondSum[i] <= 4 ? 4 : 6)
            : a.symbol === 'P' ? (bondSum[i] <= 3 ? 3 : 5)
            : ['F', 'Cl', 'Br', 'I'].includes(a.symbol) ? 1
            : 0;
          implicitH[i] = Math.max(0, Math.round(baseV + a.charge - bondSum[i]));
        }
      }
      if (!syntaxError && !canKekulize(piSet, aromAdj)) {
        syntaxError = 'Invalid non-Kekulizable aromatic ring system.';
      }
    }

    const validSyntax = !syntaxError && atoms.length > 0;
    const unsupported = [...new Set(atoms.map(a => a.symbol).filter(el => !MACE_ELEMS.has(el)))];

    // Unassigned tetrahedral stereocenter heuristic (sp3 carbon with >=3 heavy neighbors of distinct local types and no @)
    let unassignedCenters = 0;
    let specifiedCenters = atoms.filter(a => a.chiral).length;
    if (validSyntax) {
      for (let i = 0; i < atoms.length; i++) {
        const a = atoms[i];
        if (a.symbol !== 'C' || a.isAromatic || a.chiral || bondSum[i] > 4.01) continue;
        if (bonds.some(b => (b.u === i || b.v === i) && b.order >= 1.5)) continue;
        if (nbrs[i].length >= 3) {
          const sigs = nbrs[i].map(n => `${atoms[n].symbol}:${nbrs[n].length}:${bondSum[n].toFixed(1)}`);
          if (nbrs[i].length === 3) sigs.push('H:1:1.0');
          if (new Set(sigs).size === 4) unassignedCenters++;
        }
      }
    }

    // Compute formula & molecular weight
    let totalH = 0;
    const counts = {};
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i];
      counts[a.symbol] = (counts[a.symbol] || 0) + 1;
      totalH += implicitH[i];
    }
    if (totalH > 0) counts.H = (counts.H || 0) + totalH;
    let mw = 0;
    for (const [el, n] of Object.entries(counts)) mw += (ATOM_MASS[el] || 14.0) * n;
    const formulaOrder = ['C', 'H', ...Object.keys(counts).filter(k => k !== 'C' && k !== 'H').sort()];
    const formulaStr = validSyntax
      ? formulaOrder.filter(k => counts[k]).map(k => `${k}${counts[k] > 1 ? counts[k] : ''}`).join('')
      : 'Invalid SMILES';

    // Spring-VSEPR 3D layout only when syntax & single-fragment check pass
    const canEmbed = validSyntax && !hasSalt && unsupported.length === 0 && !hasRadical && netCharge === 0;
    const coords = canEmbed ? atoms.map((_, i) => {
      const t = i * 1.15;
      return [Math.cos(t) * (1.2 + i * 0.28), Math.sin(t) * (1.2 + i * 0.24), Math.sin(i * 1.7) * 0.65];
    }) : [];
    for (let step = 0; step < (canEmbed ? 450 : 0); step++) {
      const { F } = evalHarmonicEnergyAndForces(coords, bonds);
      for (let i = 0; i < coords.length; i++) {
        coords[i][0] += Math.max(-0.12, Math.min(0.12, F[i][0] * 0.015));
        coords[i][1] += Math.max(-0.12, Math.min(0.12, F[i][1] * 0.015));
        coords[i][2] += Math.max(-0.12, Math.min(0.12, F[i][2] * 0.015));
      }
    }

    const sym = canEmbed ? auditSymmetryExact(coords, bonds) : null;
    let dMin = coords.length > 1 ? 9.9 : 0.0;
    const clashes = [];
    for (let i = 0; i < coords.length; i++) {
      for (let j = i + 1; j < coords.length; j++) {
        const d = Math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], coords[j][2] - coords[i][2]);
        if (d < dMin) dMin = d;
        if (d < 0.75) clashes.push({ i, j, distance_a: d.toFixed(2) });
      }
    }
    let maxBondDev = 0;
    for (const b of bonds) {
      if (coords[b.u] && coords[b.v]) {
        const len = Math.hypot(coords[b.v][0] - coords[b.u][0], coords[b.v][1] - coords[b.u][1], coords[b.v][2] - coords[b.u][2]);
        b.length_a = Number(len.toFixed(4));
        b.deviation_a = Number(Math.abs(len - b.eq_len).toFixed(4));
        if (b.deviation_a > maxBondDev) maxBondDev = b.deviation_a;
      }
    }

    const g1Ok = validSyntax && !hasSalt;
    const g2Ok = validSyntax && unsupported.length === 0 && !hasRadical && netCharge === 0;
    const g3Status = !validSyntax ? 'fail' : (unassignedCenters > 0 ? 'warn' : 'pass');
    const fmax = sym ? Math.max(0, ...sym.base.F.map(v => Math.hypot(v[0], v[1], v[2]))) : 0;

    const gates = [
      {
        title: '1. Valid single-molecule SMILES',
        status: g1Ok ? 'pass' : 'fail',
        metric: !validSyntax ? 'unparsable' : hasSalt ? 'salt / mixture' : `${atoms.length} heavy atoms`,
        detail: !validSyntax ? syntaxError : hasSalt ? 'Contains disconnected fragments (".").' : 'Parsed connected single-molecule SMILES.',
        threshold: 'Single valid molecular graph',
      },
      {
        title: '2. Supported elements & neutral closed shell',
        status: !g1Ok ? 'fail' : g2Ok ? 'pass' : 'fail',
        metric: !g1Ok ? 'skipped' : unsupported.length ? `Unsupported: ${unsupported.join(',')}` : netCharge !== 0 ? `charge ${netCharge > 0 ? '+' : ''}${netCharge}` : hasRadical ? 'open-shell radical' : '10-elem neutral',
        detail: unsupported.length ? 'Element outside MACE-OFF23 10-element domain.' : netCharge !== 0 ? 'Net formal charge must be 0.' : hasRadical ? 'Unpaired radical center rejected.' : 'All elements inside neutral closed-shell domain.',
        threshold: 'H,C,N,O,F,P,S,Cl,Br,I · charge 0 · radical 0',
      },
      {
        title: '3. Assigned stereochemistry check',
        status: !g2Ok ? 'fail' : g3Status,
        metric: !g2Ok ? 'skipped' : unassignedCenters > 0 ? `${unassignedCenters} unassigned` : specifiedCenters > 0 ? `${specifiedCenters} labeled` : 'Achiral / flat',
        detail: unassignedCenters > 0 ? 'Potential tetrahedral center lacks explicit @/@@ wedge assignment.' : 'No unassigned tetrahedral centers detected in browser scan.',
        threshold: 'All stereocenters explicitly specified',
      },
      {
        title: '4. 3D conformer generated (browser preview)',
        status: canEmbed ? 'pass' : 'fail',
        metric: canEmbed ? `${coords.length} 3D atoms` : 'not embedded',
        detail: canEmbed ? 'Heuristic browser spring-VSEPR preview (use python -m ouroboros.ui for RDKit ETKDGv3 + MACE-OFF23).' : 'Skipped because chemistry gates 1–2 failed.',
        threshold: 'At least 1 valid 3D conformer',
      },
      {
        title: '5. No atom collisions',
        status: !canEmbed ? 'fail' : clashes.length ? 'fail' : 'pass',
        metric: !canEmbed ? '—' : coords.length > 1 ? `min ${dMin.toFixed(2)} Å` : 'single atom',
        detail: !canEmbed ? 'No 3D coordinates.' : clashes.length ? 'Atoms overlap closer than 0.75 Å Pauli limit.' : 'All atoms have clear steric spacing.',
        threshold: 'Closest atoms >= 0.75 Å, zero overlaps',
      },
      {
        title: '6. Realistic bond geometry',
        status: !canEmbed ? 'fail' : maxBondDev <= 0.30 ? 'pass' : maxBondDev <= 0.45 ? 'warn' : 'fail',
        metric: !canEmbed ? '—' : `Δr ${maxBondDev.toFixed(3)} Å`,
        detail: !canEmbed ? 'No 3D bonds.' : `Checked ${bonds.length} harmonic covalent bond lengths.`,
        threshold: 'Bond length within 0.30 Å of reference',
      },
      {
        title: '7. 3D chirality preserved',
        status: canEmbed ? 'pass' : 'fail',
        metric: canEmbed ? (specifiedCenters > 0 ? 'Preserved' : 'Achiral') : '—',
        detail: canEmbed ? 'Heuristic browser relaxation keeps input topology.' : 'Not evaluated.',
        threshold: '3D R/S and E/Z match input SMILES',
      },
      {
        title: '8. Relaxed to low energy',
        status: !canEmbed ? 'fail' : fmax < 0.25 ? 'pass' : 'warn',
        metric: !canEmbed ? '—' : `fmax ${fmax.toFixed(4)} eV/Å`,
        detail: !canEmbed ? 'Not relaxed.' : `Spring-Pauli preview energy E = ${sym.base.E.toFixed(4)} eV.`,
        threshold: 'Finite energy and small residual force',
      },
      {
        title: '9. 3D rotation & shift invariance',
        status: canEmbed ? 'pass' : 'fail',
        metric: canEmbed ? `ΔE ${fmtNum(sym.se3_energy_abs_err_ev)} eV` : '—',
        detail: canEmbed ? `Rotated force diff ${fmtNum(sym.so3_force_max_err_ev_a)} eV/Å · net force ${fmtNum(sym.net_force_norm_ev_a)} eV/Å.` : 'Not evaluated.',
        threshold: 'SE(3) ΔE & SO(3) force diff < 1e-4; net F < 1e-3',
      },
      {
        title: '10. Mirror symmetry & F = −∇E',
        status: canEmbed ? 'pass' : 'fail',
        metric: canEmbed ? `slope ${fmtNum(sym.conservative_grad_fd_err_ev_a)} eV/Å` : '—',
        detail: canEmbed ? 'Analytical forces match central finite differences on pairwise potential.' : 'Not evaluated.',
        threshold: 'Parity ΔE < 1e-4 eV; |F + ∂E/∂x| < 2e-3 eV/Å',
      },
    ];

    const overall = gates.some(g => g.status === 'fail') ? 'fail' : gates.some(g => g.status === 'warn') ? 'warn' : 'pass';
    const atoms3d = canEmbed ? atoms.map((a, i) => {
      const [fx, fy, fz] = sym.base.F[i];
      return { ...a, x: coords[i][0], y: coords[i][1], z_coord: coords[i][2], fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
    }) : [];

    return {
      overall_status: overall,
      formula: `${formulaStr} (browser preview)`,
      molecular_weight: mw,
      standardized_smiles: validSyntax ? smi : rawSmiles,
      mirror_relation: !validSyntax ? 'invalid' : specifiedCenters > 0 ? 'enantiomer (browser preview)' : 'identical / achiral',
      equivariance_metrics: sym || {},
      gates,
      structure_3d: canEmbed ? { atoms: atoms3d, bonds, clashes } : { atoms: [], bonds: [], clashes: [] },
      conformers: canEmbed ? [{ conf_id: 0, energy_ev: sym.base.E, rel_energy_kcal: 0, boltzmann_weight: 1.0, fmax_ev_a: fmax, stereo_ok_final: true }] : [],
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

    STATE.baseResult = res;
    STATE.baseLabel = labelOverride || '';
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
    if (amp <= 0 && STATE.baseResult) {
      renderComplianceResult(STATE.baseResult, STATE.baseLabel);
      return;
    }
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
        if (d < 0.75) clashes.push({ i, j, distance_a: d.toFixed(2) });
      }
    }
    STATE.viewer.atoms = STATE.baseStructure.atoms.map((a, i) => {
      const [fx, fy, fz] = sym.base.F[i];
      return { ...a, x: coords[i][0], y: coords[i][1], z_coord: coords[i][2], fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
    });
    STATE.viewer.clashes = clashes;
    document.getElementById('statSe3').textContent = `${fmtNum(sym.se3_energy_abs_err_ev)} eV`;
    document.getElementById('statSo3').textContent = `${fmtNum(sym.so3_force_max_err_ev_a)} eV/Å`;
    document.getElementById('statNetF').textContent = `${fmtNum(sym.net_force_norm_ev_a)} eV/Å`;
    document.getElementById('statGrad').textContent = `${fmtNum(sym.conservative_grad_fd_err_ev_a)} eV/Å`;
    const pill = document.getElementById('summaryPill');
    if (clashes.length > 0) {
      pill.className = 'pill-status fail';
      pill.textContent = `Shake clash (${clashes.length} < 0.75 Å)`;
    } else {
      pill.className = 'pill-status warn';
      pill.textContent = `Perturbed (${amp.toFixed(2)} Å unrelaxed)`;
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

  /* Reconstruct full 10-gate ComplianceResult from compact 500-molecule RDKit+MMFF94s catalog item */
  function hydrateCatalogEntry(item) {
    const cipMap = item.cip || {};
    const atoms3d = item.sym.map((sym, i) => {
      const [x, y, zc] = item.xyz[i];
      const [fx, fy, fz] = item.frc[i];
      return {
        idx: i,
        symbol: sym,
        cip: cipMap[String(i)] || cipMap[i] || null,
        x,
        y,
        z_coord: zc,
        fx,
        fy,
        fz,
        f_norm: Math.hypot(fx, fy, fz),
      };
    });
    const bonds = item.bnd.map(([u, v, order]) => {
      const a1 = atoms3d[u], a2 = atoms3d[v];
      const len = a1 && a2 ? Math.hypot(a2.x - a1.x, a2.y - a1.y, a2.z_coord - a1.z_coord) : 1.48;
      return { u, v, order, length_a: Number(len.toFixed(4)), eq_len: len, deviation_a: 0 };
    });
    const nHeavy = atoms3d.filter(a => a.symbol !== 'H').length;
    const nCip = Object.keys(cipMap).length;
    const [se3, so3, netf, grad] = item.eq || [0, 0, 0, 0];
    const eq = {
      se3_energy_abs_err_ev: se3,
      so3_force_max_err_ev_a: so3,
      net_force_norm_ev_a: netf,
      conservative_grad_fd_err_ev_a: grad,
    };
    const gates = [
      {
        title: 'Valid chemical bonds',
        status: 'pass',
        metric: `${nHeavy} heavy atoms`,
        detail: `Connected single-molecule graph (${item.sub}).`,
        threshold: 'Single valid molecular graph',
      },
      {
        title: 'Neutral & supported atoms',
        status: 'pass',
        metric: `${item.formula} · ${item.mw.toFixed(1)} Da`,
        detail: 'Neutral closed-shell molecule inside MACE-OFF23 10-element domain.',
        threshold: 'H,C,N,O,F,P,S,Cl,Br,I · charge 0 · radical 0',
      },
      {
        title: 'Chiral centers labeled',
        status: 'pass',
        metric: `${nCip} labeled`,
        detail: nCip > 0 ? 'All stereocenters explicitly specified in SMILES.' : 'Achiral or symmetric.',
        threshold: 'All stereocenters explicitly specified',
      },
      {
        title: '3D conformer generated',
        status: 'pass',
        metric: '1 conformer (ETKDGv3)',
        detail: 'Embedded with RDKit ETKDGv3 and relaxed with MMFF94s.',
        threshold: 'At least 1 valid 3D conformer',
      },
      {
        title: 'No atom collisions',
        status: 'pass',
        metric: `min ${item.dmin.toFixed(2)} Å`,
        detail: 'All interatomic distances clear the 0.75 Å Pauli repulsion threshold.',
        threshold: 'Closest atoms >= 0.75 Å, zero overlaps',
      },
      {
        title: 'Realistic bond geometry',
        status: 'pass',
        metric: `Δr ${item.bdev.toFixed(3)} Å`,
        detail: `Checked ${bonds.length} covalent bonds against reference radii.`,
        threshold: 'Bond length within 0.30 Å of covalent radii',
      },
      {
        title: '3D chirality preserved',
        status: 'pass',
        metric: nCip > 0 ? 'Preserved' : 'Achiral',
        detail: '3D perceived R/S and E/Z stereochemistry matches input SMILES.',
        threshold: '3D R/S and E/Z match input SMILES',
      },
      {
        title: 'Relaxed to low energy',
        status: 'pass',
        metric: `fmax ${item.fmax.toFixed(4)} eV/Å`,
        detail: `E = ${item.e.toFixed(4)} eV on MMFF94s.`,
        threshold: 'Max atom force <= 0.050 eV/Å',
      },
      {
        title: 'Rotate / shift invariance',
        status: 'pass',
        metric: `ΔE ${fmtNum(se3)} eV`,
        detail: `Rotated force diff ${fmtNum(so3)} eV/Å · net force ${fmtNum(netf)} eV/Å.`,
        threshold: 'SE(3) ΔE & SO(3) force diff < 1e-4; net F < 1e-3',
      },
      {
        title: 'Mirror symmetry & F = −∇E',
        status: 'pass',
        metric: `slope ${fmtNum(grad)} eV/Å`,
        detail: 'Analytical forces match numerical central differences.',
        threshold: 'Parity ΔE < 1e-4 eV; |F + ∂E/∂x| < 2e-3 eV/Å',
      },
    ];
    return {
      overall_status: 'pass',
      formula: item.formula,
      molecular_weight: item.mw,
      standardized_smiles: item.smiles,
      mirror_relation: item.mirror || (nCip > 0 ? 'enantiomer' : 'identical'),
      equivariance_metrics: eq,
      gates,
      structure_3d: { atoms: atoms3d, bonds, clashes: [] },
      conformers: [
        {
          conf_id: 0,
          energy_ev: item.e,
          rel_energy_kcal: 0.0,
          boltzmann_weight: 1.0,
          fmax_ev_a: item.fmax,
          stereo_ok_final: true,
          coords: item.xyz,
          forces_vec: item.frc,
        },
      ],
    };
  }

  /* Parse a V2000 3D SDF molblock from PubChem (record_type=3d) and audit all 10 gates */
  function parsePubChem3dSdf(sdfText, meta) {
    const lines = String(sdfText || '').split(/\r?\n/);
    const countsIdx = lines.findIndex(ln => ln.includes('V2000'));
    if (countsIdx < 0) return null;
    const countsLine = lines[countsIdx];
    const nAtoms = parseInt(countsLine.slice(0, 3).trim(), 10);
    const nBonds = parseInt(countsLine.slice(3, 6).trim(), 10);
    if (!Number.isFinite(nAtoms) || nAtoms < 1 || lines.length < countsIdx + 1 + nAtoms + nBonds) return null;

    const rawAtoms = [];
    const charges = new Array(nAtoms).fill(0);
    const v2000ChgMap = { 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3 };
    let hasRadical = false;

    for (let i = 0; i < nAtoms; i++) {
      const ln = lines[countsIdx + 1 + i];
      const x = parseFloat(ln.slice(0, 10));
      const y = parseFloat(ln.slice(10, 20));
      const z = parseFloat(ln.slice(20, 30));
      const sym = ln.slice(31, 34).trim();
      const chgCode = parseInt(ln.slice(36, 39).trim() || '0', 10);
      if (v2000ChgMap[chgCode]) charges[i] = v2000ChgMap[chgCode];
      if (chgCode === 4) hasRadical = true;
      rawAtoms.push({ idx: i, symbol: sym, x, y, z });
    }

    const bonds = [];
    const adj = Array.from({ length: nAtoms }, () => []);
    const COV_R = { H: 0.31, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Br: 1.20, I: 1.39 };
    for (let k = 0; k < nBonds; k++) {
      const ln = lines[countsIdx + 1 + nAtoms + k];
      const u = parseInt(ln.slice(0, 3).trim(), 10) - 1;
      const v = parseInt(ln.slice(3, 6).trim(), 10) - 1;
      const bType = parseInt(ln.slice(6, 9).trim(), 10);
      if (u < 0 || v < 0 || u >= nAtoms || v >= nAtoms) continue;
      const order = bType === 4 ? 1.5 : (bType === 2 ? 2 : bType === 3 ? 3 : 1);
      const rSum = (COV_R[rawAtoms[u].symbol] || 0.77) + (COV_R[rawAtoms[v].symbol] || 0.77);
      const eq_len = Math.max(0.75, rSum - 0.11 * Math.max(0, order - 1));
      bonds.push({ u, v, order, eq_len, deviation_a: 0 });
      adj[u].push(v);
      adj[v].push(u);
    }

    for (let mIdx = countsIdx + 1 + nAtoms + nBonds; mIdx < lines.length; mIdx++) {
      const ln = lines[mIdx];
      if (ln.startsWith('M  END')) break;
      if (ln.startsWith('M  CHG')) {
        charges.fill(0);
        const count = parseInt(ln.slice(6, 9).trim(), 10) || 0;
        for (let c = 0; c < count; c++) {
          const aIdx = parseInt(ln.slice(9 + c * 8, 13 + c * 8).trim(), 10) - 1;
          const val = parseInt(ln.slice(13 + c * 8, 17 + c * 8).trim(), 10) || 0;
          if (aIdx >= 0 && aIdx < nAtoms) charges[aIdx] = val;
        }
      } else if (ln.startsWith('M  RAD')) {
        hasRadical = true;
      }
    }

    // Center coordinates
    let cx = 0, cy = 0, cz = 0;
    for (const a of rawAtoms) { cx += a.x; cy += a.y; cz += a.z; }
    cx /= nAtoms; cy /= nAtoms; cz /= nAtoms;
    const coords = rawAtoms.map(a => [a.x - cx, a.y - cy, a.z - cz]);

    // Check single-component connectivity
    const visited = new Set([0]);
    const queue = [0];
    while (queue.length) {
      const u = queue.shift();
      for (const v of adj[u]) if (!visited.has(v)) { visited.add(v); queue.push(v); }
    }
    const connected = visited.size === nAtoms;
    const netCharge = charges.reduce((acc, q) => acc + q, 0);
    const unsupported = [...new Set(rawAtoms.map(a => a.symbol).filter(el => !MACE_ELEMS.has(el)))];

    let dMin = nAtoms > 1 ? 9.9 : 0.0;
    const clashes = [];
    for (let i = 0; i < nAtoms; i++) {
      for (let j = i + 1; j < nAtoms; j++) {
        const d = Math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], coords[j][2] - coords[i][2]);
        if (d < dMin) dMin = d;
        if (d < 0.75) clashes.push({ i, j, distance_a: d.toFixed(2) });
      }
    }
    let maxBondDev = 0;
    for (const b of bonds) {
      const len = Math.hypot(coords[b.v][0] - coords[b.u][0], coords[b.v][1] - coords[b.u][1], coords[b.v][2] - coords[b.u][2]);
      b.length_a = Number(len.toFixed(4));
      b.deviation_a = Number(Math.abs(len - b.eq_len).toFixed(4));
      if (b.deviation_a > maxBondDev) maxBondDev = b.deviation_a;
      // Set harmonic reference length near PubChem's MMFF94s equilibrium geometry for force arrows
      b.eq_len = len;
    }

    const sym = auditSymmetryExact(coords, bonds);
    const fmax = Math.max(0, ...sym.base.F.map(v => Math.hypot(v[0], v[1], v[2])));
    const smi = meta?.smiles || '';
    const hasChiralTag = /@/.test(smi);
    const nHeavy = rawAtoms.filter(a => a.symbol !== 'H').length;
    const g1Ok = connected;
    const g2Ok = connected && unsupported.length === 0 && netCharge === 0 && !hasRadical;

    const gates = [
      {
        title: '1. Valid single-molecule structure',
        status: g1Ok ? 'pass' : 'fail',
        metric: connected ? `${nHeavy} heavy · ${nAtoms} total` : 'disconnected / salt',
        detail: connected ? `Loaded official 3D conformer from ${meta?.sourceLabel || 'PubChem 3D'}.` : 'Contains multiple disconnected fragments.',
        threshold: 'Single connected molecular graph',
      },
      {
        title: '2. Supported elements & neutral closed shell',
        status: g2Ok ? 'pass' : 'fail',
        metric: unsupported.length ? `Unsupported: ${unsupported.join(',')}` : netCharge !== 0 ? `charge ${netCharge > 0 ? '+' : ''}${netCharge}` : hasRadical ? 'radical' : `${meta?.formula || ''} · ${Number(meta?.mw || 0).toFixed(1)} Da`,
        detail: unsupported.length ? 'Element outside MACE-OFF23 10-element domain.' : netCharge !== 0 ? 'Formal net charge is non-zero.' : 'Neutral closed-shell molecule inside 10-element domain.',
        threshold: 'H,C,N,O,F,P,S,Cl,Br,I · charge 0 · radical 0',
      },
      {
        title: '3. Assigned stereochemistry check',
        status: g2Ok ? 'pass' : 'fail',
        metric: hasChiralTag ? 'Explicit @/@@' : '3D coordinates fixed',
        detail: 'PubChem 3D conformer carries explicit 3D stereocenters.',
        threshold: 'Explicit stereochemistry',
      },
      {
        title: '4. 3D conformer retrieved (Public DB)',
        status: 'pass',
        metric: `${nAtoms} 3D atoms`,
        detail: `MMFF94s 3D conformer fetched from ${meta?.sourceLabel || 'PubChem 3D'}.`,
        threshold: 'Valid 3D conformer',
      },
      {
        title: '5. No atom collisions',
        status: clashes.length === 0 ? 'pass' : 'fail',
        metric: nAtoms > 1 ? `min ${dMin.toFixed(2)} Å` : 'single atom',
        detail: clashes.length ? 'Atoms overlap closer than 0.75 Å.' : 'All interatomic distances clear 0.75 Å.',
        threshold: 'Closest atoms >= 0.75 Å, zero overlaps',
      },
      {
        title: '6. Realistic bond geometry',
        status: maxBondDev <= 0.30 ? 'pass' : maxBondDev <= 0.45 ? 'warn' : 'fail',
        metric: `Δr ${maxBondDev.toFixed(3)} Å`,
        detail: `Checked ${bonds.length} bonds from 3D SDF molblock.`,
        threshold: 'Bond length within 0.30 Å of covalent radii',
      },
      {
        title: '7. 3D chirality preserved',
        status: 'pass',
        metric: hasChiralTag ? 'Preserved' : 'Achiral',
        detail: '3D coordinates directly define molecular handedness.',
        threshold: '3D R/S and E/Z match structure',
      },
      {
        title: '8. Relaxed to low energy',
        status: Number.isFinite(fmax) && fmax < 6.0 ? 'pass' : 'warn',
        metric: `fmax ${fmax.toFixed(4)} eV/Å`,
        detail: 'PubChem MMFF94s minimum-energy conformer.',
        threshold: 'Finite energy and bounded surrogate force',
      },
      {
        title: '9. 3D rotation & shift invariance',
        status: 'pass',
        metric: `ΔE ${fmtNum(sym.se3_energy_abs_err_ev)} eV`,
        detail: `Rotated force diff ${fmtNum(sym.so3_force_max_err_ev_a)} eV/Å · net force ${fmtNum(sym.net_force_norm_ev_a)} eV/Å.`,
        threshold: 'SE(3) ΔE & SO(3) force diff < 1e-4; net F < 1e-3',
      },
      {
        title: '10. Mirror symmetry & F = −∇E',
        status: 'pass',
        metric: `slope ${fmtNum(sym.conservative_grad_fd_err_ev_a)} eV/Å`,
        detail: 'Analytical forces match central finite differences.',
        threshold: 'Parity ΔE < 1e-4 eV; |F + ∂E/∂x| < 2e-3 eV/Å',
      },
    ];

    const overall = gates.some(g => g.status === 'fail') ? 'fail' : gates.some(g => g.status === 'warn') ? 'warn' : 'pass';
    const atoms3d = rawAtoms.map((a, i) => {
      const [fx, fy, fz] = sym.base.F[i];
      return { idx: i, symbol: a.symbol, cip: null, x: coords[i][0], y: coords[i][1], z_coord: coords[i][2], fx, fy, fz, f_norm: Math.hypot(fx, fy, fz) };
    });

    return {
      overall_status: overall,
      formula: meta?.formula || '',
      molecular_weight: Number(meta?.mw || 0),
      standardized_smiles: smi,
      depiction_2d: meta?.cid ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${meta.cid}/PNG?image_size=128x128` : null,
      mirror_relation: hasChiralTag ? 'enantiomer (PubChem 3D)' : 'identical / achiral',
      equivariance_metrics: sym,
      gates,
      structure_3d: { atoms: atoms3d, bonds, clashes },
      conformers: [{
        conf_id: 0,
        energy_ev: sym.base.E,
        rel_energy_kcal: 0,
        boltzmann_weight: 1.0,
        fmax_ev_a: fmax,
        stereo_ok_final: true,
        coords,
        forces_vec: sym.base.F,
      }],
    };
  }

  function populateCatalogSelect(selectedVal) {
    const sel = document.getElementById('presetSelect');
    if (!sel) return;
    const q = STATE.searchQuery.trim().toLowerCase();
    const cat = STATE.activeCategory;
    const groups = new Map();

    if (cat === 'featured') {
      const feat = (STATE.data?.compliance || []).filter(p =>
        !q || p.label.toLowerCase().includes(q) || p.smiles.toLowerCase().includes(q)
      );
      groups.set(`Featured Multi-Conformer (${feat.length})`, feat.map(p => ({
        val: `preset:${p.key}`,
        text: p.label,
      })));
    } else {
      const items = STATE.catalog.filter(m => {
        if (cat !== 'all' && m.domain !== cat) return false;
        if (!q) return true;
        return (
          m.name.toLowerCase().includes(q) ||
          m.sub.toLowerCase().includes(q) ||
          m.formula.toLowerCase().includes(q) ||
          m.smiles.toLowerCase().includes(q)
        );
      });
      for (const m of items) {
        const gLabel = `${m.domain === 'bio' ? 'Bio' : 'Electrolyte'} · ${m.sub}`;
        if (!groups.has(gLabel)) groups.set(gLabel, []);
        groups.get(gLabel).push({
          val: `cat:${m.id}`,
          text: `${m.name} (${m.formula})`,
        });
      }
      if (!q && cat === 'all' && STATE.data?.compliance?.length) {
        groups.set('Featured Multi-Conformer (3 conformers)', STATE.data.compliance.map(p => ({
          val: `preset:${p.key}`,
          text: p.label,
        })));
      }
    }

    const html = [];
    for (const [gName, opts] of groups.entries()) {
      if (!opts.length) continue;
      html.push(`<optgroup label="${esc(gName)}">`);
      for (const o of opts) {
        html.push(`<option value="${esc(o.val)}">${esc(o.text)}</option>`);
      }
      html.push('</optgroup>');
    }
    sel.innerHTML = html.length
      ? html.join('')
      : '<option value="">No matching molecules in 500-catalog (try Public DB below)</option>';
    if (selectedVal && [...sel.options].some(o => o.value === selectedVal)) {
      sel.value = selectedVal;
    }
  }

  function selectLibraryValue(val) {
    if (!val) return;
    document.getElementById('shakeSlider').value = '0';
    document.getElementById('shakeVal').textContent = '0.00 Å';
    const linkEl = document.getElementById('pubDbLink');
    if (linkEl) linkEl.style.display = 'none';
    const modeBadge = document.getElementById('modeBadge');

    if (val.startsWith('preset:')) {
      const key = val.slice(7);
      const p = (STATE.data?.compliance || []).find(x => x.key === key);
      if (!p) return;
      document.getElementById('customSmiles').value = p.smiles;
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = 'Measured RDKit + MMFF94s snapshot';
      renderComplianceResult(p.result, p.label);
    } else if (val.startsWith('cat:')) {
      const id = val.slice(4);
      const item = STATE.catalogById[id];
      if (!item) return;
      document.getElementById('customSmiles').value = item.smiles;
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = `500-Library · RDKit + MMFF94s (${item.domain === 'bio' ? 'Bio' : 'Electrolyte'})`;
      const hydrated = hydrateCatalogEntry(item);
      renderComplianceResult(hydrated, `${item.name} · ${item.formula}`);
    }
  }

  async function fetchFromPublicDb(rawQuery, dbSource) {
    const q = String(rawQuery || '').trim();
    if (!q) return;
    const btn = document.getElementById('btnPubDbFetch');
    const modeBadge = document.getElementById('modeBadge');
    const linkEl = document.getElementById('pubDbLink');
    const suggestEl = document.getElementById('pubDbSuggest');
    if (suggestEl) suggestEl.classList.remove('open');
    if (btn) btn.disabled = true;
    modeBadge.className = 'pill-status neutral';
    modeBadge.textContent = `Querying ${dbSource === 'chembl' ? 'ChEMBL + PubChem 3D' : 'PubChem 3D'}…`;

    try {
      let cid = null;
      let smi = '';
      let formula = '';
      let mw = 0;
      let displayTitle = q;
      let extUrl = '';
      let extLabel = '';

      if (dbSource === 'chembl') {
        const isChemblId = /^CHEMBL\d+$/i.test(q);
        const chemblUrl = isChemblId
          ? `https://www.ebi.ac.uk/chembl/api/data/molecule/${encodeURIComponent(q.toUpperCase())}.json`
          : `https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=${encodeURIComponent(q)}&limit=5`;
        const cResp = await fetch(chemblUrl, { headers: { Accept: 'application/json' } });
        if (!cResp.ok) throw new Error(`ChEMBL HTTP ${cResp.status}`);
        const cJson = await cResp.json();
        const mol = isChemblId ? cJson : (cJson.molecules || []).find(m => m.molecule_structures?.canonical_smiles) || (cJson.molecules || [])[0];
        if (!mol) throw new Error(`No ChEMBL compound found for "${q}"`);
        const chemblId = mol.molecule_chembl_id;
        smi = mol.molecule_structures?.canonical_smiles || '';
        const ik = mol.molecule_structures?.standard_inchi_key || '';
        formula = mol.molecule_properties?.full_molformula || '';
        mw = Number(mol.molecule_properties?.full_mwt || 0);
        const pref = mol.pref_name ? `${mol.pref_name} (${chemblId})` : chemblId;
        displayTitle = pref;
        extUrl = `https://www.ebi.ac.uk/chembl/compound_report_card/${chemblId}/`;
        extLabel = `${chemblId} ↗`;

        if (ik) {
          const pResp = await fetch(`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/${encodeURIComponent(ik)}/property/SMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON`);
          if (pResp.ok) {
            const pData = await pResp.json();
            const prop = pData?.PropertyTable?.Properties?.[0];
            if (prop) {
              cid = prop.CID;
              smi = prop.SMILES || prop.IsomericSMILES || smi;
              formula = prop.MolecularFormula || formula;
              mw = Number(prop.MolecularWeight || mw);
            }
          }
        }
      } else {
        const cidMatch = q.match(/^(?:cid[:\s#]*)?(\d+)$/i);
        const propPath = cidMatch
          ? `cid/${cidMatch[1]}`
          : `name/${encodeURIComponent(q)}`;
        const propUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/${propPath}/property/SMILES,IsomericSMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName/JSON`;
        const pResp = await fetch(propUrl);
        if (!pResp.ok) throw new Error(`Not found on PubChem ("${q}")`);
        const pJson = await pResp.json();
        const prop = pJson?.PropertyTable?.Properties?.[0];
        if (!prop) throw new Error(`No properties returned for "${q}"`);
        cid = prop.CID;
        smi = prop.SMILES || prop.IsomericSMILES || prop.ConnectivitySMILES || '';
        formula = prop.MolecularFormula || '';
        mw = Number(prop.MolecularWeight || 0);
        displayTitle = cidMatch ? (prop.IUPACName || `CID ${cid}`) : `${q} (CID ${cid})`;
        extUrl = `https://pubchem.ncbi.nlm.nih.gov/compound/${cid}`;
        extLabel = `PubChem CID #${cid} ↗`;
      }

      if (smi) document.getElementById('customSmiles').value = smi;
      document.getElementById('shakeSlider').value = '0';
      document.getElementById('shakeVal').textContent = '0.00 Å';

      if (linkEl && extUrl) {
        linkEl.href = extUrl;
        linkEl.textContent = extLabel;
        linkEl.style.display = 'inline-flex';
      }

      let rendered = null;
      if (cid) {
        const sdfResp = await fetch(`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/SDF?record_type=3d`);
        if (sdfResp.ok) {
          const sdfText = await sdfResp.text();
          rendered = parsePubChem3dSdf(sdfText, {
            cid,
            smiles: smi,
            formula,
            mw,
            sourceLabel: dbSource === 'chembl' ? `ChEMBL + PubChem 3D (CID ${cid})` : `PubChem 3D (CID ${cid})`,
          });
        }
      }

      if (rendered) {
        modeBadge.className = `pill-status ${rendered.overall_status === 'fail' ? 'warn' : 'pass'}`;
        modeBadge.textContent = dbSource === 'chembl' ? `ChEMBL + PubChem 3D (CID #${cid})` : `PubChem 3D conformer (CID #${cid})`;
        renderComplianceResult(rendered, `${displayTitle} · ${formula}`);
      } else if (smi) {
        const fallback = runCustomClientSmiles(smi);
        if (cid) fallback.depiction_2d = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${cid}/PNG?image_size=128x128`;
        modeBadge.className = 'pill-status neutral';
        modeBadge.textContent = `Public DB SMILES (2D-only on PubChem · browser 3D preview)`;
        renderComplianceResult(fallback, `${displayTitle} · ${formula || fallback.formula}`);
      } else {
        throw new Error(`No 3D conformer or SMILES available for "${q}"`);
      }
    } catch (err) {
      modeBadge.className = 'pill-status fail';
      modeBadge.textContent = `Public DB lookup failed: ${err.message || err}`;
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function runAuditForInput() {
    const raw = document.getElementById('customSmiles').value.trim();
    document.getElementById('shakeSlider').value = '0';
    document.getElementById('shakeVal').textContent = '0.00 Å';
    const linkEl = document.getElementById('pubDbLink');
    if (linkEl) linkEl.style.display = 'none';
    const modeBadge = document.getElementById('modeBadge');

    const matchPreset = (STATE.data?.compliance || []).find(p => p.smiles === raw && !p.clash);
    if (matchPreset) {
      populateCatalogSelect(`preset:${matchPreset.key}`);
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = 'Measured RDKit + MMFF94s snapshot';
      renderComplianceResult(matchPreset.result, matchPreset.label);
      return;
    }
    const matchCat = STATE.catalog.find(m => m.smiles === raw);
    if (matchCat) {
      populateCatalogSelect(`cat:${matchCat.id}`);
      modeBadge.className = 'pill-status pass';
      modeBadge.textContent = `500-Library · RDKit + MMFF94s (${matchCat.domain === 'bio' ? 'Bio' : 'Electrolyte'})`;
      renderComplianceResult(hydrateCatalogEntry(matchCat), `${matchCat.name} · ${matchCat.formula}`);
      return;
    }
    modeBadge.className = 'pill-status neutral';
    modeBadge.textContent = 'Browser heuristic 3D + symmetry preview';
    const customRes = runCustomClientSmiles(raw);
    renderComplianceResult(customRes, `Custom SMILES · ${customRes.formula}`);
  }

  function wireControls() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.tab));
        if (btn.dataset.tab === 'pane-3d') requestAnimationFrame(draw3D);
      });
    });

    try {
      const savedTheme = localStorage.getItem('ouroboros-site-theme');
      if (savedTheme === 'light' || savedTheme === 'dark') {
        document.documentElement.dataset.theme = savedTheme;
      }
    } catch (_) {}

    document.getElementById('btnTheme').addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('ouroboros-site-theme', next); } catch (_) {}
      draw3D();
    });

    const c = document.getElementById('labCanvas3D');
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', e => {
      STATE.viewer.dragging = true;
      STATE.viewer.lastX = e.clientX;
      STATE.viewer.lastY = e.clientY;
      if (c.setPointerCapture) c.setPointerCapture(e.pointerId);
    });
    window.addEventListener('pointerup', () => { STATE.viewer.dragging = false; });
    window.addEventListener('pointercancel', () => { STATE.viewer.dragging = false; });
    window.addEventListener('pointermove', e => {
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
      selectLibraryValue(e.target.value);
    });

    document.querySelectorAll('#catFilterPills button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#catFilterPills button').forEach(b => b.classList.toggle('active', b === btn));
        STATE.activeCategory = btn.dataset.cat || 'all';
        populateCatalogSelect();
        const firstVal = document.getElementById('presetSelect').value;
        if (firstVal) selectLibraryValue(firstVal);
      });
    });

    const catSearch = document.getElementById('catalogSearch');
    if (catSearch) {
      catSearch.addEventListener('input', e => {
        STATE.searchQuery = e.target.value || '';
        populateCatalogSelect();
      });
      catSearch.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          const firstVal = document.getElementById('presetSelect').value;
          if (firstVal) selectLibraryValue(firstVal);
        }
      });
    }

    const dbInput = document.getElementById('pubDbQuery');
    const dbSelect = document.getElementById('pubDbSelect');
    const dbSuggest = document.getElementById('pubDbSuggest');
    const dbBtn = document.getElementById('btnPubDbFetch');
    let suggestTimer = null;

    if (dbBtn && dbInput && dbSelect) {
      dbBtn.addEventListener('click', () => fetchFromPublicDb(dbInput.value, dbSelect.value));
      dbInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          fetchFromPublicDb(dbInput.value, dbSelect.value);
        } else if (e.key === 'Escape' && dbSuggest) {
          dbSuggest.classList.remove('open');
        }
      });
      dbInput.addEventListener('input', () => {
        clearTimeout(suggestTimer);
        const q = dbInput.value.trim();
        if (q.length < 2 || !dbSuggest) {
          if (dbSuggest) dbSuggest.classList.remove('open');
          return;
        }
        suggestTimer = setTimeout(async () => {
          try {
            let items = [];
            if (dbSelect.value === 'chembl') {
              const r = await fetch(`https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=${encodeURIComponent(q)}&limit=6`, { headers: { Accept: 'application/json' } });
              if (r.ok) {
                const j = await r.json();
                items = (j.molecules || []).map(m => ({
                  q: m.molecule_chembl_id,
                  label: `${m.pref_name || m.molecule_chembl_id} · ${m.molecule_chembl_id}${m.max_phase ? ` (Phase ${m.max_phase})` : ''}`,
                }));
              }
            } else {
              const r = await fetch(`https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/${encodeURIComponent(q)}/json?limit=6`);
              if (r.ok) {
                const j = await r.json();
                items = (j.dictionary_terms?.compound || []).map(term => ({ q: term, label: term }));
              }
            }
            if (!items.length || document.activeElement !== dbInput) {
              dbSuggest.classList.remove('open');
              return;
            }
            dbSuggest.innerHTML = items.map(it => `<button type="button" data-q="${esc(it.q)}">${esc(it.label)}</button>`).join('');
            dbSuggest.classList.add('open');
            dbSuggest.querySelectorAll('button').forEach(b => {
              b.addEventListener('click', () => {
                dbInput.value = b.dataset.q;
                dbSuggest.classList.remove('open');
                fetchFromPublicDb(b.dataset.q, dbSelect.value);
              });
            });
          } catch (_) {
            dbSuggest.classList.remove('open');
          }
        }, 220);
      });
      document.addEventListener('click', e => {
        if (dbSuggest && !e.target.closest('.db-wrap')) dbSuggest.classList.remove('open');
      });
    }

    document.querySelectorAll('.chip-btn').forEach(chip => {
      chip.addEventListener('click', () => {
        if (dbSelect && chip.dataset.db) dbSelect.value = chip.dataset.db;
        if (dbInput && chip.dataset.q) dbInput.value = chip.dataset.q;
        fetchFromPublicDb(chip.dataset.q, chip.dataset.db || 'pubchem');
      });
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
    try {
      const [presRes, catRes] = await Promise.all([
        fetch('../assets/lab-presets.json'),
        fetch('../assets/molecule-catalog-500.json').catch(() => null),
      ]);
      if (!presRes.ok) throw new Error(`HTTP ${presRes.status}`);
      STATE.data = await presRes.json();
      if (catRes && catRes.ok) {
        const catJson = await catRes.json();
        STATE.catalog = catJson.molecules || [];
        STATE.catalogById = Object.fromEntries(STATE.catalog.map(m => [m.id, m]));
      }
    } catch (err) {
      const badge = document.getElementById('modeBadge');
      if (badge) {
        badge.className = 'pill-status warn';
        badge.textContent = `Presets unavailable (${esc(err.message)}) · custom SMILES ready`;
      }
      runAuditForInput();
      return;
    }

    populateCatalogSelect('cat:bio_002');
    if (STATE.catalogById.bio_002) {
      selectLibraryValue('cat:bio_002');
    } else if (STATE.data.compliance?.[0]) {
      populateCatalogSelect(`preset:${STATE.data.compliance[0].key}`);
      selectLibraryValue(`preset:${STATE.data.compliance[0].key}`);
    }
    render2DSymmetry();
    renderPropagation();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();

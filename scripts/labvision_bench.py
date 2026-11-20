"""Synthetic benchmark for ouroboros.labvision (well ID on rendered 96-well plates).

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python scripts/labvision_bench.py --out /path/to/dir

Writes labvision_bench.md and labvision_bench.json. All numbers are synthetic.
"""

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # must happen before numpy is imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from ouroboros.labvision.decode import decode, locate_well, pixel_to_well  # noqa: E402
from ouroboros.labvision.geometry import apply_h  # noqa: E402
from ouroboros.labvision.keypoints import gt_keypoints, kp_from_gt, kp_from_pixels  # noqa: E402
from ouroboros.labvision.render import render  # noqa: E402
from ouroboros.labvision.spec import all_wells, default_order, label  # noqa: E402
from ouroboros.labvision.wellstate import classify, fit_centroids, patch_features  # noqa: E402

HEADER = "SYNTHETIC DATA — not indicative of real-lab accuracy."
KINDS = ("rect", "circ")
PERSP = {"rect": (0.0, 0.05, 0.1), "circ": (0.0, 0.03)}
SIGMAS = (0, 1, 2, 4)
SOURCES = ("pixel",) + tuple(f"gt_sigma{s}" for s in SIGMAS)
SWEEP = (0, 15, 45, 90, 135, 180, 225, 270, 315)
SIZE = 256


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def pitch_px(scene) -> float:
    """Ground-truth pitch in pixels near the plate centre."""
    spec = scene.spec
    if scene.kind == "rect":
        c = np.array([spec.width / 2, spec.height / 2])
        return float(np.linalg.norm(apply_h(scene.H, c + [spec.pitch, 0]) - apply_h(scene.H, c)))
    return spec.pitch / spec.radius * gt_keypoints(scene).radius


def decoded_pitch_px(dec, spec) -> float:
    """Pitch in pixels near the plate centre, from the decoded pose only."""
    c = np.array([spec.width / 2, spec.height / 2]) if dec.kind == "rect" else np.zeros(2)
    return float(np.linalg.norm(apply_h(dec.H, c + [spec.pitch, 0]) - apply_h(dec.H, c)))


def evaluate(scene, dec, latency=None) -> dict:
    """Per-image record. Raw counts ignore the ambiguous flag; aggregation applies it."""
    spec, order = scene.spec, default_order(scene.spec)
    gt = apply_h(scene.H, all_wells(spec, order=order))
    rec = {
        "rot": scene.rot_deg,
        "persp": scene.persp,
        "occluded": scene.occluded,
        "failed": dec is None,
        "ambiguous": True,
        "mirrored": False,
        "n_correct": 0,
        "hit65": False,
        "centre_err": None,
        "latency_ms": latency,
    }
    if dec is None:
        return rec
    hits = [pixel_to_well(dec, gt[j], spec, order=order) == j + 1 for j in range(96)]
    pred = np.array([locate_well(dec, i, spec, order=order) for i in range(1, 97)])
    err = np.linalg.norm(pred - gt, axis=1) / pitch_px(scene)
    rec.update(
        ambiguous=bool(dec.ambiguous),
        mirrored=bool(dec.mirrored),
        n_correct=int(sum(hits)),
        hit65=bool(err[64] <= 0.25),
        centre_err=float(np.median(err)),
    )
    return rec


def safe_decode(img, kp_fn, spec):
    try:
        return decode(img, kp_fn(), spec)
    except (np.linalg.LinAlgError, ValueError, IndexError):
        return None


def summarise(recs: list[dict]) -> dict:
    n = len(recs)
    ok = [r for r in recs if not r["ambiguous"]]
    errs = [r["centre_err"] for r in ok if r["centre_err"] is not None]
    lat = [r["latency_ms"] for r in recs if r["latency_ms"] is not None]
    out = {
        "n_images": n,
        "well_id_acc": sum(r["n_correct"] for r in ok) / (96 * n) if n else None,
        "all_correct_rate": sum(r["n_correct"] == 96 for r in ok) / n if n else None,
        "well65_hit_rate": sum(r["hit65"] for r in ok) / n if n else None,
        "acc_unflagged": sum(r["n_correct"] for r in ok) / (96 * len(ok)) if ok else None,
        "flag_rate": 1 - len(ok) / n if n else None,
        "mirrored_rate": sum(r["mirrored"] for r in recs) / n if n else None,
        "failed": sum(r["failed"] for r in recs),
        "centre_err_median_pitch": float(np.median(errs)) if errs else None,
    }
    if lat:
        out["latency_ms_median"] = float(np.median(lat))
        out["latency_ms_p95"] = float(np.percentile(lat, 95))
    return out


def run_image(kind, seed, i, persp, rot=None, sources=SOURCES):
    rng = np.random.default_rng([seed, KINDS.index(kind), i])
    occ = bool(rng.random() < 0.5)
    scene = render(kind, rng, size=SIZE, rot_deg=rot, persp=persp, occlude=occ)
    out = {}
    t0 = time.perf_counter()
    dec = safe_decode(scene.image, lambda: kp_from_pixels(scene.image, kind), scene.spec)
    out["pixel"] = (evaluate(scene, dec, 1e3 * (time.perf_counter() - t0)), dec)
    for s in SIGMAS:
        name = f"gt_sigma{s}"
        if name in sources:
            nrng = np.random.default_rng([seed, KINDS.index(kind), i, 1000 + s])
            d = safe_decode(scene.image, lambda s=s, r=nrng: kp_from_gt(scene, s, r), scene.spec)
            out[name] = (evaluate(scene, d), d)
    return scene, out


def wellstate_model(kind, seed, n):
    feats, labels = [], []
    for i in range(n):
        rng = np.random.default_rng([seed + 7919, KINDS.index(kind), i])
        scene = render(
            kind,
            rng,
            size=SIZE,
            persp=PERSP[kind][i % len(PERSP[kind])],
            occlude=bool(rng.random() < 0.5),
        )
        gt = apply_h(scene.H, all_wells(scene.spec, order=default_order(scene.spec)))
        feats.append(
            patch_features(
                scene.image, gt, 0.6 * scene.spec.well_r / scene.spec.pitch * pitch_px(scene)
            )
        )
        labels.append(scene.states)
    return fit_centroids(np.concatenate(feats), np.concatenate(labels))


def state_acc(scene, model, centres, radius) -> float:
    pred = classify(patch_features(scene.image, centres, radius), model)
    return float(np.mean(pred == scene.states))


def bench(n, sweep_n, calib_n, seed) -> dict:
    res = {
        "header": HEADER,
        "config": {
            "seed": seed,
            "images_per_kind": n,
            "sweep_images_per_angle": sweep_n,
            "wellstate_calib_images": calib_n,
            "image_size_px": SIZE,
            "cpu": cpu_model(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "threads": {
                v: os.environ.get(v)
                for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    }
    t_start = time.perf_counter()
    for kind in KINDS:
        recs = {s: [] for s in SOURCES}
        model = wellstate_model(kind, seed, calib_n)
        ws_gt, ws_dec = [], []
        for i in range(n):
            scene, out = run_image(kind, seed, i, PERSP[kind][i % len(PERSP[kind])])
            for s, (rec, _) in out.items():
                recs[s].append(rec)
            spec, order = scene.spec, default_order(scene.spec)
            radius = 0.6 * spec.well_r / spec.pitch * pitch_px(scene)
            ws_gt.append(
                state_acc(scene, model, apply_h(scene.H, all_wells(spec, order=order)), radius)
            )
            dec = out["pixel"][1]
            if dec is not None and not dec.ambiguous:
                pred = np.array([locate_well(dec, j, spec, order=order) for j in range(1, 97)])
                r_dec = 0.6 * spec.well_r / spec.pitch * decoded_pitch_px(dec, spec)
                ws_dec.append(state_acc(scene, model, pred, r_dec))
        pix = recs["pixel"]
        breakdown = {}
        for r in pix:
            key = f"persp={r['persp']}|occluded={r['occluded']}|bin={int(r['rot'] // 30) * 30}"
            breakdown.setdefault(key, []).append(r)
        sweep = {}
        for a in SWEEP:
            srecs = [
                run_image(kind, seed + 1, 10_000 + a * 1000 + j, 0.0, rot=a, sources=("pixel",))[1][
                    "pixel"
                ][0]
                for j in range(sweep_n)
            ]
            sweep[str(a)] = summarise(srecs)
        res[kind] = {
            "well65_label": label(scene.spec, 65, order=default_order(scene.spec)),
            "sources": {s: summarise(v) for s, v in recs.items()},
            "breakdown_pixel": {k: summarise(v) for k, v in sorted(breakdown.items())},
            "sweep_pixel": sweep,
            "wellstate": {
                "acc_at_gt_centres": float(np.mean(ws_gt)),
                "acc_at_decoded_centres_unflagged": float(np.mean(ws_dec)) if ws_dec else None,
                "n_unflagged_images": len(ws_dec),
            },
        }
    res["config"]["wall_time_s"] = round(time.perf_counter() - t_start, 1)
    return res


def _f(x, pct=True):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def to_markdown(res) -> str:
    c = res["config"]
    lines = [
        "# LabVision well-ID benchmark",
        "",
        f"**{HEADER}**",
        "",
        f"CPU: {c['cpu']} · single thread · image {c['image_size_px']} px · seed {c['seed']}"
        f" · {c['images_per_kind']} images per kind · wall time {c['wall_time_s']} s",
        "",
        "Ambiguous (flagged) images count as wrong in the headline columns.",
        "",
    ]

    def table(title, head, rows):
        lines.extend(
            [
                f"## {title}",
                "",
                f"_{HEADER}_",
                "",
                "| " + " | ".join(head) + " |",
                "|" + "---|" * len(head),
            ]
        )
        lines.extend("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
        lines.append("")

    rows = []
    for kind in KINDS:
        for s, m in res[kind]["sources"].items():
            rows.append(
                [
                    kind,
                    s,
                    _f(m["well_id_acc"]),
                    _f(m["all_correct_rate"]),
                    _f(m["well65_hit_rate"]),
                    _f(m["acc_unflagged"]),
                    _f(m["flag_rate"]),
                    _f(m["mirrored_rate"]),
                    _f(m["centre_err_median_pitch"], False),
                ]
            )
    table(
        "Headline (pixel source) and GT-keypoint diagnostics",
        [
            "plate",
            "source",
            "well-ID acc",
            "all-96 correct",
            "well-65 hit",
            "acc unflagged",
            "flag rate",
            "mirrored",
            "centre err (pitch)",
        ],
        rows,
    )
    rows = [
        [
            k,
            res[k]["well65_label"],
            f"{res[k]['sources']['pixel']['latency_ms_median']:.1f}",
            f"{res[k]['sources']['pixel']['latency_ms_p95']:.1f}",
        ]
        for k in KINDS
    ]
    table("Latency (kp_from_pixels + decode)", ["plate", "well 65", "median ms", "p95 ms"], rows)
    for kind in KINDS:
        bins = list(range(0, 360, 30))
        rows = []
        for p in PERSP[kind]:
            for occ in (False, True):
                row = [p, occ]
                for b in bins:
                    m = res[kind]["breakdown_pixel"].get(f"persp={p}|occluded={occ}|bin={b}")
                    row.append(f"{_f(m['well_id_acc'])} ({m['n_images']})" if m else "–")
                rows.append(row)
        table(
            f"Breakdown, {kind}, pixel source: well-ID acc (n) by 30° rotation bin",
            ["persp", "occluded"] + [f"{b}°" for b in bins],
            rows,
        )
    rows = []
    for kind in KINDS:
        for a, m in res[kind]["sweep_pixel"].items():
            rows.append(
                [
                    kind,
                    f"{a}°",
                    m["n_images"],
                    _f(m["well_id_acc"]),
                    _f(m["all_correct_rate"]),
                    _f(m["flag_rate"]),
                ]
            )
    table(
        "Rotation sweep (perspective 0, occluder p=0.5, pixel source)",
        ["plate", "rotation", "n", "well-ID acc", "all-96 correct", "flag rate"],
        rows,
    )
    rows = [
        [
            k,
            _f(res[k]["wellstate"]["acc_at_gt_centres"]),
            _f(res[k]["wellstate"]["acc_at_decoded_centres_unflagged"]),
            res[k]["wellstate"]["n_unflagged_images"],
        ]
        for k in KINDS
    ]
    table(
        "Well-state stand-in (mean/std nearest centroid, separate calibration seed)",
        ["plate", "acc at GT centres", "acc at decoded centres (unflagged)", "n images"],
        rows,
    )
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300, help="random-set images per plate kind")
    ap.add_argument("--sweep-n", type=int, default=20, help="images per sweep angle")
    ap.add_argument("--calib-n", type=int, default=40, help="well-state calibration images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    res = bench(args.n, args.sweep_n, args.calib_n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labvision_bench.json").write_text(json.dumps(res, indent=1))
    (args.out / "labvision_bench.md").write_text(to_markdown(res))
    for kind in KINDS:
        m = res[kind]["sources"]["pixel"]
        print(
            f"{kind}: well-ID acc {_f(m['well_id_acc'])}, all-96 {_f(m['all_correct_rate'])}, "
            f"well-65 hit {_f(m['well65_hit_rate'])}, flag rate {_f(m['flag_rate'])}, "
            f"latency median {m['latency_ms_median']:.1f} ms  [{HEADER}]"
        )
    return res


if __name__ == "__main__":
    main()

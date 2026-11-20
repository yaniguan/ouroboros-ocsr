# Lab vision: well identification on 96-well plates

`ouroboros/labvision` maps a well index to a pixel, and a pixel to a well index, from one
top-view image of a 96-well plate. It handles any rotation, mild perspective, an occluder
and glare. It uses numpy only: no training, no OpenCV, no torch. It is a prototype tested
on **synthetic images only**.

## Conventions

Indices are 1-based. You must always pass an `IndexOrder`; there is no default, and an
order that does not fit the plate kind raises an error.

| Plate | Order | Well 65 |
|---|---|---|
| Rect 8×12 (SBS) | `ROW_MAJOR` (A1, A2, ... A12, B1, ...) | **F5** |
| Rect 8×12 (SBS) | `COL_MAJOR` (A1, B1, ... H1, A2, ...; Opentrons order) | **A9** |
| Circular | `RING_MAJOR` (inner ring first, slots clockwise from the notch) | **R4S04** |

- Rect geometry follows ANSI/SLAS 4-2004: 9 mm pitch, A1 at (14.38, 11.24) mm.
- Circular plate: rings of 6, 12, 18, 24 and 36 slots at 0.25 to 0.85 of the plate radius.
  Slot 0 lies on the notch ray, so R4S04 is on the outer ring, 40° clockwise of the notch.
- Angles are measured clockwise in the image, starting from "up".

## Pipeline

1. **Keypoints** (`keypoints.py`), from an Otsu plate mask.
   - Rect: the extreme mask pixels along the 4 diagonals of the mask's PCA frame.
   - Circ: the farthest mask pixel on each of 360 rays, then a Kåsa circle fit. The worst
     10% of points are dropped and the circle is refit.
   - `kp_from_gt` adds noise to ground-truth keypoints. It is for diagnosis only.
2. **Decode** (`decode.py`).
   - Rect: sort the corners by angle; the long edges are the 12-column axis. This leaves 2
     pose candidates (4 if the side ratio is below 1.15). Fit a normalised-DLT homography
     for each candidate, then pick the one with a dark A1 dot.
   - Circ: unwrap the image to polar coordinates (64×720). The phase of the 6-fold pattern
     on the inner ring gives a coarse angle; the 36-fold phase on the outer ring refines it.
     This fixes the angle up to a multiple of 60°.
3. **Inverse map**: `pixel_to_well` snaps a pixel to the nearest well. It returns `None`
   if the pixel is more than 0.5 pitch from any well centre.
4. **Well state** (`wellstate.py`) is a placeholder classifier: (mean, std) per well,
   then nearest centroid.

## How symmetry is broken

- **Rect, 180° turn:** the A1 corner has a chamfer and a dark dot. The dot is scored
  against its own local surround. This makes it harder for glare or a dark occluder to
  fake a mark, but it does not rule it out: a few occluded rect images are decoded wrongly
  without a flag.
- **Circ, 60° steps:** the rim notch, tested at the 6 candidate angles.
- **Circ, mirror image:** a chiral dot sits at notch + 20° clockwise. If it shows up on
  the other side, the decode sets both `mirrored` and `ambiguous`. Decoding with
  `CircSpec(clockwise=False)` reads a flipped plate correctly.
- **Margin test:** margin = min(best − second, best) / σ_MAD. If margin < 3, the decode
  sets `ambiguous`. The benchmark counts every well in an ambiguous image as wrong.
- The lattice phase is equivariant: rolling the polar image by n bins shifts it by exactly
  n·Δθ (tested). This is a fixed, hand-made canonicalisation (Kaba et al.); no network yet.

## Run

```
python -m pytest tests/test_labvision_*.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/labvision_bench.py --out runs/labvision
```

The benchmark writes `labvision_bench.md` and `.json` (about 50 s).

## Measured numbers

**SYNTHETIC DATA — not indicative of real-lab accuracy.**

Setup: rendered 256 px images, 300 per plate kind, seed 0, single thread, CPU "Intel(R)
Xeon(R) Processor". Rotation is uniform in [0, 360); occluder p = 0.5; glare always on;
perspective {0, 0.05, 0.1} for rect and {0, 0.03} for circ.

| Plate | Source | Well-ID acc | All-96 correct | Well-65 hit | Acc unflagged | Flag rate | Median / p95 latency |
|---|---|---|---|---|---|---|---|
| rect | pixels | 97.2% | 93.7% | 94.0% | 99.2% | 2.0% | 2.7 / 10.4 ms |
| circ | pixels | 97.0% | 97.0% | 97.0% | 100.0% | 3.0% | 10.5 / 30.3 ms |
| rect | GT keypoints, σ = 2 px | 75.2% | 72.0% | 62.0% | 99.8% | 24.7% | – |
| circ | GT keypoints, σ = 2 px | 58.7% | 57.3% | 50.7% | 99.0% | 40.7% | – |

- Latency was measured on a shared 8-core machine under load. A second run gave
  2.6 / 4.7 ms (rect) and 9.0 / 21.2 ms (circ); accuracy numbers were identical.
- On unflagged images, the median well-centre error is 0.066 pitch (rect) and 0.017
  pitch (circ). The well-state placeholder scores 81.5% / 85.6% at ground-truth centres
  and 74.8% / 74.1% at decoded centres (rect / circ).
- Almost all failures come with an occluder. The flag catches most of them; a few rect
  images are still decoded wrongly without a flag.
- Accuracy drops fast as keypoint noise grows: the fiducial dots are only about 6 px
  across, so a 2 px keypoint error moves the scoring window off the dot. At σ = 2 px,
  wrong-but-confident decodes stay rare (unflagged accuracy is 99% or more), but the flag
  rate rises. At σ = 4 px, unflagged accuracy falls to 81.0% (rect) and 86.7% (circ).

Two bugs were fixed during development, before the benchmark was run: (1) the coarse
6-fold phase used all radii; it now uses only the inner ring, the only 6-fold lattice;
(2) the rect A1 keypoint now uses the chamfer midpoint, where the mask extreme lies.

## Known gaps before real-camera use

- Synthetic data only: no real lighting, lens distortion, plate lids, labels or
  condensation.
- No ellipse rectification: circ perspective turns circles into ellipses and the unwrap
  degrades.
- An occluder on the plate edge corrupts the mask extremes; there is no 3-corner fallback.
- Rect mirror images are not detected.
- The fiducials are tiny, which makes the decode sensitive to keypoint error.
- The well-state classifier is a placeholder.
- Only 96-well plates are supported: no 384/1536, no multi-plate scenes, no 3D pose.

## Next steps

- An escnn OrientationNet prior to propose the plate angle and mark location, kept outside
  `ouroboros/labvision`.
- A learned keypoint head to replace the Otsu extremes, so the decode is robust to occluded
  edges and corners.

## References

- escnn: https://github.com/QUVA-Lab/escnn
- Kaba et al., equivariance with learned canonicalisation: https://proceedings.mlr.press/v202/kaba23a.html
- Esteves et al., Polar Transformer Networks: https://arxiv.org/abs/1709.01889
- ANSI/SLAS 4-2004 well positions: https://www.slas.org/slas/assets/file/public/standards/ansi_slas_4-2004_wellpositions.pdf
- Opentrons labware definitions: https://docs.opentrons.com/ot-2/labware/definitions/
- RoboCulture (plate corners + homography for well targeting): https://arxiv.org/abs/2505.14941

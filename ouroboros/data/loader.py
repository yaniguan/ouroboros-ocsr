"""Map-style dataset over WebDataset tar shards, with exact (resumable) sample order.

The shards follow the WebDataset layout (``{key}.png`` + ``{key}.json`` per sample), so they
also stream with ``webdataset.WebDataset``. For training we instead index the tar members once
(offsets cached in ``{shard}.idx.json``) and read samples by random access. That makes the data
order a pure function of (seed, epoch, position), so a resumed run consumes exactly the samples
the uninterrupted run would have (Phase 2 resume test).

Image convention: ``ink = 1 - gray / 255`` in [0, 1], background = 0. Zero background means
rotating an image with zero fill introduces no artificial border.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from ouroboros.decode.tokenizer import SmilesTokenizer


def index_shard(path: str | Path) -> list[dict]:
    """Return ``[{key, png: (offset, size), json: (offset, size)}, ...]``; cached on disk."""
    path = Path(path)
    idx_path = path.with_suffix(".idx.json")
    if idx_path.exists() and idx_path.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(idx_path.read_text())
    samples: dict[str, dict] = {}
    with tarfile.open(path, "r:") as tar:
        for m in tar:
            key, ext = m.name.split(".", 1)
            samples.setdefault(key, {"key": key})[ext] = (m.offset_data, m.size)
    out = [s for s in samples.values() if "png" in s and "json" in s]
    try:
        idx_path.write_text(json.dumps(out))
    except OSError:
        pass  # read-only location (e.g. Drive mount): just don't cache
    return out


def list_shards(root: str | Path, split: str) -> list[Path]:
    shards = sorted(Path(root).glob(f"{split}-*.tar"))
    if not shards:
        raise FileNotFoundError(f"no {split}-*.tar shards under {root}")
    return shards


class ShardDataset(Dataset):
    """Random-access dataset over tar shards. ``max_samples`` takes a prefix (nested sizes)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        tokenizer: SmilesTokenizer | None = None,
        max_samples: int | None = None,
        image_size: int | None = None,
    ):
        self.entries: list[tuple[Path, dict]] = []
        for shard in list_shards(root, split):
            for s in index_shard(shard):
                self.entries.append((shard, s))
                if max_samples is not None and len(self.entries) >= max_samples:
                    break
            if max_samples is not None and len(self.entries) >= max_samples:
                break
        self.tokenizer = tokenizer
        self.image_size = image_size
        self._fh: dict[Path, io.BufferedReader] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _read(self, shard: Path, off_size) -> bytes:
        fh = self._fh.get(shard)
        if fh is None:  # opened lazily per worker process
            fh = self._fh[shard] = open(shard, "rb")  # noqa: SIM115
        fh.seek(off_size[0])
        return fh.read(off_size[1])

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_fh"] = {}
        return d

    def meta(self, i: int) -> dict:
        shard, s = self.entries[i]
        return json.loads(self._read(shard, s["json"]))

    def __getitem__(self, i: int) -> dict:
        shard, s = self.entries[i]
        img = Image.open(io.BytesIO(self._read(shard, s["png"]))).convert("L")
        if self.image_size is not None and img.size[0] != self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        ink = 1.0 - torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
        meta = json.loads(self._read(shard, s["json"]))
        out = {"image": ink[None], "smiles": meta["smiles"], "key": s["key"], "index": i}
        if self.tokenizer is not None:
            out["ids"] = torch.tensor(self.tokenizer.encode(meta["smiles"]), dtype=torch.long)
        return out


class ResumableSampler(Sampler[int]):
    """Infinite stream of indices: a fresh seeded permutation per epoch.

    State = number of samples already yielded; ``set_position`` makes the next yielded index
    identical to the one an uninterrupted run would have produced.
    """

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True):
        self.n, self.seed, self.shuffle = n, seed, shuffle
        self.position = 0

    def set_position(self, position: int) -> None:
        self.position = position

    def _perm(self, epoch: int) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.n)
        g = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
        return torch.randperm(self.n, generator=g)

    def __iter__(self):
        pos = self.position
        while True:
            epoch, off = divmod(pos, self.n)
            perm = self._perm(epoch)
            for j in range(off, self.n):
                yield int(perm[j])
            pos = (epoch + 1) * self.n

    def __len__(self) -> int:  # nominal (one epoch); the stream is infinite
        return self.n


def collate(batch: list[dict], pad_id: int = 0) -> dict:
    images = torch.stack([b["image"] for b in batch])
    out = {
        "image": images,
        "smiles": [b["smiles"] for b in batch],
        "index": torch.tensor([b["index"] for b in batch]),
    }
    if "ids" in batch[0]:
        L = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        for k, b in enumerate(batch):
            ids[k, : len(b["ids"])] = b["ids"]
        out["ids"] = ids
    return out


def rotate_images(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Rotate a batch ``[B, C, H, W]`` counter-clockwise about the image centre (zero fill).

    Multiples of 90 deg use ``torch.rot90`` (exact pixel permutation, needed for equivariance
    tests); other angles use bilinear resampling. Content is kept inside the inscribed circle by
    the renderer, so nothing is cropped.
    """
    out = torch.empty_like(images)
    for k in range(images.shape[0]):
        a = float(angles_deg[k]) % 360.0
        q = a / 90.0
        if abs(q - round(q)) < 1e-9:
            out[k] = torch.rot90(images[k], int(round(q)) % 4, dims=(-2, -1))
            continue
        t = math.radians(a)
        c, s = math.cos(t), math.sin(t)
        # affine_grid maps output coords -> input coords (normalized, x right, y down). This
        # theta reproduces torch.rot90(k=1) exactly at t = 90 deg (checked in tests), so both
        # branches share one rotation convention.
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=images.dtype, device=images.device)
        grid = torch.nn.functional.affine_grid(
            theta[None], [1, *images.shape[1:]], align_corners=False
        )
        out[k] = torch.nn.functional.grid_sample(
            images[k : k + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]
    return out

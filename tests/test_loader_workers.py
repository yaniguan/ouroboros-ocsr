"""Forked DataLoader workers must not share the parent's shard file offsets.

Regression for the Oct 30 Colab failure: ``python -m ouroboros.train`` with ``data.num_workers=8``
and ``data.exclude_keys`` set died with ``UnicodeDecodeError ... byte 0x89`` (PNG bytes parsed as
JSON) because the exclusion scan opened shard handles in the parent and forked workers raced on
the inherited, shared file offset.
"""

import io
import json
import multiprocessing as mp
import tarfile

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ouroboros.data.loader import ShardDataset


def _shard(path, n):
    with tarfile.open(path, "w") as tar:
        for i in range(n):
            blob = io.BytesIO()
            # varying sizes so offsets of png/json members interleave irregularly
            Image.new("L", (16 + i % 7, 16 + i % 5), 255 - i % 200).save(blob, format="PNG")
            meta = json.dumps({"smiles": "C" * (1 + i % 9), "i": i, "pad": "x" * (i % 13)})
            for ext, data in (("png", blob.getvalue()), ("json", meta.encode())):
                item = tarfile.TarInfo(f"s_{i:05d}.{ext}")
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_forked_workers_do_not_share_parent_file_offsets(tmp_path):
    torch.set_num_threads(1)
    n = 400
    _shard(tmp_path / "train-000000.tar", n)
    # exclude_key14 makes __init__ read every sample's metadata -> handles open in the parent
    ds = ShardDataset(tmp_path, "train", image_size=16, exclude_key14=frozenset({"NOPE"}))
    assert ds._fh, "precondition: parent process holds an open shard handle"
    expected = [ds.meta(i)["i"] for i in range(n)]
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        multiprocessing_context="fork",
        collate_fn=lambda b: b[0],
    )
    for _ in range(3):
        got = {item["index"]: item["smiles"] for item in loader}
        assert got == {i: "C" * (1 + expected[i] % 9) for i in range(n)}
    # parent handles still work after the workers are gone
    assert ds.meta(n - 1)["i"] == n - 1

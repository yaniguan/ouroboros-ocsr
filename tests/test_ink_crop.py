import math
import numpy as np
import pytest
from PIL import Image
from ouroboros.data.real import crop_ink, fit_to_canvas, ingest


def test_blank_transparent_and_margins():
    assert crop_ink(Image.new('L', (90, 60), 255)).size == (90, 60)
    image = Image.new('RGBA', (100, 80), (0, 0, 0, 0))
    image.putpixel((3, 10), (0, 0, 0, 255))
    image.putpixel((75, 60), (245, 245, 245, 255))
    assert crop_ink(image, margin=5).size == (81, 61)
    assert crop_ink(image, threshold=240, margin=0).size == (1, 1)
    with pytest.raises(ValueError):
        crop_ink(image, margin=-1)


def test_crop_canvas_rotation_bounds():
    image = Image.new('L', (500, 300), 255)
    for x in range(200, 240):
        for y in range(100, 120):
            image.putpixel((x, y), 0)
    fitted = fit_to_canvas(image, 128, ink_crop=True)
    assert fitted.size == (128, 128)
    ys, xs = np.where(np.array(fitted) < 128)
    assert max(np.hypot(xs-63.5, ys-63.5)) <= 64
    assert len(xs) > (np.asarray(fit_to_canvas(image, 128)) < 128).sum()


def test_ingestion_shared_policy(tmp_path):
    import json
    import tarfile
    import io
    image = Image.new('L', (100, 80), 255)
    image.paste(0, (35, 20, 65, 50))
    image.save(tmp_path / 'image.png')
    (tmp_path / 'm.jsonl').write_text(json.dumps(dict(image='image.png', smiles='CC'))+'\n')
    ingest(tmp_path / 'm.jsonl', tmp_path / 'out', size=64, ink_crop=True)
    with tarfile.open(next((tmp_path / 'out').glob('*.tar'))) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith('.png'))
        got = Image.open(io.BytesIO(tar.extractfile(member).read()))
        assert np.array_equal(got, fit_to_canvas(image, 64, ink_crop=True))


def test_image_inference_uses_same_policy(tmp_path, monkeypatch):
    import torch
    from scripts import image_to_forcefield as ff
    image = Image.new('RGBA', (130, 80), 'white')
    image.paste('black', (40, 30, 60, 45))
    path = tmp_path / 'drawing.png'
    image.save(path)
    class Model:
        def generate(self, tensor):
            expected = 1 - np.asarray(fit_to_canvas(image, 64, ink_crop=True), dtype=np.float32)/255
            np.testing.assert_allclose(tensor[0, 0].cpu().numpy(), expected)
            return ['CC']
    monkeypatch.setattr(ff, 'load_model', lambda *args: (Model(), 64))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert list(ff.rows_from_images([path], path, 1, 1, ink_crop=True))[0]['smiles'] == 'CC'

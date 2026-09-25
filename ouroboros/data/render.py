"""Render molecules to images with randomized drawing style, and derive labels FROM the drawing.

Label integrity strategy
------------------------
The label is not copied from the source SMILES. It is re-derived from what is drawn:

    mol --2D coords + wedge/hash assignment--> [optional mirror of coordinates]
        --> MolBlock (coords + wedge flags) --> RDKit re-perceives stereo --> label SMILES

The very same MolBlock-derived molecule is what gets drawn (``useMolBlockWedging``), so the
picture, the wedges and the label cannot disagree. For an unmirrored drawing the label must equal
the canonical source SMILES (checked by the generator; mismatches are dropped and counted). For a
mirrored drawing the label becomes the enantiomer, which is what the mirror test verifies.

Geometry of the canvas
----------------------
The molecule is drawn into a centred square of side ``floor(size / sqrt(2))`` so that the drawn
content lies inside the inscribed circle of the ``size x size`` canvas. Any rotation of the final
image about its centre therefore never crops the molecule (needed for rotation augmentation and
the rotation sweep).
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D


def _font_dir() -> Path:
    import matplotlib

    return Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


# Fonts shipped with matplotlib, so the same set exists locally and on Colab.
# "" means RDKit's built-in font.
FONT_NAMES = [
    "",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf",
    "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf",
    "DejaVuSansMono.ttf",
    "STIXGeneral.ttf",
    "STIXGeneralBol.ttf",
    "cmss10.ttf",
]
PALETTES = ["bw", "default", "cdk", "avalon"]


@dataclass
class RenderStyle:
    font: str = ""
    bond_line_width: float = 2.0
    bond_length: float = 25.0  # px at the inner-canvas scale; shrunk automatically if too big
    font_scale: float = 0.6  # RDKit baseFontSize
    palette: str = "bw"
    explicit_methyl: bool = False
    comic: bool = False
    multiple_bond_offset: float = 0.15
    label_padding: float = 0.0
    coordgen: bool = False  # 2D layout engine: CoordGen vs RDKit depictor
    # post-processing (applied to the rasterized image)
    blur: float = 0.0  # Gaussian radius in px, 0 = off
    noise_std: float = 0.0  # additive Gaussian noise, in [0, 1] intensity units
    salt_pepper: float = 0.0  # fraction of pixels flipped
    jpeg_quality: int = 0  # 0 = no JPEG round-trip

    def to_dict(self) -> dict:
        return asdict(self)


def sample_style(rng: random.Random) -> RenderStyle:
    """Draw a random style. Ranges are chosen to span typical publication / patent drawings."""
    return RenderStyle(
        font=rng.choice(FONT_NAMES),
        bond_line_width=rng.uniform(1.0, 4.0),
        bond_length=rng.uniform(18.0, 40.0),
        font_scale=rng.uniform(0.45, 0.9),
        palette=rng.choices(PALETTES, weights=[0.55, 0.25, 0.1, 0.1])[0],
        explicit_methyl=rng.random() < 0.15,
        comic=rng.random() < 0.1,
        multiple_bond_offset=rng.uniform(0.1, 0.25),
        label_padding=rng.uniform(0.0, 0.15),
        coordgen=rng.random() < 0.5,
        blur=rng.uniform(0.3, 1.2) if rng.random() < 0.3 else 0.0,
        noise_std=rng.uniform(0.01, 0.08) if rng.random() < 0.3 else 0.0,
        salt_pepper=rng.uniform(0.0005, 0.005) if rng.random() < 0.15 else 0.0,
        jpeg_quality=rng.randint(30, 90) if rng.random() < 0.3 else 0,
    )


@dataclass
class Rendered:
    image: Image.Image  # mode "L", size x size
    label: str  # canonical isomeric SMILES re-derived from the drawing
    molblock: str  # exactly what was drawn (coords + wedges)


def depict(mol: Chem.Mol, style: RenderStyle, mirror: bool = False) -> Chem.Mol:
    """Return the molecule to draw: 2D coords, kekulized, wedges assigned, re-read from MolBlock.

    With ``mirror=True`` the 2D coordinates are reflected (x -> -x) while wedge/hash flags are
    kept, i.e. exactly what a mirrored drawing shows. Stereo is then re-perceived from the
    mirrored drawing, which inverts every tetrahedral center (E/Z is unchanged by a reflection).
    """
    m = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(style.coordgen)
    rdDepictor.Compute2DCoords(m)
    m = rdMolDraw2D.PrepareMolForDrawing(m, kekulize=True, addChiralHs=True, wedgeBonds=True)
    if mirror:
        conf = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, Point3D(-p.x, p.y, p.z))
    mb = Chem.MolToMolBlock(m, kekulize=True)
    drawn = Chem.MolFromMolBlock(mb, sanitize=True, removeHs=False)
    if drawn is None:
        raise ValueError("MolBlock round-trip failed")
    return drawn


def label_of(drawn: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(drawn))


def _draw(drawn: Chem.Mol, style: RenderStyle, inner: int) -> Image.Image:
    d = rdMolDraw2D.MolDraw2DCairo(inner, inner)
    o = d.drawOptions()
    o.prepareMolsBeforeDrawing = False
    o.useMolBlockWedging = True
    o.bondLineWidth = style.bond_line_width
    o.scaleBondWidth = False
    o.fixedBondLength = style.bond_length
    o.baseFontSize = style.font_scale
    o.explicitMethyl = style.explicit_methyl
    o.comicMode = style.comic
    o.multipleBondOffset = style.multiple_bond_offset
    o.additionalAtomLabelPadding = style.label_padding
    o.padding = 0.03
    o.addStereoAnnotation = False  # never leak CIP labels into the image
    o.includeChiralFlagLabel = False
    if style.font:
        o.fontFile = str(_font_dir() / style.font)
    {
        "bw": o.useBWAtomPalette,
        "default": o.useDefaultAtomPalette,
        "cdk": o.useCDKAtomPalette,
        "avalon": o.useAvalonAtomPalette,
    }[style.palette]()
    mol = Chem.Mol(drawn)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("L")


def _postprocess(img: Image.Image, style: RenderStyle, rng: np.random.Generator) -> Image.Image:
    if style.blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(style.blur))
    if style.noise_std > 0 or style.salt_pepper > 0:
        a = np.asarray(img, dtype=np.float32) / 255.0
        if style.noise_std > 0:
            a = a + rng.normal(0.0, style.noise_std, a.shape).astype(np.float32)
        if style.salt_pepper > 0:
            flip = rng.random(a.shape) < style.salt_pepper
            a[flip] = rng.integers(0, 2, int(flip.sum())).astype(np.float32)
        img = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), mode="L")
    if style.jpeg_quality > 0:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=style.jpeg_quality)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def render(
    mol: Chem.Mol,
    style: RenderStyle,
    size: int = 384,
    mirror: bool = False,
    seed: int = 0,
) -> Rendered:
    """Render ``mol`` to a ``size x size`` grayscale image, inside the inscribed circle."""
    drawn = depict(mol, style, mirror=mirror)
    inner = int(math.floor(size / math.sqrt(2)))
    inner -= inner % 2  # keep the paste offset integral and the layout symmetric
    canvas = Image.new("L", (size, size), 255)
    off = (size - inner) // 2
    canvas.paste(_draw(drawn, style, inner), (off, off))
    img = _postprocess(canvas, style, np.random.default_rng(seed))
    return Rendered(image=img, label=label_of(drawn), molblock=Chem.MolToMolBlock(drawn))


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()

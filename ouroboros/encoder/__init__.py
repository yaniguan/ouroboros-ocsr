"""Image encoders. All arms emit an ``EncoderOutput`` consumed by the shared decoder."""

from ouroboros.encoder.base import EncoderOutput, ImageEncoder

__all__ = ["EncoderOutput", "ImageEncoder"]

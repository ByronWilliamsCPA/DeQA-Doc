"""Image encoding utilities for VLM API calls."""

from __future__ import annotations

import base64
from io import BytesIO


def encode_image_base64(
    image_path: str,
    max_pixels: int = 1024,
) -> tuple[str, str]:
    """Load, resize, and base64-encode an image for API transmission.

    Resizes the longest edge to ``max_pixels`` to reduce API costs while
    preserving enough detail for quality assessment.

    Args:
        image_path: Absolute path to the image file.
        max_pixels: Maximum size for the longest edge.

    Returns:
        Tuple of (base64_data, media_type).
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    w, h = image.size
    if max_pixels > 0 and max(w, h) > max_pixels:
        scale = max_pixels / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_data, "image/jpeg"

"""Message content that is more than text: images.

Inside MUYAH-CODE a message's content is either a string or a list of parts in the OpenAI format:
    [{"type": "text", "text": "..."},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}]
The Anthropic client turns image parts into Claude image blocks. Models that cannot see images never get
them: a short note says an image was left out (see `Agent._images_message`).

Images come from Read on an image file, MCP tools that return screenshots (e.g. the browser), and
@image.png mentions in your prompt (dragging a file into the terminal pastes its path, which works too).
"""

from __future__ import annotations

import base64
import re
import struct
from pathlib import Path

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024        # the strictest provider limit (Anthropic)
IMAGE_TOKENS = 1100                      # rough cost of one image in the context (varies by provider/size)
VISION_NAMES = re.compile(r"(llava|bakllava|-vl\b|-vl-|vision|pixtral|gemma-?3|minicpm-v|moondream|qwen2\.5-vl|"
                          r"qwen3-vl|internvl|molmo|llama-?4|gpt-4o|gpt-4\.1|gpt-5|claude|gemini)", re.IGNORECASE)


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_TYPES


def image_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from PNG / GIF / JPEG / WebP headers (no imaging library needed)."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", data[16:24])
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", data[6:10])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if chunk == b"VP8L":
                b = data[21:25]
                return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
            if chunk == b"VP8X":
                return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except (struct.error, IndexError):
        return None
    return None


def load_image(path: Path) -> tuple[str, str, bytes]:
    """(media type, base64, raw bytes). ValueError if it is not a supported image or is too big."""
    media = IMAGE_TYPES.get(path.suffix.lower())
    if not media:
        raise ValueError(f"{path.name} is not a PNG, JPEG, GIF or WebP image")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{path.name} is {len(data) / 2**20:.1f} MB; images must be under 5 MB")
    return media, base64.b64encode(data).decode("ascii"), data


def image_part(media_type: str, b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}


def parse_data_url(url: str) -> tuple[str, str] | None:
    m = re.match(r"data:([\w/+.-]+);base64,(.*)", url or "", re.DOTALL)
    return (m.group(1), m.group(2)) if m else None


def describe(path_or_label: str, data: bytes) -> str:
    size = image_size(data)
    dims = f"{size[0]}×{size[1]}, " if size else ""
    return f"{path_or_label} ({dims}{len(data) / 1024:.0f} KB)"


def text_of(content) -> str:
    """The text of any content (image parts become "[image]")."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                out.append(p.get("text", ""))
            elif isinstance(p, dict) and p.get("type") in ("image_url", "image"):
                out.append("[image]")
        return "\n".join(out)
    return str(content or "")


def count_images(content) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") in ("image_url", "image"))


def strip_images(content, note: str = "[image removed to save context]"):
    """The same content with every image replaced by a short note."""
    if not count_images(content):
        return content
    return [p if not (isinstance(p, dict) and p.get("type") in ("image_url", "image"))
            else {"type": "text", "text": note} for p in content]


def model_sees_images(model: str, table_says: bool | None, setting) -> bool:
    """`vision` setting true/false wins; otherwise the price table, then the model's name."""
    if isinstance(setting, bool):
        return setting
    if isinstance(setting, str) and setting.lower() in ("true", "false", "on", "off", "yes", "no"):
        return setting.lower() in ("true", "on", "yes")
    if table_says is not None:
        return table_says
    return bool(VISION_NAMES.search(model or ""))

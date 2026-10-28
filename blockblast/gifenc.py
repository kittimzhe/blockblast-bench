"""Minimal animated-GIF writer (GIF89a + LZW), pure stdlib.

Only what the benchmark needs: indexed frames with one global palette
(<= 16 colours), per-frame delays and infinite looping. Correctness is
pixel-verified against Pillow's decoder in the test suite.
"""

from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

MAX_CODE = 4095  # 12-bit LZW ceiling
MAX_DELAY_CS = 0xFFFF  # GIF stores the frame delay in one 16-bit field


class _BitPacker:
    """LSB-first bit packer used by GIF's LZW stream."""

    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._nbits = 0

    def put(self, code: int, width: int) -> None:
        self._acc |= code << self._nbits
        self._nbits += width
        while self._nbits >= 8:
            self._out.append(self._acc & 0xFF)
            self._acc >>= 8
            self._nbits -= 8

    def finish(self) -> bytes:
        if self._nbits:
            self._out.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._out)


def lzw_encode(indices: Sequence[int], min_code_size: int) -> bytes:
    """GIF-flavoured LZW with variable code width and dictionary resets."""
    clear = 1 << min_code_size
    eoi = clear + 1
    packer = _BitPacker()
    width = min_code_size + 1
    next_code = eoi + 1
    table = {}

    packer.put(clear, width)
    it = iter(indices)
    try:
        prefix = next(it)
    except StopIteration:
        packer.put(eoi, width)
        return packer.finish()

    for symbol in it:
        key = (prefix << 8) | symbol
        found = table.get(key)
        if found is not None:
            prefix = found
            continue
        packer.put(prefix, width)
        if next_code <= MAX_CODE:
            table[key] = next_code
            next_code += 1
            # the decoder widens after assigning entry (1 << width) - 1,
            # which corresponds to our next_code hitting (1 << width) + 1
            if next_code == (1 << width) + 1 and width < 12:
                width += 1
        else:
            packer.put(clear, width)
            table.clear()
            next_code = eoi + 1
            width = min_code_size + 1
        prefix = symbol

    packer.put(prefix, width)
    packer.put(eoi, width)
    return packer.finish()


def _sub_blocks(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i : i + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0)
    return bytes(out)


def write_gif(
    path,
    size: Tuple[int, int],
    palette: List[Tuple[int, int, int]],
    frames: Sequence[Sequence[int]],
    delays_cs: Sequence[int],
    loop: int = 0,
) -> None:
    """Write an animated GIF.

    ``frames`` are palette-index buffers of length width*height;
    ``delays_cs`` are per-frame delays in 1/100 s; ``loop=0`` = forever.
    Delays are clamped into the 1..65535 range that GIF's 16-bit field can
    represent, so an extreme ``--speed`` cannot surface as a struct error.
    """
    width, height = size
    if not frames:
        raise ValueError("no frames to write")
    if len(frames) != len(delays_cs):
        raise ValueError("frames and delays length mismatch")
    if not 2 <= len(palette) <= 16:
        raise ValueError("palette must hold 2..16 colours")

    # pad palette to 16 entries -> GCT size field 3, LZW min code size 4
    pal = list(palette) + [(0, 0, 0)] * (16 - len(palette))
    min_code_size = 4

    out = bytearray()
    out.extend(b"GIF89a")
    out.extend(struct.pack("<HHBBB", width, height, 0b1_111_0_011, 0, 0))
    for r, g, b in pal:
        out.extend((r, g, b))

    # Netscape looping extension
    out.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01")
    out.extend(struct.pack("<H", loop))
    out.append(0)

    for frame, delay in zip(frames, delays_cs):
        if len(frame) != width * height:
            raise ValueError("frame size mismatch")
        if any(v > 15 for v in frame):
            raise ValueError("palette index out of range")
        # graphic control: disposal=1 (leave frame in place), no transparency
        out.extend(b"\x21\xf9\x04\x04")
        out.extend(struct.pack("<H", min(MAX_DELAY_CS, max(1, int(delay)))))
        out.extend(b"\x00\x00")
        # image descriptor: full-frame, no local colour table
        out.extend(b"\x2c")
        out.extend(struct.pack("<HHHH", 0, 0, width, height))
        out.append(0)
        out.append(min_code_size)
        out.extend(_sub_blocks(lzw_encode(frame, min_code_size)))

    out.append(0x3B)
    with open(path, "wb") as fh:
        fh.write(bytes(out))

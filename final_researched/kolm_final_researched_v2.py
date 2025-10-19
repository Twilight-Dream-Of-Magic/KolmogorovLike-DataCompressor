#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kolm_final_researched_v2.py
===========================

This module implements a deeply researched prototype compressor/decompressor
guided by the "前布尔电路后统计概率" (boolean-circuit first, probability last)
philosophy. The goal is to expose and remove as much structure as possible
via cheap, reversible bitwise transforms before handing the residual to
classical entropy/dictionary coders. All transformations are written from
scratch without external compression libraries.

Compared with earlier revisions, V2 makes two key changes:
- **Chunking modes:** supports both fixed-size chunking and a non-recursive,
  strict **FastCDC** (gear-hash) splitter. The container header uses the
  high bit of the 32-bit "block_size" field as a **mode flag**
  (`0 = FIXED`, `1 = CDC`); the remaining 31 bits store the scale
  (fixed `block_size` or CDC `avg_size`). Decoding does not depend on the
  mode and remains fully compatible.
- **Pipeline hygiene:** removes the invalid **BBWT+LFSR mixing** pipelines.
  The LFSR-based component is retained **only** as a standalone predictor
  model. A **new V2 pipeline** is added to the candidate set.

Features provided:

- **Content-defined chunking (FastCDC):**
  deterministic 256×32-bit gear table (fixed seed), mask selection derived
  from the target average size, linear scan between `[min, max)` with
  forced cut at `max`, and orphan-tail merge to avoid tiny trailing blocks.
- **BBWT line:** Duval’s linear-time Lyndon factorization → bijective BWT
  (BBWT) by merging rotations per factor; then **MTF** to cluster small
  integers (cf. standard BWT/BBWT literature).
- **Reversible bitwise modules** (toggleable inside the BBWT→MTF→Rice line):
  - Bit-plane interleaving (group high-order planes across bytes)
  - Nibble swap (high/low 4-bit exchange; self-inverse)
  - Bit-reversal within byte (via LUT)
  - **Gray code** option
  *(Note: LFSR is **not** a bitwise sub-module in V2; it exists as a
  standalone predictor candidate only.)*
- **Integer coders:** Rice/Golomb for geometric-like symbols; Elias gamma/δ
  for small metadata; ULEB128 for general variable-length integers.
- **Dictionary/grammar models:** a lightweight **LZ77** and a simple
  **Re-Pair** to capture long repeats.
- **Model selection (MDL):** per-block evaluation among
  RAW, XOR, BBWT→MTF→Rice (+ optional bitwise variants), LZ77,
  **LFSR predictor**, **Re-Pair**, and the **V2 new pipeline**; the smallest
  representation is chosen independently for each block.
- **Container format:** magic ``b'KOLR'`` +
  packed mode/size word +
  total original length (u32) +
  number of blocks (u16) +
  for each block: method_id (u8), original_length (u32),
  payload_length (u32), payload bytes.
  *(Tip: u16 limits blocks to ≤ 65,535; switch to ULEB128 if you need more.)*

Operation:

- The encoder works on chunk boundaries (fixed or CDC). Each block is
  transformed and encoded independently; the selected model is recorded
  implicitly by `method_id`. The decoder inverts each block using the same
  ordering of decoders; it does **not** depend on the chunking mode bit.

Usage:
    # Compress (fixed blocks, default 8192 bytes)
    python3 kolm_final_researched_v2.py input.bin

    # Compress with FastCDC (avg_size from --block; min=avg//2, max=2*avg)
    python3 kolm_final_researched_v2.py --FastCDC -b 8192 input.bin

    # Decompress
    python3 kolm_final_researched_v2.py -d input.kolr

    # Built-in experiment
    python3 kolm_final_researched_v2.py --experiment

This file is self-contained and serves as an educational reference for
combining string algorithms (Duval/BBWT/MTF) with bit-level circuit tricks
under an MDL-driven per-block selection strategy.
"""

from __future__ import annotations
import itertools
from collections import Counter
import math
import struct
import random
from typing import List, Tuple, Dict, Callable, Optional, Any

# 全局开关（CLI 会设置）
G_NO_LZ77: bool = False
G_ONLY_METHOD: Optional[str] = None  # 例如 'v2_new' / 'lz77' / 'raw' 等

# 进度开关（由 CLI 设置）
G_PROGRESS: bool = False

def _print_progress(label: str, i: int, n: int, final: bool = False) -> None:
    """简单的块级进度打印：[{label}] block i/n ... / done."""
    if not G_PROGRESS:
        return
    if not final:
        print(f"[{label}] block {i}/{n} ...", end="\r", flush=True)
    else:
        print(f"[{label}] block {n}/{n} done.", flush=True)

###############################################################################
# Utility functions: ULEB128
###############################################################################

def uleb128_encode(n: int) -> bytes:
    """Encode a non‑negative integer into unsigned LEB128."""
    if n < 0:
        raise ValueError("ULEB128 only supports unsigned integers")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def uleb128_decode_stream(data: bytes, pos: int = 0) -> Tuple[int, int]:
    """Decode a ULEB128 value from data starting at position ``pos``."""
    shift = 0
    result = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated ULEB128")
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7

###############################################################################
# FastCDC content‑defined chunking
###############################################################################

# =========================================
# 确定性的 GEAR 表（固定种子，跨平台可复现）
# =========================================
def _make_gear(seed: int = 0x243F6A88) -> List[int]:
    """
    生成 256 项 32-bit gear 表，基于 xorshift32 的确定性序列。
    为避免出现 0 值，最后与 1 做 OR。
    """
    x = seed & 0xFFFFFFFF
    tbl: List[int] = []
    for _ in range(256):
        # xorshift32
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5)  & 0xFFFFFFFF
        tbl.append((x | 1) & 0xFFFFFFFF)
    return tbl

_GEAR: List[int] = _make_gear()

# =========================================
# FastCDC（严格、非递归）
# =========================================
def _clamp_mask_bits(avg_size: int) -> int:
    if avg_size <= 0:
        return 6
    k = (avg_size).bit_length() - 1
    if k < 6:  k = 6
    if k > 20: k = 20
    return k

def _roll_gear(h: int, byte_val: int) -> int:
    return ((h << 1) & 0xFFFFFFFF) + _GEAR[byte_val]

def cdc_fast_boundaries_strict(data: bytes,
                               min_size: int = 4096,
                               avg_size: int = 8192,
                               max_size: int = 16384,
                               merge_orphan_tail: bool = True) -> List[Tuple[int, int]]:
    n = len(data)
    if n == 0:
        return []
    if not (min_size > 0 and min_size <= avg_size <= max_size):
        raise ValueError("Require 0 < min_size ≤ avg_size ≤ max_size")
    if avg_size < 64:
        raise ValueError("avg_size too small; use ≥ 64")

    k = _clamp_mask_bits(avg_size)
    mask = (1 << k) - 1

    boundaries: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        start = i
        end_min = min(n, start + min_size)
        end_max = min(n, start + max_size)
        i = end_min
        h = 0
        found = False
        while i < end_max:
            h = _roll_gear(h, data[i])
            if (h & mask) == 0:
                i += 1
                found = True
                break
            i += 1
        if not found:
            i = end_max
        boundaries.append((start, i))

    if merge_orphan_tail and len(boundaries) >= 2:
        last_s, last_e = boundaries[-1]
        if (last_e - last_s) < min_size:
            prev_s, prev_e = boundaries[-2]
            boundaries[-2] = (prev_s, last_e)
            boundaries.pop()

    assert boundaries[0][0] == 0 and boundaries[-1][1] == n
    return boundaries

# =========================================
# 固定大小分块
# =========================================
def fixed_boundaries(data: bytes, block_size: int = 8192) -> List[Tuple[int, int]]:
    n = len(data)
    if n == 0:
        return []
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [(i, min(n, i + block_size)) for i in range(0, n, block_size)]

###############################################################################
# Duval Lyndon factorisation and BBWT
###############################################################################

def duval_lyndon(s: bytes) -> List[Tuple[int, int]]:
    """Compute the Lyndon factorisation of ``s`` using Duval's O(n) algorithm.

    The algorithm greedily finds the next Lyndon word and outputs
    factors in non‑increasing lexicographic order【154237816494091†L294-L301】.  Each factor
    is represented as a (start, end) index pair.
    """
    n = len(s)
    i = 0
    out: List[Tuple[int, int]] = []
    while i < n:
        j = i + 1
        k = i
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        p = j - k
        while i <= k:
            out.append((i, i + p))
            i += p
    return out

def bbwt_forward(s: bytes) -> bytes:
    """Compute the bijective Burrows–Wheeler transform (BBWT) of ``s``.

    We factorise ``s`` into Lyndon words, compute the rotation order of
    each word via a simple suffix array, then perform a k‑way merge of
    rotations.  The resulting last column constitutes the BBWT【555806496076120†L507-L516】.
    """
    if not s:
        return b""
    facs = duval_lyndon(s)
    # local suffix array using prefix doubling for each doubled factor
    def sa_prefix_doubling(t: bytes) -> List[int]:
        n = len(t)
        k = 1
        rank = list(t)
        tmp = [0] * n
        idx = list(range(n))
        while True:
            idx.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
            tmp[idx[0]] = 0
            for j in range(1, n):
                a, b = idx[j - 1], idx[j]
                tmp[b] = tmp[a] + ((rank[a], rank[a + k] if a + k < n else -1) < (rank[b], rank[b + k] if b + k < n else -1))
            rank, tmp = tmp, rank
            if rank[idx[-1]] == n - 1:
                break
            k <<= 1
        return idx
    # compute rotation order per factor
    factors: List[Tuple[bytes, List[int]]] = []
    for a, b in facs:
        w = s[a:b]
        m = len(w)
        ww = w + w
        sa = sa_prefix_doubling(ww)
        rot_order = [p for p in sa if p < m]
        factors.append((w, rot_order))
    # define rotation comparator: compares infinite repetitions
    import heapq
    class Node:
        __slots__ = ("fi", "k", "w", "order")
        def __init__(self, fi: int, k: int, w: bytes, order: List[int]):
            self.fi = fi; self.k = k; self.w = w; self.order = order
        def __lt__(self, other: 'Node') -> bool:
            i = self.order[self.k]
            j = other.order[other.k]
            u, v = self.w, other.w
            m, n = len(u), len(v)
            # compare u[i:] + u[:i] and v[j:] + v[:j]
            p = 0
            while p < m + n:
                cu = u[(i + p) % m]
                cv = v[(j + p) % n]
                if cu != cv:
                    return cu < cv
                p += 1
            return (self.fi, i) < (other.fi, j)
    heap: List[Node] = []
    for fi, (w, order) in enumerate(factors):
        if order:
            heap.append(Node(fi, 0, w, order))
    heapq.heapify(heap)
    out = bytearray()
    while heap:
        nd = heapq.heappop(heap)
        i = nd.order[nd.k]
        w = nd.w
        m = len(w)
        out.append(w[(i - 1) % m])
        nd.k += 1
        if nd.k < len(nd.order):
            heapq.heappush(heap, nd)
    return bytes(out)

def bbwt_inverse(L: bytes) -> bytes:
    """Invert the bijective Burrows–Wheeler transform."""
    n = len(L)
    if n == 0:
        return b""
    order = sorted(range(n), key=lambda idx: (L[idx], idx))
    pi = order[:]
    seen = [False] * n
    cycles: List[List[int]] = []
    for i in range(n):
        if not seen[i]:
            cur = i
            cyc: List[int] = []
            while not seen[cur]:
                seen[cur] = True
                cyc.append(cur)
                cur = pi[cur]
            cycles.append(cyc)
    cycles.sort(key=lambda cyc: min(cyc))
    factors: List[bytes] = []
    for cyc in cycles:
        i0 = min(cyc)
        d = len(cyc)
        cur = i0
        seq: List[int] = []
        for _ in range(d):
            cur = pi[cur]
            seq.append(L[cur])
        factors.append(bytes(seq))
    return b"".join(reversed(factors))

###############################################################################
# Move‑to‑front
###############################################################################

def mtf_encode(data: bytes) -> List[int]:
    table = list(range(256))
    out: List[int] = []
    for b in data:
        idx = table.index(b)
        out.append(idx)
        table.pop(idx)
        table.insert(0, b)
    return out

def mtf_decode(seq: List[int]) -> bytes:
    table = list(range(256))
    out = bytearray()
    for idx in seq:
        b = table[idx]
        out.append(b)
        table.pop(idx)
        table.insert(0, b)
    return bytes(out)

def lfsr_whiten(data: bytes, taps: int = 0b10010110, seed: int = 1) -> bytes:
    """Whiten data using an 8‑bit LFSR with the given taps and seed.

    Each next state bit is the XOR of tapped bits of the current state【879695488005067†L166-L179】.
    The function produces an output by XORing each byte with the evolving
    LFSR state.  Applying the same function twice recovers the original
    sequence.
    """
    state = seed & 0xFF
    out = bytearray()
    for b in data:
        out.append(b ^ state)
        # update LFSR: compute feedback bit
        fb = 0
        for bit in range(8):
            if (taps >> bit) & 1:
                fb ^= (state >> bit) & 1
        state = ((state << 1) & 0xFF) | fb
    return bytes(out)

###############################################################################
# New pipeline model (V2)
###############################################################################

"""
Boolean-circuit toolkit (word/byte level) + reversible byte-wise automata.

Purpose
-------
This module provides a set of *pure Boolean* building blocks (AND/OR/NOT/XOR),
word-level prefix operations, a byte-equality predicate that returns a *mask*
(0xFF or 0x00), and a 2:1 multiplexer. On top of these primitives, it defines
four reversible, byte-wise transformation modes (automata) that map an input
byte sequence to a *residual* sequence under a simple predictor. These mappings
are designed for semi-explainable preprocessing in compression pipelines: they
reduce local redundancy by canceling predictable components while remaining
strictly invertible. Mode 4 adds a run-segment selector that switches the
predictor based on local equality, again expressed with gates and a MUX.

Binary model (selected and assumed throughout)
----------------------------------------------
- **Bit-parallel word model**: operations act on all bits in parallel, using
  Python's integers as bit containers. A configurable bit **width** (default 8
  for bytes) determines the active bit-lane via a mask `(1<<width)-1`.
- **Two's-complement masking**: `NOT` is implemented as bitwise complement
  followed by width-masking to keep results within the lane; left shifts are
  masked to width to emulate fixed-width registers; right shifts are logical.
- **Mask-as-Boolean**: predicates return an all-ones or all-zeros *mask*
  (e.g., `0xFF` or `0x00` for 8-bit) so that decisions can be expressed with
  Boolean algebra only. The 2:1 MUX implements `(m & a) | (~m & b)`.
- **Reversibility**: each forward mode has a matching inverse that reconstructs
  the original sequence exactly, given only the mode identifier.

Engineering notes
-----------------
- Names and docstrings are written to make the circuit semantics explicit.
- No arithmetic beyond shifts and bitwise gates is required for the core logic.
- Time complexity is linear in the input length for all modes (single pass).
- `prefix_or_word` uses a log-doubling schedule (`1,2,4,8,...`) like a
  parallel scan; `left_band` is a convenience band operator built from it.
"""

###############################################################################
# Boolean gate primitives (word/byte) + prefix/band + 2:1 MUX (pure gates)
###############################################################################
from typing import Tuple, Dict


def gate_and(a: int, b: int) -> int:
    """Bitwise AND (pure Boolean gate)."""
    return a & b


def gate_or(a: int, b: int) -> int:
    """Bitwise OR (pure Boolean gate)."""
    return a | b


def gate_not(a: int, width: int = 8) -> int:
    """Bitwise NOT limited to *width* bits via masking.

    Parameters
    ----------
    a : int
        Input word.
    width : int, default 8
        Active bit-lane width. Only the lowest `width` bits are kept.

    Returns
    -------
    int
        Bitwise complement of `a` in the selected lane.
    """
    lane_mask = (1 << width) - 1
    return (~a) & lane_mask


def gate_xor(a: int, b: int, width: int = 8) -> int:
    """Bitwise XOR expressed with gates only: `(a OR b) AND NOT(a AND b)`.

    Notes
    -----
    Using `NOT` with width-masking preserves the fixed-lane model.
    """
    return (a | b) & gate_not((a & b), width)


def prefix_or_word(x: int, width: int) -> int:
    """Intra-word prefix-OR (`scan OR`) using only shifts and ORs.

    For each bit position *i* (0-indexed, LSB), the result bit is `1` iff any
    of the lower-or-equal bits of `x` up to *i* is `1`. Implements a standard
    log-doubling parallel scan (d ∈ {1,2,4,8,...}).

    Parameters
    ----------
    x : int
        Input word.
    width : int
        Active bit-lane width.
    """
    distance: int = 1
    result: int = x & ((1 << width) - 1)
    lane_mask: int = (1 << width) - 1
    while distance < width:
        result = gate_or(result, (result << distance) & lane_mask)
        distance <<= 1
    return result


def left_band(beta: int, L: int, width: int) -> int:
    """`band(beta, L) = prefix_or(beta) & ~prefix_or(beta << L)`.

    Intuition: selects a *left-closed, right-open* band of ones whose length is
    controlled by `L`, built purely from prefix-ORs and masking.
    """
    lane_mask = (1 << width) - 1
    pref = prefix_or_word(beta & lane_mask, width)
    cut = prefix_or_word((beta << L) & lane_mask, width)
    return gate_and(pref, gate_not(cut, width))


def byte_eq(a: int, b: int) -> int:
    """
    Branch-free, gate-only byte equality that returns a canonical mask.

    Returns
    -------
    int
        0xFF if a == b, else 0x00. The result is constructed purely with
        Boolean gates (AND/OR/NOT/XOR) and shifts — no arithmetic on values
        and no data-dependent branches.

    Binary model
    ------------
    - 8-bit lane; inputs are masked to the lane (two's-complement NOT with width).
    - Predicates are expressed as all-ones / all-zeros masks (mask-as-Boolean).

    Implementation (all gates, no branches)
    ---------------------------------------
    1) x := a XOR b  (within the 8-bit lane). If a == b then x == 0.
    2) Reduce to a 1-bit “any_one” flag by OR-folding:
         x1 = x | (x >> 4); x1 = x1 | (x1 >> 2); x1 = x1 | (x1 >> 1)
       Now any_one = x1 & 1  (1 iff any bit of x was 1).
    3) eq_bit := NOT(any_one) in a 1-bit lane → eq_bit ∈ {0,1}.
    4) Replicate eq_bit across the 8-bit lane by OR-ing shifted copies:
         mask = (eq_bit<<0) | (eq_bit<<1) | ... | (eq_bit<<7)
       This yields 0xFF when eq_bit==1, else 0x00.

    Correctness sketch
    ------------------
    - If a == b ⇒ x == 0 ⇒ any_one == 0 ⇒ eq_bit == 1 ⇒ mask == 0xFF.
    - If a != b ⇒ x != 0 ⇒ any_one == 1 ⇒ eq_bit == 0 ⇒ mask == 0x00.

    Complexity
    ----------
    O(1) bit-operations on fixed-width bytes; no data-dependent control flow.
    """
    width = 8
    lane_mask = (1 << width) - 1

    # 1) XOR in-lane
    x = gate_xor(a & lane_mask, b & lane_mask, width)

    # 2) OR-fold to “any bit set”
    x1 = gate_or(x, (x >> 4) & 0x0F)
    x1 = gate_or(x1, (x1 >> 2) & 0x3F)
    x1 = gate_or(x1, (x1 >> 1) & 0x7F)
    any_one = x1 & 1  # 1 if any bit in x was 1

    # 3) 1-bit NOT to get equality bit
    eq_bit = gate_not(any_one, 1) & 1  # ensure ∈ {0,1}

    # 4) Replicate eq_bit to full 8-bit mask by ORing shifted copies
    mask  = 0
    mask  = (eq_bit << 0) & 0xFF
    mask |= (eq_bit << 1) & 0xFF
    mask |= (eq_bit << 2) & 0xFF
    mask |= (eq_bit << 3) & 0xFF
    mask |= (eq_bit << 4) & 0xFF
    mask |= (eq_bit << 5) & 0xFF
    mask |= (eq_bit << 6) & 0xFF
    mask |= (eq_bit << 7) & 0xFF

    return mask


def mux_byte(m: int, a: int, b: int, width: int = 8) -> int:
    """2:1 multiplexer on a byte lane: `m=0xFF` selects `a`; `m=0x00` selects `b`.

    Implemented as `(m & a) | (~m & b)` with width-masked NOT. Non-canonical
    masks (other values) will behave bitwise and may mix bits; callers should
    pass canonical masks (all-ones or all-zeros) for selection semantics.
    """
    return gate_or(gate_and(m, a), gate_and(gate_not(m, width), b))


def _gray_pred(v: int) -> int:
    """Gray-code predictor: `g = v XOR (v >> 1)` built from gates.

    Used as an alternative local predictor in the automata below.
    """
    return gate_xor(v, (v >> 1), 8)


###############################################################################
# Four automaton modes (all reversible; pure gates in the critical path),
# plus the run-segment selection Mode 4.
###############################################################################


def _mode1_forward(block: bytes) -> bytes:
    """Mode 1 (order-1 residual): `y[i] = x[i] XOR x[i-1]`.

    The first byte is copied through to seed the recursion.
    """
    if not block:
        return b""
    length = len(block)
    output_bytes = bytearray(length)
    output_bytes[0] = block[0]
    for index in range(1, length):
        output_bytes[index] = gate_xor(block[index], block[index - 1], 8)
    return bytes(output_bytes)


def _mode1_inverse(residual: bytes) -> bytes:
    """Inverse of Mode 1: reconstructs `x[i]` from residuals and history."""
    if not residual:
        return b""
    length = len(residual)
    output_bytes = bytearray(length)
    output_bytes[0] = residual[0]
    for index in range(1, length):
        output_bytes[index] = gate_xor(residual[index], output_bytes[index - 1], 8)
    return bytes(output_bytes)


def _mode2_forward(block: bytes) -> bytes:
    """Mode 2 (Gray predictor): `y[i] = x[i] XOR Gray(x[i-1])`."""
    if not block:
        return b""
    length = len(block)
    output_bytes = bytearray(length)
    output_bytes[0] = block[0]
    for index in range(1, length):
        predictor = _gray_pred(block[index - 1])
        output_bytes[index] = gate_xor(block[index], predictor, 8)
    return bytes(output_bytes)


def _mode2_inverse(residual: bytes) -> bytes:
    """Inverse of Mode 2 using the same Gray predictor derived from decoded history."""
    if not residual:
        return b""
    length = len(residual)
    output_bytes = bytearray(length)
    output_bytes[0] = residual[0]
    for index in range(1, length):
        predictor = _gray_pred(output_bytes[index - 1])
        output_bytes[index] = gate_xor(residual[index], predictor, 8)
    return bytes(output_bytes)


def _mode3_forward(block: bytes) -> bytes:
    """Mode 3 (order-2 residual):
    `y[1] = x[1] XOR x[0]`; for `i >= 2`, `y[i] = x[i] XOR x[i-2]`.
    """
    length = len(block)
    if length == 0:
        return b""
    if length == 1:
        return bytes([block[0]])
    output_bytes = bytearray(length)
    output_bytes[0] = block[0]
    output_bytes[1] = gate_xor(block[1], block[0], 8)
    for index in range(2, length):
        output_bytes[index] = gate_xor(block[index], block[index - 2], 8)
    return bytes(output_bytes)


def _mode3_inverse(residual: bytes) -> bytes:
    """Inverse of Mode 3 (order-2 residual)."""
    length = len(residual)
    if length == 0:
        return b""
    if length == 1:
        return bytes([residual[0]])
    output_bytes = bytearray(length)
    output_bytes[0] = residual[0]
    output_bytes[1] = gate_xor(residual[1], output_bytes[0], 8)
    for index in range(2, length):
        output_bytes[index] = gate_xor(residual[index], output_bytes[index - 2], 8)
    return bytes(output_bytes)


def _mode4_forward(block: bytes) -> bytes:
    """Mode 4 (run-segment selector, reversible, gates + MUX only).

    For `i >= 2`, choose between two predictors based on *local equality* of the
    last two decoded bytes, expressed as a canonical mask via `byte_eq`:

    - `pred_run = x[i-1]` when a run continues (`x[i-1] == x[i-2]`).
    - `pred_alt = Gray(x[i-1])` otherwise, to avoid overfitting to runs.

    The selector is realized as `pred = mux_byte(m, pred_run, pred_alt)` with
    `m ∈ {0xFF, 0x00}`. Residuals are formed by XORing the current byte with the
    predictor.
    """
    length = len(block)
    if length == 0:
        return b""
    if length == 1:
        return bytes([block[0]])
    output_bytes = bytearray(length)
    output_bytes[0] = block[0]
    output_bytes[1] = gate_xor(block[1], block[0], 8)
    for index in range(2, length):
        selector_mask = byte_eq(block[index - 1], block[index - 2])  # 0xFF or 0x00
        pred_run = block[index - 1] & 0xFF
        pred_alt = _gray_pred(block[index - 1])
        predictor = mux_byte(selector_mask, pred_run, pred_alt)
        output_bytes[index] = gate_xor(block[index], predictor, 8)
    return bytes(output_bytes)


def _mode4_inverse(residual: bytes) -> bytes:
    """Inverse of Mode 4 using the same run/alt selection recomputed from history."""
    length = len(residual)
    if length == 0:
        return b""
    if length == 1:
        return bytes([residual[0]])
    output_bytes = bytearray(length)
    output_bytes[0] = residual[0]
    output_bytes[1] = gate_xor(residual[1], output_bytes[0], 8)
    for index in range(2, length):
        selector_mask = byte_eq(output_bytes[index - 1], output_bytes[index - 2])
        pred_run = output_bytes[index - 1] & 0xFF
        pred_alt = _gray_pred(output_bytes[index - 1])
        predictor = mux_byte(selector_mask, pred_run, pred_alt)
        output_bytes[index] = gate_xor(residual[index], predictor, 8)
    return bytes(output_bytes)


###############################################################################
# External wrapper (mode selection + reversibility). Mode 4 is included.
###############################################################################

def circuit_map_automaton_forward(block: bytes) -> Tuple[bytes, Dict]:
    """Select the best reversible mapping by minimum 0th-order entropy.

    Returns
    -------
    (mapped_bytes, {"mode": m}) with `m ∈ {0,1,2,3,4}`.

    Notes
    -----
    Mode 0 is the identity (no transform). Ties on entropy break toward the
    smaller mode number to ensure deterministic selection.
    """
    from collections import Counter
    import math

    def H0(data: bytes) -> float:
        if not data:
            return 0.0
        length_local = len(data)
        counts = Counter(data)
        h = 0.0
        for v in counts.values():
            p = v / length_local
            h -= p * math.log2(p)
        return h

    candidates = [
        (0, block),
        (1, _mode1_forward(block)),
        (2, _mode2_forward(block)),
        (3, _mode3_forward(block)),
        (4, _mode4_forward(block)),  # run-segment selection automaton
    ]

    best_mode = 0
    best_bytes = block
    best_entropy = H0(block)

    # Prefer lower mode index on exact entropy ties
    for mode_id, mapped in candidates[1:]:
        ent = H0(mapped)
        if ent < best_entropy - 1e-9 or (abs(ent - best_entropy) <= 1e-9 and mode_id < best_mode):
            best_mode, best_bytes, best_entropy = mode_id, mapped, ent

    return best_bytes, {"mode": best_mode}


def circuit_map_automaton_inverse(mapped: bytes, theta: Dict) -> bytes:
    """Inverse of the selected mapping.

    Parameters
    ----------
    mapped : bytes
        The transformed bytes.
    theta : Dict
        Must contain key `"mode"` with the integer mode id.
    """
    mode_id = int(theta.get("mode", 0)) & 0xFF
    if mode_id == 0:
        return mapped
    if mode_id == 1:
        return _mode1_inverse(mapped)
    if mode_id == 2:
        return _mode2_inverse(mapped)
    if mode_id == 3:
        return _mode3_inverse(mapped)
    if mode_id == 4:
        return _mode4_inverse(mapped)
    # Unknown mode → conservative: identity (keeps reversibility of known modes)
    return mapped

def _first_order_bit_entropy(block: bytes) -> float:
    """Approximate first‑order (Markov) bit entropy of a byte block.

    We treat the block as a sequence of bits and estimate the conditional
    entropy H(bit_t | bit_{t-1}).  Transitions between adjacent bits are
    counted to obtain joint and marginal distributions.  If the block
    is too short, a default entropy of 1.0 is returned (maximally
    random)."""
    nbits = len(block) * 8
    if nbits < 2:
        return 1.0
    # counts[x][y] counts transitions from bit x to bit y
    counts = [[0, 0], [0, 0]]
    # count transitions
    prev = (block[0] >> 7) & 1
    bit_idx = 1
    for b in block:
        for k in range(8):
            if b == block[0] and k == 0:
                # skip first bit (already prev)
                continue
            bit = (b >> (7 - k)) & 1
            counts[prev][bit] += 1
            prev = bit
            bit_idx += 1
    # compute probabilities
    total_trans = float(nbits - 1)
    H = 0.0
    # compute marginal P(x) (sum counts[x][0]+counts[x][1])
    for x in [0, 1]:
        px = counts[x][0] + counts[x][1]
        if px == 0:
            continue
        for y in [0, 1]:
            pxy = counts[x][y]
            if pxy == 0:
                continue
            p_y_given_x = pxy / px
            H -= (pxy / total_trans) * math.log2(p_y_given_x)
    return H

###############################################################################
# Bitwise reversible circuit modules
###############################################################################

def bitplane_interleave(data: bytes) -> bytes:
    """Interleave bitplanes of a byte stream.

    For every block of 8 bytes, produce 8 bytes where the first byte
    contains all the MSBs of the block, the second byte contains all the
    second MSBs, and so on.  The transform is its own inverse when
    applied twice.
    """
    out = bytearray()
    it = iter(data)
    block = list(itertools.islice(it, 8))
    while block:
        while len(block) < 8:
            block.append(0)
        for bit in range(8):
            v = 0
            for i, b in enumerate(block):
                v |= ((b >> (7 - bit)) & 1) << (7 - i)
            out.append(v)
        block = list(itertools.islice(it, 8))
    return bytes(out)

def bitplane_deinterleave(data: bytes, orig_len: int) -> bytes:
    out = bytearray()
    it = iter(data)
    block = list(itertools.islice(it, 8))
    while block:
        for i in range(8):
            out.append(0)
        for bit in range(8):
            byte = block[bit]
            for i in range(8):
                out[-8 + i] |= ((byte >> (7 - i)) & 1) << (7 - bit)
        block = list(itertools.islice(it, 8))
    return bytes(out[:orig_len])

# === BEGIN V2 helper utilities (inserted above encode_new_pipeline) ===
def bytes_to_bitplanes(data: bytes) -> Tuple[List[List[int]], int]:
    """
    Split byte stream into 8 MSB-first bit-planes.
    planes[j][t] is bit j (0..7, MSB=0) of data[t], as 0/1 int.
    Returns (planes, L) where L=len(data).
    """
    L = len(data)
    planes = [[0] * L for _ in range(8)]
    for t, b in enumerate(data):
        # MSB-first: j=0 => bit7, j=7 => bit0
        planes[0][t] = (b >> 7) & 1
        planes[1][t] = (b >> 6) & 1
        planes[2][t] = (b >> 5) & 1
        planes[3][t] = (b >> 4) & 1
        planes[4][t] = (b >> 3) & 1
        planes[5][t] = (b >> 2) & 1
        planes[6][t] = (b >> 1) & 1
        planes[7][t] = (b >> 0) & 1
    return planes, L
def bitplanes_to_bytes(planes: List[List[int]]) -> bytes:
    """
    Inverse of bytes_to_bitplanes. Expects exactly 8 planes, MSB-first.
    """
    if not planes:
        return b""
    assert len(planes) == 8, "expect 8 bit-planes"
    L = len(planes[0])
    for j in range(8):
        assert len(planes[j]) == L, "plane length mismatch"

    out = bytearray(L)
    for t in range(L):
        # MSB-first pack
        val = ((planes[0][t] & 1) << 7) | ((planes[1][t] & 1) << 6) | \
              ((planes[2][t] & 1) << 5) | ((planes[3][t] & 1) << 4) | \
              ((planes[4][t] & 1) << 3) | ((planes[5][t] & 1) << 2) | \
              ((planes[6][t] & 1) << 1) | ((planes[7][t] & 1) << 0)
        out[t] = val
    return bytes(out)
    
def avg_run_bits(bits: list[int]) -> float:
    if not bits: return 0.0
    runs, prev = 1, bits[0]
    for b in bits[1:]:
        if b != prev:
            runs += 1; prev = b
    return len(bits)/runs
def H0_bits(bits: list[int]) -> float:
    if not bits: return 0.0
    n = len(bits); c1 = sum(bits)
    import math
    p = c1/n
    return 0.0 if p==0.0 or p==1.0 else -p*math.log2(p) - (1-p)*math.log2(1-p)
    
def rle_binary(bits: list[int]) -> tuple[int, list[int]]:
    if not bits: return (0, [])
    runs=[]; cur=1
    for i in range(1, len(bits)):
        if bits[i]==bits[i-1]: cur+=1
        else: runs.append(cur); cur=1
    runs.append(cur)
    return bits[0], runs
def unrle_binary(first_bit: int, runs: list[int]) -> list[int]:
    out=[]; b=first_bit & 1
    for r in runs:
        out.extend([b]*r); b ^= 1
    return out
    
def pack_bits_to_bytes(bits: list[int]) -> bytes:
    n=len(bits); out=bytearray((n+7)//8)
    for i,bit in enumerate(bits):
        if bit & 1:
            out[i>>3] |= 1<<(7-(i&7))
    return bytes(out)
def unpack_bits_from_bytes(buf: bytes, nbits: int) -> list[int]:
    out=[0]*nbits
    for i in range(nbits):
        out[i] = (buf[i>>3] >> (7-(i&7))) & 1
    return out
# === END V2 helper utilities ===

class _BitReader:
    """Simple bitstream reader over a bytes object."""
    __slots__ = ("buf", "byte", "bit")  # bit in [0..7], points to next bit to read (MSB-first)
    def __init__(self, buf: bytes, byte_pos: int = 0, bit_pos: int = 0):
        self.buf = buf
        self.byte = byte_pos
        self.bit  = bit_pos  # 0..7, 0 means next read will take MSB of current byte

    def read_bit(self) -> int:
        if self.byte >= len(self.buf):
            raise ValueError("BitReader: out of data")
        b = self.buf[self.byte]
        v = (b >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit = 0
            self.byte += 1
        return v

    def align_next_byte(self) -> None:
        if self.bit != 0:
            self.bit = 0
            self.byte += 1

    def tell(self) -> tuple[int, int]:
        return (self.byte, self.bit)

class _BitWriter:
    __slots__ = ("buf", "cur", "bitpos")
    def __init__(self):
        self.buf = bytearray(); self.cur = 0; self.bitpos = 0  # 0..7, 写到MSB->LSB
    def write_bit(self, b: int):
        self.cur |= ( (b & 1) << (7 - self.bitpos) )
        self.bitpos += 1
        if self.bitpos == 8:
            self.buf.append(self.cur); self.cur = 0; self.bitpos = 0
    def write_unary(self, q: int):
        for _ in range(q): self.write_bit(1)
        self.write_bit(0)
    def write_kbits(self, val: int, k: int):
        for i in range(k - 1, -1, -1): self.write_bit((val >> i) & 1)
    def pad_to_byte(self):
        if self.bitpos: self.buf.append(self.cur); self.cur = 0; self.bitpos = 0
    def align_next_byte(self):
        # Alias for symmetry with _BitReader
        self.pad_to_byte()
    def getvalue_bits(self):
        # Return (bytes, bitlen) without padding; we must include partial byte if any
        bitlen = len(self.buf)*8 + self.bitpos
        out = bytes(self.buf) + (bytes([self.cur]) if self.bitpos else b'')
        return out, bitlen

###############################################################################
# Integer coders (Rice/Golomb and Elias)
###############################################################################

# ---- ZigZag for signed deltas (CDC orig_len around avg) ----
def _zz_enc(x: int) -> int:
    return (x << 1) if x >= 0 else ((-x) << 1) - 1
def _zz_dec(n: int) -> int:
    return (n >> 1) if (n & 1) == 0 else -((n + 1) >> 1)

# ---- Simple canonical Huffman (code lengths + canonical numbering) ----
class _HuffNode:
    __slots__ = ('w','sym','left','right')
    def __init__(self, w, sym=None, left=None, right=None):
        self.w=w; self.sym=sym; self.left=left; self.right=right
    def __lt__(self, other):
        if self.w != other.w: return self.w < other.w
        # stabilize
        a = self.sym if self.sym is not None else -1
        b = other.sym if other.sym is not None else -1
        return a < b

def _huff_lengths(freq: dict[int,int]) -> dict[int,int]:
    import heapq
    heap = []
    for s,f in freq.items():
        heap.append(_HuffNode(max(1,f), sym=s))
    if not heap:
        return {}
    if len(heap)==1:
        return {heap[0].sym: 1}
    heapq.heapify(heap)
    while len(heap)>1:
        a=heapq.heappop(heap); b=heapq.heappop(heap)
        heapq.heappush(heap, _HuffNode(a.w+b.w, left=a, right=b))
    # DFS to assign lengths
    lengths={}
    stack=[(heap[0],0)]
    while stack:
        nd,d=stack.pop()
        if nd.sym is not None:
            lengths[nd.sym] = max(1,d)
        else:
            stack.append((nd.left, d+1)); stack.append((nd.right, d+1))
    return lengths

def _huff_canonical(lengths: dict[int,int]):
    # Returns (enc_tbl: sym->(code,len), dec_tbl: (len,code)->sym, maxlen)
    items = sorted(lengths.items(), key=lambda kv:(kv[1], kv[0]))
    enc={}; dec={}
    code=0; prev=0; maxlen=0
    for sym,L in items:
        if L!=prev:
            code <<= (L-prev); prev=L
        enc[sym]=(code, L); dec[(L, code)]=sym; maxlen=max(maxlen, L); code+=1
    return enc, dec, maxlen

def _huff_encode_symbols(bw: _BitWriter, enc_tbl: dict[int,tuple[int,int]], syms: list[int]):
    for s in syms:
        c,L = enc_tbl[s]
        bw.write_kbits(c, L)

def _huff_decode_symbols(br: _BitReader, dec_tbl: dict[tuple[int,int],int], maxlen: int, nvals: int) -> list[int]:
    out=[]; 
    for _ in range(nvals):
        c=0
        for L in range(1, maxlen+1):
            c = (c<<1) | br.read_bit()
            key=(L,c)
            if key in dec_tbl:
                out.append(dec_tbl[key]); break
        else:
            raise ValueError("Huffman decode failed")
    return out

# ---- Rice helpers (bit-precise, no padding) ----
def rice_write_values(bw: _BitWriter, seq: list[int], k: int):
    M = 1<<k
    for n in seq:
        q, r = (n // M, n % M) if k>0 else (n, 0)
        for _ in range(q): bw.write_bit(1)
        bw.write_bit(0)
        if k>0: bw.write_kbits(r, k)

def rice_read_n(br: _BitReader, k: int, nvals: int) -> list[int]:
    M=1<<k; out=[]
    for _ in range(nvals):
        q=0
        while br.read_bit()==1: q+=1
        r=0
        for _ in range(k):
            r = (r<<1) | br.read_bit()
        out.append(q*M + r)
    return out

# ---- Elias–Fano for cumulative payload ends ----
def _ef_choose_l(U: int, n: int) -> int:
    if n <= 0 or U <= 1: return 0
    avg = U // n
    if avg <= 1: return 0
    import math
    return max(0, int(math.floor(math.log2(avg))))

def ef_write_positions(bw: _BitWriter, P: list[int], U: int):
    n = len(P)
    l = _ef_choose_l(U, n)
    # Low bits
    for x in P:
        bw.write_kbits(x & ((1<<l)-1), l)
    # High bitvector B of length m + n with ones at hi_i + i
    m = (U + ((1<<l)-1)) >> l  # ceil(U/2^l)
    total = m + n
    # Build positions
    pos_bits = [0]*total
    for i,x in enumerate(P):
        hi = x >> l
        idx = hi + i
        pos_bits[idx] = 1
    for b in pos_bits:
        bw.write_bit(b)

def ef_read_positions(br: _BitReader, U: int, n: int) -> list[int]:
    l = _ef_choose_l(U, n)
    lows = [0]*n
    for i in range(n):
        v=0
        for _ in range(l):
            v = (v<<1) | br.read_bit()
        lows[i]=v
    m = (U + ((1<<l)-1)) >> l
    total = m + n
    ones_pos=[]
    for idx in range(total):
        bit = br.read_bit()
        if bit==1:
            ones_pos.append(idx)
            if len(ones_pos)==n:  # early exit if we've got all ones
                # consume remaining bits if any
                for _ in range(idx+1, total):
                    br.read_bit()
                break
    # Reconstruct P[i] = ((ones_pos[i]-i) << l) | lows[i]
    P=[0]*n
    for i in range(n):
        hi = ones_pos[i] - i
        P[i] = (hi << l) | lows[i]
    return P

# ---- RLE for method ids ----
def _rle_ids(ids: list[int]) -> tuple[list[int], list[int]]:
    if not ids: return [], []
    syms=[ids[0]]; runs=[1]
    for x in ids[1:]:
        if x==syms[-1]: runs[-1]+=1
        else: syms.append(x); runs.append(1)
    return syms, runs

def rice_encode(seq, k: int) -> bytes:
    bw = _BitWriter()
    M = 1 << k
    for n in seq:
        q, r = (n // M, n % M) if k > 0 else (n, 0)
        bw.write_unary(q)
        if k > 0: bw.write_kbits(r, k)
    bw.pad_to_byte()
    return bytes(bw.buf)

def rice_decode(data: bytes, k: int, nvals: int) -> List[int]:
    bitstr = ''.join(f'{b:08b}' for b in data)
    i = 0; M = 1 << k
    out: List[int] = []
    N = len(bitstr)

    def need(bits=1):
        # 返回是否还够 bits 位可读
        return i + bits <= N

    for _ in range(nvals):
        # 读一元 q：直到遇到 '0'
        q = 0
        while True:
            if not need(1):
                raise ValueError("Rice stream truncated while reading unary part")
            if bitstr[i] == '1':
                q += 1; i += 1
            else:
                i += 1  # skip '0'
                break
        # 读余数 r（k==0 时无余数位）
        if k > 0:
            if not need(k):
                raise ValueError("Rice stream truncated while reading remainder")
            r = int(bitstr[i:i+k], 2); i += k
        else:
            r = 0
        out.append(q * M + r)
    return out

def _rice_decode_until_len(br: _BitReader, k: int, target_len: int) -> list[int]:
    """
    Decode Rice-coded run lengths from current bit position until sum(runs) == target_len.
    Unary part is '1'*q + '0'; remainder is k bits (k may be 0).
    """
    runs: list[int] = []
    total = 0
    M = 1 << k
    while total < target_len:
        # read unary q
        q = 0
        while True:
            bit = br.read_bit()  # may raise on truncation
            if bit == 1:
                q += 1
            else:
                break  # saw the '0' terminator
        # read remainder r
        if k == 0:
            r = 0
        else:
            r = 0
            for _ in range(k):
                r = (r << 1) | br.read_bit()
        val = q * M + r
        if val <= 0:
            # RLE run must be positive
            raise ValueError("Invalid Rice value (non-positive)")
        runs.append(val)
        total += val
        if total > target_len:
            raise ValueError("RLE overrun: sum(runs) > target_len")
    return runs


def _choose_best_rice(runs: list[int]) -> tuple[int, bytes]:
    """Try k in [0..15], return (best_k, encoded_bytes_with_byte_padding)."""
    best_k, best_bytes = 0, None
    for k in range(16):
        buf = rice_encode(runs, k)  # already byte-padded
        if best_bytes is None or len(buf) < len(best_bytes):
            best_k, best_bytes = k, buf
    return best_k, best_bytes

def encode_new_pipeline(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    V2 pipeline (slim header, NOT backward compatible):

      bytes --circuit_map_automaton--> mapped
      mapped --split--> 8 bit-planes (MSB-first)
      per plane: compare RAW vs (BBWT -> RLE -> Rice with optimal k), pick smaller
      header = [hdr0 | raw_mask | b1_mask | k_list_for_encoded_planes]
      payload = concat_j RAW_bytes or Rice_bitstream (each plane byte-aligned)

    Notes
    -----
    - No internal flag/L/run_count/paylen. Decoder uses container's `orig_len` as L.
    - After each encoded plane's Rice stream, we pad to the next byte boundary.
    """
    if not block:
        return b"", {}

    # 1) reversible byte-wise circuit (mode autodetected by H0)
    mapped, theta = circuit_map_automaton_forward(block)
    mode = int(theta.get("mode", 0)) & 7  # 3 bits are enough

    # 2) split to bit-planes
    planes, L = bytes_to_bitplanes(mapped)
    nbytes_plane = (L + 7) // 8

    raw_mask = 0
    b1_mask  = 0
    k_bytes: list[int] = []
    payload_chunks: list[bytes] = []

    for j in range(8):
        Uj = planes[j]
        # Build both candidates then pick smaller:
        # RAW candidate
        raw_bytes = pack_bits_to_bytes(Uj)

        # ENCODED candidate: BBWT -> RLE -> Rice (best k)
        Lj = bbwt_forward(bytes(Uj))
        Lj_bits = list(Lj)
        b1, runs = rle_binary(Lj_bits)
        if not runs:
            # Degenerate: treat as RAW
            raw_mask |= (1 << j)
            payload_chunks.append(raw_bytes)
            continue
        k_opt, rice_bytes = _choose_best_rice(runs)

        # Compare sizes (count 1 extra byte for k if encoded)
        if len(raw_bytes) <= len(rice_bytes) + 1:
            # Choose RAW
            raw_mask |= (1 << j)
            payload_chunks.append(raw_bytes)
        else:
            # Choose ENCODED
            # record b1 and k
            if b1 & 1:
                b1_mask |= (1 << j)
            k_bytes.append(k_opt & 0xFF)
            payload_chunks.append(rice_bytes)

    # 3) header assembly
    hdr0 = (mode & 0x07) << 5
    header = bytearray([hdr0, raw_mask & 0xFF, b1_mask & 0xFF])
    # append k_list in j order for encoded planes
    enc_iter = iter(k_bytes)
    # 保证顺序一致：按 j=0..7 只对 encoded 平面写 k
    #（这里其实不需要回写，因为 k_bytes 已经按生成顺序了；此步仅作自检）
    # 若更稳妥，可改成：逐平面扫描 raw_mask==0 时从 enc_iter 取一个 k 写入。
    for j in range(8):
        if ((raw_mask >> j) & 1) == 0:
            header.append(next(enc_iter))

    # 4) payload assembly（编码平面已按字节对齐；我们不再跨平面共享比特位）
    payload = b"".join(payload_chunks)

    return bytes(header) + payload, {}  # meta 无需额外信息


# ------------------- REPLACE: decode_new_pipeline (slim header; uses orig_len) -------------------

def decode_new_pipeline(payload: bytes, orig_len: int, meta: Dict[str, Any]) -> bytes:
    """
    Inverse of slim-header V2 pipeline.

    Header layout:
      hdr0(1B):  mode in top 3 bits (mode = hdr0 >> 5)
      raw_mask:  1B
      b1_mask:   1B
      k_list:    1B per encoded plane (in j order, j=0..7)

    Payload layout:
      For j=0..7:
        if RAW:     ceil(L/8) bytes of packed bits
        if ENCODED: Rice bitstream (byte-aligned after finishing this plane)
    """
    L = int(orig_len)
    if L == 0:
        return b""

    if len(payload) < 3:
        raise ValueError("V2 slim header truncated")

    pos = 0
    hdr0 = payload[pos]; pos += 1
    raw_mask = payload[pos]; pos += 1
    b1_mask  = payload[pos]; pos += 1
    mode = (hdr0 >> 5) & 0x07

    # Read k_list (encoded planes only, in j order)
    k_list: list[int] = []
    enc_count = 8 - bin(raw_mask).count("1")
    if pos + enc_count > len(payload):
        raise ValueError("V2 slim header k_list truncated")
    for _ in range(enc_count):
        k_list.append(payload[pos]); pos += 1

    # Prepare to read payload at byte boundary
    data = payload[pos:]
    data_pos = 0  # byte index into 'data'

    planes: list[list[int]] = []
    k_iter = iter(k_list)

    for j in range(8):
        if ((raw_mask >> j) & 1) == 1:
            # RAW plane
            need = (L + 7) // 8
            if data_pos + need > len(data):
                raise ValueError("V2 payload truncated in RAW plane")
            buf = data[data_pos:data_pos + need]; data_pos += need
            Uj = unpack_bits_from_bytes(buf, L)
            planes.append(Uj)
        else:
            # ENCODED plane: read Rice from current bitpos until sum==L; then align to next byte
            k = next(k_iter)
            b1 = (b1_mask >> j) & 1
            br = _BitReader(data, data_pos, 0)
            runs = _rice_decode_until_len(br, k, L)
            # align to next byte for the next plane
            br.align_next_byte()
            data_pos, _bit = br.tell()
            # rebuild plane bits
            Lj_bits = unrle_binary(b1, runs)
            Uj_bytes = bbwt_inverse(bytes(Lj_bits))
            Uj = list(Uj_bytes)
            if len(Uj) != L:
                Uj = Uj[:L] if len(Uj) > L else (Uj + [0] * (L - len(Uj)))
            planes.append(Uj)

    # Recombine planes -> bytes, then invert circuit automaton
    mapped = bitplanes_to_bytes(planes)
    block  = circuit_map_automaton_inverse(mapped, {"mode": mode})
    return block

def nibble_swap(data: bytes) -> bytes:
    """Swap the high and low 4‑bit nibbles of each byte."""
    return bytes(((b & 0x0F) << 4) | ((b & 0xF0) >> 4) for b in data)

_BIT_REVERSE_TABLE = bytes(int('{:08b}'.format(i)[::-1], 2) for i in range(256))

def bit_reverse(data: bytes) -> bytes:
    """Reverse the bit order of each byte using a lookup table."""
    return bytes(_BIT_REVERSE_TABLE[b] for b in data)

def gray_encode_bytes(data: bytes) -> bytes:
    """Convert each byte into its Gray code: g = x ^ (x >> 1)."""
    return bytes(((b ^ (b >> 1)) & 0xFF) for b in data)

def gray_decode_bytes(data: bytes) -> bytes:
    """Recover original bytes from their Gray codes.

    Given a Gray code g, recover n by iterative XORs with right shifts:
        n = g
        n ^= (n >> 1)
        n ^= (n >> 2)
        n ^= (n >> 4)
    """
    out = bytearray()
    for g in data:
        n = g
        n ^= (n >> 1)
        n ^= (n >> 2)
        n ^= (n >> 4)
        out.append(n & 0xFF)
    return bytes(out)

# -----------------------------------------------------------------------------
# LZ77 (ULEB128) – overlap-safe, 4KiB window, container-compatible
# -----------------------------------------------------------------------------

def _lz77_match_len_overlap(window: bytearray, block: bytes, pos: int, dist: int, n: int) -> int:
    """
    Compute match length allowing overlaps.
    When matched >= dist, the reference byte comes from the bytes that would
    have just been produced (i.e., block[pos + matched - dist]).
    """
    win_len = len(window)
    matched = 0
    while pos + matched < n:
        if matched < dist:
            idx = win_len - dist + matched  # reference in existing window
            if idx >= win_len:  # safety guard; shouldn't happen if dist <= win_len
                break
            ref = window[idx]
        else:
            src = pos + matched - dist      # overlapping reference
            if src >= n:
                break
            ref = block[src]
        if ref != block[pos + matched]:
            break
        matched += 1
    return matched


def encode_lz77(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Encode `block` with a simple LZ77 using ULEB128 for (length, distance).
    Stream format:
      - Literal: [0][byte]
      - Match  : [1][ULEB length][ULEB distance]
    Window keeps the last 4096 bytes.
    """
    WINDOW_MAX = 4096
    MIN_MATCH  = 3

    window = bytearray()
    out = bytearray()
    n = len(block)
    pos = 0

    while pos < n:
        best_len = 0
        best_dist = 0

        max_window = min(len(window), WINDOW_MAX)
        if max_window >= 1:
            # Search near to far (near matches often longer)
            for dist in range(1, max_window + 1):
                m = _lz77_match_len_overlap(window, block, pos, dist, n)
                if m > best_len:
                    best_len = m
                    best_dist = dist
                    # Optional early break if extremely long, e.g., Deflate-style 258
                    # if best_len >= 258:
                    #     break

        if best_len >= MIN_MATCH:
            # Emit match
            out.append(1)
            out += uleb128_encode(best_len)
            out += uleb128_encode(best_dist)
            # Advance and push matched bytes into the window
            window.extend(block[pos:pos + best_len])
            pos += best_len
        else:
            # Emit literal
            b = block[pos]
            out.append(0)
            out.append(b)
            window.append(b)
            pos += 1

        # Trim window
        if len(window) > WINDOW_MAX:
            del window[:-WINDOW_MAX]

    return bytes(out), {}

def decode_lz77(data: bytes, orig_len: int) -> bytes:
    """
    Decode LZ77 stream produced by `encode_lz77`.
    Reads 0/1 marker, ULEB128 length/dist, supports overlap copying.
    Validates final output length.
    """
    WINDOW_MAX = 4096

    window = bytearray()
    out = bytearray()
    i = 0
    n = len(data)

    while i < n and len(out) < orig_len:
        flag = data[i]; i += 1

        if flag == 0:
            if i >= n:
                raise ValueError("LZ77 truncated literal")
            b = data[i]; i += 1
            out.append(b)
            window.append(b)

        elif flag == 1:
            length, i = uleb128_decode_stream(data, i)
            dist,   i = uleb128_decode_stream(data, i)
            if dist == 0:
                raise ValueError("LZ77 invalid distance 0")

            # Copy with overlap: always read from "window tail - dist"
            for _ in range(length):
                if dist > len(window):
                    raise ValueError("LZ77 distance beyond window")
                b = window[-dist]
                out.append(b)
                window.append(b)
                if len(out) == orig_len:
                    break
        else:
            raise ValueError("LZ77 unknown flag")

        if len(window) > WINDOW_MAX:
            del window[:-WINDOW_MAX]

    if len(out) != orig_len:
        raise ValueError("LZ77 output length mismatch")

    return bytes(out)

# -----------------------------
# Re-Pair compression
# -----------------------------
def _count_pairs(seq: List[int]) -> Dict[Tuple[int, int], int]:
    freq: Dict[Tuple[int,int], int] = {}
    for a, b in zip(seq, seq[1:]):
        pair = (a, b)
        freq[pair] = freq.get(pair, 0) + 1
    return freq

def _replace_non_overlapping(seq: List[int], target: Tuple[int,int], new_sym: int) -> Tuple[List[int], int]:
    """Replace all non-overlapping occurrences of 'target' with new_sym, left-to-right."""
    a, b = target
    i = 0
    out: List[int] = []
    replaced = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i+1] == b:
            out.append(new_sym)
            i += 2
            replaced += 1
        else:
            out.append(seq[i])
            i += 1
    return out, replaced

def repair_compress(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Strict Re-Pair compressor producing a straight-line grammar:
      - Each rule is A -> XY (two symbols), terminals are 0..255.
      - Each round replaces ALL non-overlapping occurrences of the most frequent pair.
      - Output format (all integers = ULEB128):
          magic 'RP' (0x52 0x50),
          terminals=256,
          nrules,
          rules (each rule: rhs0, rhs1) in creation order; LHS ids are implicit: 256+i,
          final_seq_len,
          final_seq symbols.
    References: Larsson & Moffat (1999/2000); Bille et al. (2017)."""
    if not block:
        # Encode empty grammar: 0 rules, empty seq
        out = bytearray(b'RP')
        out += uleb128_encode(256)      # terminal alphabet
        out += uleb128_encode(0)        # nrules
        out += uleb128_encode(0)        # seq len
        return bytes(out), {"rules": {}, "final_len": 0}

    seq: List[int] = list(block)          # initial sequence over terminals 0..255
    next_sym = 256                         # first non-terminal id
    rules: Dict[int, Tuple[int,int]] = {}  # LHS -> (rhs0, rhs1)

    while True:
        freq = _count_pairs(seq)
        if not freq:
            break
        # choose most frequent pair with frequency >= 2; tie-break by lexicographic pair for determinism
        best_pair, best_f = None, 1
        for p, f in freq.items():
            if f > best_f or (f == best_f and best_pair is not None and p < best_pair):
                best_pair, best_f = p, f
        if best_pair is None or best_f < 2:
            break

        # concurrent non-overlapping replacement in this round
        new_seq, replaced = _replace_non_overlapping(seq, best_pair, next_sym)
        if replaced < 2:
            # 保险：若由于重叠导致有效替换<2，则不引入无效规则
            break

        rules[next_sym] = (best_pair[0], best_pair[1])
        seq = new_seq
        next_sym += 1

    # ---- Serialize
    out = bytearray()
    out += b'RP'                          # magic
    out += uleb128_encode(256)          # terminal alphabet size
    nrules = next_sym - 256
    out += uleb128_encode(nrules)

    # rules are implicit LHS ids: 256,257,... in creation order
    for i in range(nrules):
        rhs = rules[256 + i]
        out += uleb128_encode(rhs[0])
        out += uleb128_encode(rhs[1])

    out += uleb128_encode(len(seq))
    for s in seq:
        out += uleb128_encode(s)

    meta: Dict[str, Any] = {
        "rules": rules,               # for debugging / introspection
        "final_len": len(seq),
        "terminals": 256,
        "nrules": nrules,
    }
    return bytes(out), meta

# -----------------------------
# Re-Pair decompression
# -----------------------------
def repair_decompress(data: bytes, orig_len: int) -> bytes:
    """Inverse of repair_compress."""
    i = 0
    if len(data) < 2 or data[0:2] != b'RP':
        raise ValueError("Bad magic")
    i = 2

    terminals, i = uleb128_decode_stream(data, i)
    if terminals != 256:
        # 这版编码写死 256 终结符；需要可配置时再做扩展
        raise ValueError("Unsupported terminal alphabet")

    nrules, i = uleb128_decode_stream(data, i)
    # read rules as implicit LHS = 256 + idx
    rules: Dict[int, Tuple[int,int]] = {}
    for ridx in range(nrules):
        a, i = uleb128_decode_stream(data, i)
        b, i = uleb128_decode_stream(data, i)
        rules[256 + ridx] = (a, b)

    seq_len, i = uleb128_decode_stream(data, i)
    seq: List[int] = []
    for _ in range(seq_len):
        s, i = uleb128_decode_stream(data, i)
        seq.append(s)

    # Expand grammar iteratively with memoization to avoid deep recursion
    cache: Dict[int, bytes] = {}

    def expand_symbol(sym: int) -> bytes:
        if sym < 256:
            return bytes((sym,))
        if sym in cache:
            return cache[sym]
        # iterative stack expansion (post-order)
        stack: List[Tuple[int, int]] = [(sym, 0)]  # (node, state 0=go left,1=go right,2=emit)
        out_stack: List[bytes] = []
        while stack:
            node, st = stack.pop()
            if node < 256:
                out_stack.append(bytes((node,)))
                continue
            if st == 0:
                rhs = rules[node]
                stack.append((node, 2))
                stack.append((rhs[1], 0))
                stack.append((rhs[0], 0))
            else:
                # pop two expansions
                right = out_stack.pop()
                left  = out_stack.pop()
                val = left + right
                cache[node] = val
                out_stack.append(val)
        return out_stack[-1]

    out = bytearray()
    for s in seq:
        out += expand_symbol(s)

    if len(out) != orig_len:
        raise RuntimeError(f"RePair output length mismatch: got {len(out)}, expect {orig_len}")
    return bytes(out)
    
###############################################################################
# LFSR predictor (XOR deltas)
###############################################################################

def encode_lfsr_predict(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Encode ``block`` by predicting each byte using an LFSR and coding the residual.

    The predictor uses an 8‑bit LFSR to generate a pseudo‑random stream and
    subtracts it from the input.  Residuals are encoded as ULEB128.
    """
    state = 1
    out = bytearray()
    for b in block:
        pred = state
        delta = (b - pred) & 0xFF
        out += uleb128_encode(delta)
        # update LFSR
        fb = 0
        taps = 0b10010110
        for bit in range(8):
            if (taps >> bit) & 1:
                fb ^= (state >> bit) & 1
        state = ((state << 1) & 0xFF) | fb
    return bytes(out), {}

def decode_lfsr_predict(data: bytes, orig_len: int) -> bytes:
    state = 1
    out = bytearray()
    pos = 0
    for _ in range(orig_len):
        delta, pos = uleb128_decode_stream(data, pos)
        b = (delta + state) & 0xFF
        out.append(b)
        fb = 0
        taps = 0b10010110
        for bit in range(8):
            if (taps >> bit) & 1:
                fb ^= (state >> bit) & 1
        state = ((state << 1) & 0xFF) | fb
    return bytes(out)

###############################################################################
# BBWT→MTF→Rice model with optional bitwise modules
###############################################################################

from functools import lru_cache
import itertools

def encode_bbwt_mtf_rice(block: bytes, use_bitplane: bool = False,
                         use_lfsr: bool = False,
                         use_nibble: bool = False,
                         use_bitrev: bool = False,
                         use_gray: bool = False,
                         rice_param: int = 2) -> Tuple[bytes, Dict[str, Any]]:
    """Encode a block using BBWT→MTF and optional bitwise transforms before Rice coding.

    Flags:
      * ``use_bitplane`` – apply bitplane interleaving.
      * ``use_lfsr``     – apply LFSR whitening.
      * ``use_nibble``   – swap nibbles of each byte.
      * ``use_bitrev``   – reverse bits within each byte.
    A small Rice parameter ``rice_param`` (k) is chosen empirically; a larger
    k may be used for more dispersed distributions.
    """
    # BBWT and MTF
    bbwt = bbwt_forward(block)
    mtf_list = mtf_encode(bbwt)
    # Partition into zeros and non‑zeros; record zero run lengths
    zeros: List[int] = []
    nonzero_vals: List[int] = []
    for val in mtf_list:
        zeros.append(val)
    seq_bytes = bytes(zeros)
    # apply bitwise transforms
    if use_bitplane:
        seq_bytes = bitplane_interleave(seq_bytes)
    if use_lfsr:
        seq_bytes = lfsr_whiten(seq_bytes)
    if use_nibble:
        seq_bytes = nibble_swap(seq_bytes)
    if use_bitrev:
        seq_bytes = bit_reverse(seq_bytes)
    if use_gray:
        seq_bytes = gray_encode_bytes(seq_bytes)
    # encode as Rice; choose k adaptively (here fixed)
    payload = rice_encode(list(seq_bytes), rice_param)
    flags = 0
    if use_bitplane: flags |= 1
    if use_lfsr:   flags |= 2
    if use_nibble: flags |= 4
    if use_bitrev: flags |= 8
    if use_gray:   flags |= 16
    meta = {"flags": flags, "k": rice_param, "length": len(seq_bytes), "orig_len": len(block)}
    return payload, meta

def decode_bbwt_mtf_rice(payload: bytes, meta: Dict[str, Any]) -> bytes:
    flags = meta["flags"]
    k = meta["k"]
    length = meta["length"]
    orig_len = meta["orig_len"]
    seq = rice_decode(payload, k, length)
    seq_bytes = bytes(seq)
    if flags & 16: seq_bytes = gray_decode_bytes(seq_bytes)
    if flags & 8:  seq_bytes = bit_reverse(seq_bytes)
    if flags & 4:  seq_bytes = nibble_swap(seq_bytes)
    if flags & 2:  seq_bytes = lfsr_whiten(seq_bytes)
    if flags & 1:  seq_bytes = bitplane_deinterleave(seq_bytes, length)
    mtf_list = list(seq_bytes)
    bbwt = mtf_decode(mtf_list)
    return bbwt_inverse(bbwt)

###############################################################################
# MDL model selection and container
###############################################################################

ModelEncoder = Callable[[bytes], Tuple[bytes, Dict[str, Any]]]
ModelDecoder = Callable[[bytes, int, Dict[str, Any]], bytes]

def encode_raw(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    return block, {}

def decode_raw(payload: bytes, length: int, meta: Dict[str, Any]) -> bytes:
    assert len(payload) == length
    return payload

def encode_xor(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    out = bytearray()
    prev = 0
    for b in block:
        out += uleb128_encode((b - prev) & 0xFF)
        prev = b
    return bytes(out), {}

def decode_xor(payload: bytes, length: int, meta: Dict[str, Any]) -> bytes:
    out = bytearray()
    prev = 0
    pos = 0
    for _ in range(length):
        delta, pos = uleb128_decode_stream(payload, pos)
        b = (prev + delta) & 0xFF
        out.append(b)
        prev = b
    return bytes(out)


# =========================================
# 头字段的模式打包/解包（兼容位）
# =========================================
MODE_FIXED = 0  # 固定分块
MODE_CDC   = 1  # FastCDC

def _pack_mode_and_size(mode: int, size: int) -> int:
    """
    用 block_size 字段的最高位作为模式位（1 bit），其余 31bit 写 size：
    - MODE_FIXED: size = block_size
    - MODE_CDC  : size = avg_size
    """
    if mode not in (MODE_FIXED, MODE_CDC):
        raise ValueError("invalid mode")
    if size < 0 or size > 0x7FFFFFFF:
        raise ValueError("size out of range (must fit in 31 bits)")
    return ((mode & 1) << 31) | (size & 0x7FFFFFFF)

def _unpack_mode_and_size(word: int) -> Tuple[int, int]:
    mode = (word >> 31) & 1
    size = word & 0x7FFFFFFF
    return mode, size

# Encoders select (ordered by increasing cost)
# Define candidate models for MDL selection.
# V2 removes the invalid pipelines (BBWT with LFSR mixing) and introduces a new pipeline ``encode_new_pipeline``.  
# Each candidate returns a payload and metadata; the smallest payload is selected for each block.
def _select_encoders() -> List[Tuple[Callable[[bytes], Tuple[bytes, Dict[str, Any]]], str]]:
    all_list: List[Tuple[Callable[[bytes], Tuple[bytes, Dict[str, Any]]], str]] = [
        (encode_raw, 'raw'),
        (encode_xor, 'xor'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, False, rice_param=2), 'bbwt'),
        (lambda b: encode_bbwt_mtf_rice(b, True,  False, False, False, False, rice_param=2), 'bbwt_bp'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, True,  False, False, rice_param=2), 'bbwt_nib'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, True,  False, rice_param=2), 'bbwt_br'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, True,  rice_param=2), 'bbwt_gray'),
        (encode_lz77, 'lz77'),
        (encode_lfsr_predict, 'lfsr_pred'),
        (repair_compress, 'repair'),
        (encode_new_pipeline, 'v2_new'),
    ]

    # 过滤 LZ77
    if G_NO_LZ77:
        all_list = [(enc, name) for (enc, name) in all_list if name != 'lz77']

    # 只用某个模型
    if G_ONLY_METHOD is not None:
        only = G_ONLY_METHOD.lower()
        all_list = [(enc, name) for (enc, name) in all_list if name.lower() == only]
        if not all_list:
            raise ValueError(f"--only={G_ONLY_METHOD} not found in candidates")

    return all_list

# Decoder select aligned with selected encoders. 
# Each entry must correspond to the model used during compression.  
# The order is:
#   0: raw
#   1: xor
#   2: bbwt
#   3: bbwt+bitplane
#   4: bbwt+nibble
#   5: bbwt+bitrev
#   6: bbwt+gray
#   7: lz77
#   8: lfsr predictor
#   9: re‑pair
#   10: V2 new pipeline
def _select_decoders() -> List[Callable[[bytes, int, Dict[str, Any]], bytes]]:
    return [
        decode_raw,
        decode_xor,
        lambda p, l, meta=None: decode_bbwt_mtf_rice(p, {"flags": 0,  "k": 2, "length": l, "orig_len": l}),
        lambda p, l, meta=None: decode_bbwt_mtf_rice(p, {"flags": 1,  "k": 2, "length": l, "orig_len": l}),
        lambda p, l, meta=None: decode_bbwt_mtf_rice(p, {"flags": 4,  "k": 2, "length": l, "orig_len": l}),
        lambda p, l, meta=None: decode_bbwt_mtf_rice(p, {"flags": 8,  "k": 2, "length": l, "orig_len": l}),
        lambda p, l, meta=None: decode_bbwt_mtf_rice(p, {"flags": 16, "k": 2, "length": l, "orig_len": l}),
        lambda p, l, meta=None: decode_lz77(p, l),
        lambda p, l, meta=None: decode_lfsr_predict(p, l),
        lambda p, l, meta=None: repair_decompress(p, l),
        decode_new_pipeline,
    ]

# =========================================
# 压缩（CDC 版；头字段含模式位）
# =========================================

def compress_blocks_cdc(data: bytes, min_size: int = 4096,
                        avg_size: int = 8192,
                        max_size: int = 16384) -> bytes:
    magic = b'KOLR'
    boundaries = cdc_fast_boundaries_strict(data, min_size, avg_size, max_size)
    out = bytearray()
    out += magic
    out += struct.pack('<I', _pack_mode_and_size(MODE_CDC, avg_size))
    out += struct.pack('<I', len(data))
    out += struct.pack('<H', len(boundaries))

    candidates = _select_encoders() if '_select_encoders' in globals() else _select_encoders()
    method_ids: List[int] = []
    orig_lens:  List[int] = []
    payloads:   List[bytes] = []
    payload_lens: List[int] = []
    
    nblocks = len(boundaries)
    _print_progress("Fast CDC", 0, nblocks)

    for idx, (start, end) in enumerate(boundaries, 1):
        block = data[start:end]
        best = (None, None, None)  # (size, payload, mid)
        for mid, (encoder, _name) in enumerate(candidates):
            try:
                payload, _meta = encoder(block)
            except Exception:
                continue
            size = len(payload)
            if best[0] is None or size < best[0]:
                best = (size, payload, mid)
        if best[1] is None:
            payload, _ = candidates[0][0](block)
            mid = 0
        else:
            mid = best[2]; payload = best[1]
        method_ids.append(mid)
        orig_lens.append(len(block))
        payloads.append(payload)
        payload_lens.append(len(payload))
        _print_progress("Fast CDC COMPRESS", idx, nblocks)
    _print_progress("FastCDC COMPRESS", nblocks, nblocks, final=True)

    total_payload = sum(payload_lens)
    # ---- Build TOC header (bytes) ----
    # RLE + Huffman for method ids
    run_syms, run_lens = _rle_ids(method_ids)
    from collections import Counter
    lengths = _huff_lengths(Counter(run_syms))
    enc_tbl, dec_tbl, maxlen = _huff_canonical(lengths)
    # Choose Rice k for run lengths (0..7) by brute force on bytes length with our bitwriter (approx by bits)
    best_k = 0; best_bits = 1<<60
    for k in range(8):
        bw_tmp = _BitWriter(); rice_write_values(bw_tmp, run_lens, k)
        _bytes, bits = bw_tmp.getvalue_bits()
        if bits < best_bits: best_bits = bits; best_k = k

    # CDC orig len scheme
    MODE = MODE_CDC

    toc_header = bytearray()
    # n_runs
    toc_header += uleb128_encode(len(run_syms))
    # codebook size K and (sym, length) pairs in canonical order
    K = len(enc_tbl)
    toc_header += uleb128_encode(K)
    for sym, L in sorted(lengths.items(), key=lambda kv:(kv[1], kv[0])):
        toc_header += uleb128_encode(sym)
        toc_header += uleb128_encode(L)
    # rice k for runs
    toc_header += uleb128_encode(best_k)
    if MODE == MODE_FIXED:
        # store only last block orig len
        last_len = orig_lens[-1] if nblocks>0 else 0
        toc_header += uleb128_encode(last_len)
    else:
        # CDC: ZigZag + Rice (choose k)
        # compute deltas
        avg_size = avg_size
        deltas = [_zz_enc(ol - avg_size) for ol in orig_lens]
        best_k2 = 0; best_bits2 = 1<<60
        for k in range(8):
            bw_tmp = _BitWriter(); rice_write_values(bw_tmp, deltas, k)
            _bytes, bits = bw_tmp.getvalue_bits()
            if bits < best_bits2: best_bits2 = bits; best_k2 = k
        toc_header += uleb128_encode(best_k2)

    # ---- Build TOC bitstream ----
    bw = _BitWriter()
    # method ids (runs) via Huffman
    _huff_encode_symbols(bw, enc_tbl, run_syms)
    # run lengths via Rice(best_k)
    rice_write_values(bw, run_lens, best_k)
    if MODE == MODE_CDC:
        # deltas via Rice(best_k2)
        rice_write_values(bw, deltas, best_k2)
    # payload cumulative ends via Elias–Fano
    P=[]; s=0
    for L in payload_lens:
        s += L; P.append(s)
    ef_write_positions(bw, P, total_payload)
    toc_bits, toc_bitlen = bw.getvalue_bits()

    # Write toc_hdr_len, toc_bitlen, total_payload
    out += uleb128_encode(len(toc_header))
    out += uleb128_encode(toc_bitlen)
    out += uleb128_encode(total_payload)

    # Append TOC header and TOC bits, then payloads
    out += toc_header
    out += toc_bits
    for payload in payloads:
        out += payload
    return bytes(out)

# =========================================
# 压缩（固定分块；头字段含模式位）
# =========================================

def compress_blocks_fixed(data: bytes, block_size: int = 8192) -> bytes:
    magic = b'KOLR'
    boundaries = fixed_boundaries(data, block_size)
    out = bytearray()
    out += magic
    out += struct.pack('<I', _pack_mode_and_size(MODE_FIXED, block_size))
    out += struct.pack('<I', len(data))
    out += struct.pack('<H', len(boundaries))

    candidates = _select_encoders() if '_select_encoders' in globals() else _select_encoders()
    method_ids: List[int] = []
    orig_lens:  List[int] = []
    payloads:   List[bytes] = []
    payload_lens: List[int] = []
    
    nblocks = len(boundaries)
    _print_progress("FIXED", 0, nblocks)

    for idx, (start, end) in enumerate(boundaries, 1):
        block = data[start:end]
        best = (None, None, None)  # (size, payload, mid)
        for mid, (encoder, _name) in enumerate(candidates):
            try:
                payload, _meta = encoder(block)
            except Exception:
                continue
            size = len(payload)
            if best[0] is None or size < best[0]:
                best = (size, payload, mid)
        if best[1] is None:
            payload, _ = candidates[0][0](block)
            mid = 0
        else:
            mid = best[2]; payload = best[1]
        method_ids.append(mid)
        orig_lens.append(len(block))
        payloads.append(payload)
        payload_lens.append(len(payload))
        
        # 进度更新
        _print_progress("FIXED COMPRESS", idx, nblocks)
    _print_progress("FIXED COMPRESS", nblocks, nblocks, final=True)

    total_payload = sum(payload_lens)
    # ---- Build TOC header (bytes) ----
    # RLE + Huffman for method ids
    run_syms, run_lens = _rle_ids(method_ids)
    from collections import Counter
    lengths = _huff_lengths(Counter(run_syms))
    enc_tbl, dec_tbl, maxlen = _huff_canonical(lengths)
    # Choose Rice k for run lengths (0..7) by brute force on bytes length with our bitwriter (approx by bits)
    best_k = 0; best_bits = 1<<60
    for k in range(8):
        bw_tmp = _BitWriter(); rice_write_values(bw_tmp, run_lens, k)
        _bytes, bits = bw_tmp.getvalue_bits()
        if bits < best_bits: best_bits = bits; best_k = k

    # CDC orig len scheme
    MODE = MODE_FIXED

    toc_header = bytearray()
    # n_runs
    toc_header += uleb128_encode(len(run_syms))
    # codebook size K and (sym, length) pairs in canonical order
    K = len(enc_tbl)
    toc_header += uleb128_encode(K)
    for sym, L in sorted(lengths.items(), key=lambda kv:(kv[1], kv[0])):
        toc_header += uleb128_encode(sym)
        toc_header += uleb128_encode(L)
    # rice k for runs
    toc_header += uleb128_encode(best_k)
    if MODE == MODE_FIXED:
        # store only last block orig len
        last_len = orig_lens[-1] if nblocks>0 else 0
        toc_header += uleb128_encode(last_len)
    else:
        # CDC: ZigZag + Rice (choose k)
        # compute deltas
        avg_size = avg_size
        deltas = [_zz_enc(ol - avg_size) for ol in orig_lens]
        best_k2 = 0; best_bits2 = 1<<60
        for k in range(8):
            bw_tmp = _BitWriter(); rice_write_values(bw_tmp, deltas, k)
            _bytes, bits = bw_tmp.getvalue_bits()
            if bits < best_bits2: best_bits2 = bits; best_k2 = k
        toc_header += uleb128_encode(best_k2)

    # ---- Build TOC bitstream ----
    bw = _BitWriter()
    # method ids (runs) via Huffman
    _huff_encode_symbols(bw, enc_tbl, run_syms)
    # run lengths via Rice(best_k)
    rice_write_values(bw, run_lens, best_k)
    if MODE == MODE_CDC:
        # deltas via Rice(best_k2)
        rice_write_values(bw, deltas, best_k2)
    # payload cumulative ends via Elias–Fano
    P=[]; s=0
    for L in payload_lens:
        s += L; P.append(s)
    ef_write_positions(bw, P, total_payload)
    toc_bits, toc_bitlen = bw.getvalue_bits()

    # Write toc_hdr_len, toc_bitlen, total_payload
    out += uleb128_encode(len(toc_header))
    out += uleb128_encode(toc_bitlen)
    out += uleb128_encode(total_payload)

    # Append TOC header and TOC bits, then payloads
    out += toc_header
    out += toc_bits
    for payload in payloads:
        out += payload
    return bytes(out)

# =========================================
# 解压（兼容旧容器；可读取模式位）
# =========================================

def decompress(container: bytes) -> bytes:
    pos = 0
    if len(container) < 4 or container[:4] != b'KOLR':
        raise ValueError('Invalid magic')
    pos = 4
    packed = struct.unpack_from('<I', container, pos)[0]; pos += 4
    mode, size_field = _unpack_mode_and_size(packed)
    total_len  = struct.unpack_from('<I', container, pos)[0]; pos += 4
    nblocks    = struct.unpack_from('<H', container, pos)[0]; pos += 2

    # New format: toc header length (bytes), toc_bitlen (bits), total_payload (bytes)
    toc_hdr_len, pos = uleb128_decode_stream(container, pos)
    toc_bitlen, pos  = uleb128_decode_stream(container, pos)
    total_payload, pos = uleb128_decode_stream(container, pos)

    # Read TOC header and bitstream
    if pos + toc_hdr_len > len(container):
        raise ValueError("Truncated TOC header")
    toc_header = container[pos:pos+toc_hdr_len]; pos += toc_hdr_len
    toc_bit_bytes = (toc_bitlen + 7) // 8
    if pos + toc_bit_bytes > len(container):
        raise ValueError("Truncated TOC bits")
    toc_bits = container[pos:pos+toc_bit_bytes]; pos += toc_bit_bytes

    # Parse TOC header (byte-level fields)
    p = 0
    n_runs, p = uleb128_decode_stream(toc_header, p)
    K, p = uleb128_decode_stream(toc_header, p)
    lengths = {}
    for _ in range(K):
        sym, p = uleb128_decode_stream(toc_header, p)
        L, p   = uleb128_decode_stream(toc_header, p)
        lengths[sym] = L
    k_runs, p = uleb128_decode_stream(toc_header, p)

    if mode == MODE_FIXED:
        last_orig_len, p = uleb128_decode_stream(toc_header, p)
    else:
        k_orig, p = uleb128_decode_stream(toc_header, p)

    # Decode TOC bitstreams
    enc_tbl, dec_tbl, maxlen = _huff_canonical(lengths)
    br = _BitReader(toc_bits, 0, 0)
    run_syms = _huff_decode_symbols(br, dec_tbl, maxlen, n_runs)
    run_lens = rice_read_n(br, k_runs, n_runs)

    # Expand method_ids
    method_ids = []
    for s, r in zip(run_syms, run_lens):
        method_ids.extend([s]*r)
    if len(method_ids) != nblocks:
        raise ValueError("Method id RLE expands to wrong size")

    # orig_lens
    if mode == MODE_FIXED:
        block_size = size_field & 0x7FFFFFFF
        orig_lens = [block_size]*(nblocks-1) + ([last_orig_len] if nblocks>0 else [])
    else:
        avg_size = size_field & 0x7FFFFFFF
        deltas = rice_read_n(br, k_orig, nblocks)
        orig_lens = [avg_size + _zz_dec(x) for x in deltas]

    # payload cumulative ends via Elias–Fano
    P = ef_read_positions(br, total_payload, nblocks)
    if P and P[-1] != total_payload:
        raise ValueError("Payload EF sum mismatch")

    # Now read payload area
    if pos + total_payload > len(container):
        raise ValueError("Truncated payload area")
    data_payload = container[pos:pos+total_payload]; pos += total_payload

    # Slice per block and decode
    decoders = _select_decoders() if '_select_decoders' in globals() else _candidate_decoders()
    out = bytearray()
    start = 0
    
    _print_progress("DECOMPRESS", 0, nblocks)
    
    for i in range(nblocks):
        pay_end = P[i]
        pay_len = pay_end - start
        payload = data_payload[start:pay_end]
        start = pay_end
        mid = method_ids[i]
        orig_len = orig_lens[i]
        if mid < 0 or mid >= len(decoders):
            raise ValueError(f'Unknown method_id {mid}')
        block = decoders[mid](payload, orig_len, {})
        out += block
        
        _print_progress("DECOMPRESS", i + 1, nblocks)
    _print_progress("DECOMPRESS", nblocks, nblocks, final=True)

    if len(out) != total_len:
        raise ValueError(f'Length mismatch: got {len(out)}, expect {total_len}')
    if pos != len(container):
        # allow trailing zeros? be strict:
        raise ValueError(f'Extra trailing {len(container)-pos} bytes after container end')
    return bytes(out)


###############################################################################
# Built‑in experiment: compare models
###############################################################################

def run_experiment() -> None:
    """跑一组内置数据集，按当前候选表评估体积比；优先画图，缺依赖则打印表格。"""
    datasets = {
        'text': (
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
            "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
            "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
            "hobbit-hole, and that means comfort."
        ).encode('utf-8') * 10,
        'random': bytes(random.getrandbits(8) for _ in range(10240)),
        'repetitive': b'a' * 20480,
    }

    encoders = _select_encoders()  # [(enc, name), ...] 已考虑 --no-lz77 / --only
    names = [name for _, name in encoders]

    ratios: Dict[str, List[float]] = {ds: [] for ds in datasets}
    for ds_name, data in datasets.items():
        for enc, enc_name in encoders:
            try:
                payload, _meta = enc(data)
                size = len(payload)
            except Exception as e:
                # 出错时把比率记为 1.0（视同原样）；同时给出简短提示
                size = len(data)
                print(f"[warn] encoder '{_name}' on '{ds_name}' raised: {e}")
            ratios[ds_name].append(size / len(data) if len(data) else 1.0)

    # 优先画图；如果 matplotlib/numpy 不在环境里，则退化为表格输出
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(8, 1.2*len(names)), 6))
        x = np.arange(len(datasets))
        bar_width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            vals = [ratios[ds][i] for ds in datasets]
            ax.bar(x + i * bar_width, vals, bar_width, label=name)
        ax.set_xticks(x + bar_width * (len(names) - 1) / 2)
        ax.set_xticklabels(list(datasets.keys()))
        ax.set_ylabel('Compressed size / Original size (lower is better)')
        ax.set_title('Model compression ratios across datasets')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        outpng = './kolm_researched_v2_ratios.png'
        plt.savefig(outpng)
        print(f'Experiment completed, plot saved to {outpng}')
    except Exception:
        # 退化：打印等宽表格
        colw = max(len(n) for n in names + ["model"])
        print("\n== Experiment (text / random / repetitive) ==")
        header = f"{'model'.ljust(colw)}  {'text':>8}  {'random':>8}  {'repetitive':>11}"
        print(header)
        print('-' * len(header))
        for i, name in enumerate(names):
            t = ratios['text'][i]
            r = ratios['random'][i]
            rep = ratios['repetitive'][i]
            print(f"{name.ljust(colw)}  {t:8.3f}  {r:8.3f}  {rep:11.3f}")

###############################################################################
# CLI
###############################################################################

if __name__ == '__main__':
    import argparse, sys, os, random  # random 供 run_experiment 使用
    parser = argparse.ArgumentParser(description='Kolmogorov researched compressor')

    parser.add_argument('-i', '--input', nargs='?', help='Input file to compress or decompress')
    parser.add_argument('-d', '--decompress', action='store_true', help='Decompress')
    parser.add_argument('-o', '--output', help='Output file')

    # 统一块参数：固定模式下为固定块大小；FastCDC 模式下作为 avg_size 使用
    parser.add_argument('-b', '--block', type=int, default=2048,
                        help='Target block size (FIXED) or avg_size (FastCDC)')

    # FastCDC 模式开关（大小写都支持）
    parser.add_argument('--FastCDC', '--fastcdc', dest='fastcdc', action='store_true', help='Use Fast Content-Defined Chunking (avg_size = --block). When off, use fixed-size chunking.')

    # 实验开关保留
    parser.add_argument('--experiment', action='store_true', help='Run built-in experiment')

    parser.add_argument('--no-lz77', action='store_true',
                        help='Disable LZ77 candidate (affects selection and experiment)')
    parser.add_argument('--only', type=str, default=None,
                        help='Only use a single model by name (e.g. v2_new, lz77, raw)')

    parser.add_argument('--progress', action='store_true',
                    help='Show per-block progress like C++')

    args = parser.parse_args()

    # 设置全局开关（放在最后一次 parse_args() 之后）
    G_NO_LZ77   = bool(args.no_lz77)
    G_ONLY_METHOD = args.only
    G_PROGRESS  = bool(args.progress)

    if args.experiment:
        run_experiment()
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(0)

    if args.decompress:
        data = open(args.input, 'rb').read()
        out = decompress(data)
        outname = args.output or (os.path.splitext(args.input)[0] + '.out')
        with open(outname, 'wb') as f:
            f.write(out)
        print(f'Decompressed {len(data)} bytes to {len(out)} bytes → {outname}')
    else:
        data = open(args.input, 'rb').read()

        if args.fastcdc:
            # 由 --block 作为 avg_size；自动派生 min/max，确保 0 < min ≤ avg ≤ max
            avg = max(64, args.block)
            min_size = max(64, min(avg, avg // 2 if avg >= 2 else 64))
            max_size = max(avg, avg * 2)

            blob = compress_blocks_cdc(data, min_size=min_size, avg_size=avg, max_size=max_size)
            mode_desc = f'FastCDC(min={min_size}, avg={avg}, max={max_size})'
        else:
            blob = compress_blocks_fixed(data, block_size=args.block)
            mode_desc = f'FIXED(block={args.block})'

        outname = args.output or (args.input + '.kolr')
        with open(outname, 'wb') as f:
            f.write(blob)

        ratio = len(blob) / len(data) if len(data) else 1.0
        print(f'[{mode_desc}] Compressed {len(data)} bytes to {len(blob)} bytes (ratio {ratio:.3f}) → {outname}')
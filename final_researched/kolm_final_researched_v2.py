#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kolm_final_researched.py
========================

This module implements a deeply researched prototype compressor/decompressor
guided by the "前布尔电路后统计概率" (boolean‑circuit first, probability last)
philosophy.  The aim is to expose and remove as much structure as possible
using cheap, reversible bitwise transforms before handing the residual data
to classical entropy coders.  All transformations are written from scratch
without relying on external compression libraries.  In addition to the
bijective BWT and move‑to‑front (MTF) pipeline, this version introduces
several novel binary circuit modules such as bit‑plane interleaving,
LFSR whitening, nibble swapping and bit reversal.  A model selector
chooses among these transformations and simple dictionary coders based
on a minimum description length (MDL) criterion.

Features provided:

  * Content‑defined chunking via a simplified FastCDC (gear hash) implementation.
  * Duval's linear time Lyndon factorisation and a bijective BWT (BBWT)
    constructed by merging the cyclic rotations of each factor.  The
    theoretical basis can be found in CP‑Algorithms and BWT literature【154237816494091†L294-L301】【555806496076120†L507-L516】.
  * Move‑to‑front encoding to bring repeated symbols closer to zero.
  * Bitwise reversible modules:
      ‑ **Bit plane interleaving**: interleave the bitplanes of successive
        bytes to group high‑order bits together.
      ‑ **LFSR whitening**: treat the byte stream as a sequence of 8‑bit
        states and predict the next bit via a small LFSR; XORing the
        predicted bit sequence removes linear patterns【879695488005067†L166-L179】.
      ‑ **Nibble swapping**: swap the high and low 4‑bit nibbles of each
        byte; applying twice returns the original.
      ‑ **Bit reversal**: reverse the bit order within each byte using a
        precomputed lookup table.
  * Integer coders: Rice/Golomb encoding for geometric distributions【966564189297361†L152-L170】,
    Elias gamma/δ codes for metadata, and ULEB128 for variable length
    integers.
  * Lightweight dictionary coders: a naive LZ77 encoder/decoder and a
    simple Re‑Pair grammar compressor【967225900425034†L139-L166】 to handle long
    repeats.
  * An MDL‑based model selector that chooses the smallest representation
    among raw, XOR, BBWT→Rice (with any combination of bitwise modules),
    LZ77, LFSR predictor and Re‑Pair.
  * A simple container format beginning with ``b'KOLR'`` (Kolmogorov
    Learned Researcher) followed by block descriptors.

The encoder works on blocks emitted by the CDC stage.  Each block is
independently transformed and encoded; metadata describing the chosen
model and its parameters is recorded in the header.  The decoder uses
this metadata to invert the transforms and reconstruct the original
input.  All code paths are fully lossless and reversible.

Usage examples:
    python3 kolm_final_researched.py input.bin        # compress
    python3 kolm_final_researched.py -d input.kolr    # decompress
    python3 kolm_final_researched.py --experiment     # run built‑in experiments

This file is self‑contained and can be studied as an educational
reference on combining string algorithms and bit tricks for data
compression.
"""

from __future__ import annotations
from collections import Counter
import math
import struct
import random
from typing import List, Tuple, Dict, Callable, Optional, Any

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
# FastCDC content‑defined chunking (simplified)
###############################################################################

def _gear_table(seed: int = 2025) -> List[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(256)]

_GEAR = _gear_table()

def cdc_fast_boundaries(data: bytes, min_size: int = 4096, avg_size: int = 8192,
                        max_size: int = 16384) -> List[Tuple[int, int]]:
    """Identify cut boundaries using a fast gear hash (simplified FastCDC).

    FastCDC chooses a mask to achieve a target average chunk size and
    searches for positions where the rolling hash masked bits are zero【629903081498632†L40-L52】.  If no
    boundary is found before ``max_size``, the chunk is forcibly cut.
    """
    n = len(data)
    if n == 0:
        return []
    # choose mask bits based on average size
    k = max(6, min(20, (avg_size).bit_length() - 1))
    mask = (1 << k) - 1
    boundaries: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        start = i
        h = 0
        end_min = min(n, start + min_size)
        end_max = min(n, start + max_size)
        i = end_min
        while i < end_max:
            h = ((h << 1) & 0xFFFFFFFF) + _GEAR[data[i]]
            if (h & mask) == 0:
                i += 1
                break
            i += 1
        boundaries.append((start, i))
    return boundaries

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


def encode_new_pipeline(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    V2 pipeline with circuit_map_automaton + per-plane BBWT + RLE + Rice.
      - Block bypass on ORIGINAL block.
      - Apply circuit_map_automaton_forward() to get (mapped, theta={'mode':m})
      - Split to 8 bitplanes
      - Per-plane: bypass or BBWT → RLE → Rice
      - Header: [flag][L][mode][per-plane tags...][payload]
    """
    if not block:
        return bytes([1]), {}

    # Block-level bypass (original block stats)
    def byte_entropy(b: bytes) -> float:
        if not b: return 0.0
        cnt = Counter(b)
        n = len(b); h = 0.0
        for v in cnt.values():
            p = v/n; h -= p*math.log2(p)
        return h
    def avg_run_bytes(b: bytes) -> float:
        if not b: return 0.0
        runs = 1; prev = b[0]
        for x in b[1:]:
            if x != prev: runs += 1; prev = x
        return len(b)/runs

    H0 = byte_entropy(block); avgR = avg_run_bytes(block)
    if H0 >= 7.95 and avgR <= 1.05:
        return bytes([1]) + block, {}

    # circuit automaton (reversible) on bytes
    mapped, theta = circuit_map_automaton_forward(block)
    mode = theta.get('mode', 0) & 0xFF

    # Bitplanes
    planes, L = bytes_to_bitplanes(mapped)

    header = bytearray()
    payload = bytearray()
    header.append(0)                 # flag=COMP
    header += uleb128_encode(L)      # symbols per plane
    header.append(mode)              # <-- store automaton mode

    for j in range(8):
        Uj = planes[j]
        if H0_bits(Uj) >= 0.99 and avg_run_bits(Uj) <= 1.02:
            header.append(0x00)  # RAW plane
            packed = pack_bits_to_bytes(Uj)
            header += uleb128_encode(len(packed))
            payload += packed
            continue
        Lj = bbwt_forward(bytes(Uj))
        Lj_bits = list(Lj)
        b1, runs = rle_binary(Lj_bits)
        mean_r = (sum(runs)/len(runs)) if runs else 1.0
        k = int(max(0, math.floor(math.log2(mean_r)))) if mean_r>0 else 0
        rb = rice_encode(runs, k)
        header.append(0x01)       # BBWT+RLE+Rice
        header.append(b1 & 1)
        header.append(k & 0xFF)
        header += uleb128_encode(len(runs))
        header += uleb128_encode(len(rb))
        payload += rb

    return bytes(header) + bytes(payload), {}

def decode_new_pipeline(payload: bytes, orig_len: int, meta: Dict[str, Any]) -> bytes:
    """
    Inverse of encode_new_pipeline with circuit_map_automaton.
    """
    if not payload:
        return b""
    pos = 0
    flag = payload[pos]; pos += 1
    if flag == 1:
        return payload[pos:pos+orig_len]

    L, pos = uleb128_decode_stream(payload, pos)
    mode = payload[pos]; pos += 1

    descs = []
    for _ in range(8):
        tag = payload[pos]; pos += 1
        if tag == 0x00:
            nbytes, pos = uleb128_decode_stream(payload, pos)
            descs.append(('raw', nbytes))
        elif tag == 0x01:
            b1 = payload[pos]; k = payload[pos+1]; pos += 2
            run_count, pos = uleb128_decode_stream(payload, pos)
            paylen, pos = uleb128_decode_stream(payload, pos)
            descs.append(('enc', b1, k, run_count, paylen))
        else:
            raise ValueError("Unknown plane tag")

    data_pos = pos
    planes = []
    for d in descs:
        if d[0] == 'raw':
            nbytes = d[1]
            buf = payload[data_pos:data_pos+nbytes]; data_pos += nbytes
            Uj = unpack_bits_from_bytes(buf, L)
            planes.append(Uj)
        else:
            _, b1, k, run_count, paylen = d
            rice_buf = payload[data_pos:data_pos+paylen]; data_pos += paylen
            runs = rice_decode(rice_buf, k, run_count)
            if len(runs) != run_count:
                raise ValueError("run_count mismatch")
            if sum(runs) != L:
                raise ValueError("RLE runs sum != plane length L")
            Lj_bits = unrle_binary(b1, runs)
            Uj_bytes = bbwt_inverse(bytes(Lj_bits))
            Uj = list(Uj_bytes)
            if len(Uj) != L:
                Uj = Uj[:L] if len(Uj) > L else Uj + [0]*(L-len(Uj))
            planes.append(Uj)

    # Bitplanes
    mapped = bitplanes_to_bytes(planes)
    block = circuit_map_automaton_inverse(mapped, {'mode': mode})
    return block


def nibble_swap(data: bytes) -> bytes:
    """Swap the high and low 4‑bit nibbles of each byte."""
    return bytes(((b & 0x0F) << 4) | ((b & 0xF0) >> 4) for b in data)

_BIT_REVERSE_TABLE = bytes(int('{:08b}'.format(i)[::-1], 2) for i in range(256))

def bit_reverse(data: bytes) -> bytes:
    """Reverse the bit order of each byte using a lookup table."""
    return bytes(_BIT_REVERSE_TABLE[b] for b in data)

###############################################################################
# Integer coders (Rice/Golomb and Elias)
###############################################################################

def gamma_encode(n: int) -> bytes:
    """Elias gamma code for positive integers."""
    assert n > 0
    b = n.bit_length()
    return b'0' * (b - 1) + format(n, f'b').encode('ascii')

def gamma_decode(bitstr: str, pos: int) -> Tuple[int, int]:
    i = pos
    while i < len(bitstr) and bitstr[i] == '0':
        i += 1
    l = i - pos + 1
    value = int('1' + bitstr[i + 1 - 1:i + l], 2)
    return value, i + l



def rice_encode(seq: List[int], k: int) -> bytes:
    """Encode a list of non‑negative integers using Rice coding with parameter 2^k.

    For k == 0, the code degenerates to pure unary (n = '1'*n + '0'), with
    no remainder bits. We must not emit any extra '0' here, otherwise the
    decoder will recover spurious zeros between symbols.
    """
    out_bits = []
    M = 1 << k
    for n in seq:
        q = n // M
        r = n % M
        out_bits.append('1' * q + '0')
        if k > 0:
            out_bits.append(format(r, f'0{k}b'))
    bitstr = ''.join(out_bits)
    pad = (8 - len(bitstr) % 8) % 8
    bitstr += '0' * pad
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i + 8], 2))
    return bytes(out)

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

###############################################################################
# Naive LZ77 and Re‑Pair coders
###############################################################################

def encode_lz77(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Encode ``block`` using a simplistic LZ77 with ULEB128 coding."""
    window = bytearray()
    out = bytearray()
    pos = 0
    n = len(block)
    while pos < n:
        best_len = 0
        best_dist = 0
        # search limited window
        for dist in range(1, min(len(window), 255) + 1):
            length = 0
            while (length < 255 and pos + length < n and
                   window[-dist + length] == block[pos + length]):
                length += 1
            if length > best_len:
                best_len = length
                best_dist = dist
        if best_len >= 3:
            out.append(1)  # marker for match
            out += uleb128_encode(best_len)
            out += uleb128_encode(best_dist)
            for _ in range(best_len):
                window.append(block[pos])
                pos += 1
        else:
            out.append(0)  # literal
            out.append(block[pos])
            window.append(block[pos])
            pos += 1
    return bytes(out), {}

def decode_lz77(data: bytes, orig_len: int) -> bytes:
    window = bytearray()
    out = bytearray()
    pos = 0
    i = 0
    n = len(data)
    while i < n:
        flag = data[i]; i += 1
        if flag == 0:
            b = data[i]; i += 1
            out.append(b)
            window.append(b)
        else:
            length, i = uleb128_decode_stream(data, i)
            dist, i = uleb128_decode_stream(data, i)
            for _ in range(length):
                b = window[-dist]
                out.append(b)
                window.append(b)
        # keep window manageable
        if len(window) > 4096:
            del window[:-4096]
    assert len(out) == orig_len
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

def compress_blocks(data: bytes, block_size: int = 8192) -> bytes:
    magic = b'KOLR'
    boundaries = cdc_fast_boundaries(data, avg_size=block_size)
    out = bytearray()
    out += magic
    out += struct.pack('<I', block_size)
    out += struct.pack('<I', len(data))
    out += struct.pack('<H', len(boundaries))
    # candidate encoders (ordered by increasing cost)
    # Define candidate models for MDL selection.  V2 removes the invalid
    # pipelines (BBWT with LFSR mixing) and introduces a new pipeline
    # ``encode_new_pipeline``.  Each candidate returns a payload and
    # metadata; the smallest payload is selected for each block.
    candidates: List[Tuple[ModelEncoder, str]] = [
        (encode_raw, 'raw'),
        (encode_xor, 'xor'),
        # base BBWT→MTF→Rice (no bitwise module)
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, False, rice_param=2), 'bbwt'),
        # BBWT with bitplane interleaving
        (lambda b: encode_bbwt_mtf_rice(b, True, False, False, False, False, rice_param=2), 'bbwt_bp'),
        # BBWT with nibble swap
        (lambda b: encode_bbwt_mtf_rice(b, False, False, True, False, False, rice_param=2), 'bbwt_nib'),
        # BBWT with bit reversal
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, True, False, rice_param=2), 'bbwt_br'),
        # BBWT with Gray code
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, True, rice_param=2), 'bbwt_gray'),
        (encode_lz77, 'lz77'),
        (encode_lfsr_predict, 'lfsr_pred'),
        (repair_compress, 'repair'),
        # New V2 pipeline
        (encode_new_pipeline, 'v2_new'),
    ]
    # mapping from method index to decoder
    decoders: List[Callable[[bytes, int, Dict[str, Any]], bytes]] = []
    method_ids: Dict[str, int] = {}
    # fill decoders after building out, and record method names
    # we will assign incremental IDs as we append
    encodings: List[Tuple[int, int, bytes]] = []  # (method_id, orig_len, payload)
    for start, end in boundaries:
        block = data[start:end]
        best_size = None
        best_payload = None
        best_meta = None
        best_id = None
        # try each candidate
        for mid, (encoder, name) in enumerate(candidates):
            try:
                payload, meta = encoder(block)
            except Exception:
                continue
            size = len(payload)
            if best_size is None or size < best_size:
                best_size = size
                best_payload = payload
                best_meta = meta
                best_id = mid
        # record
        encodings.append((best_id, len(block), best_payload))
    # write encodings
    for method_id, orig_len, payload in encodings:
        out.append(method_id)
        out += struct.pack('<I', orig_len)
        out += struct.pack('<I', len(payload))
        out += payload
    return bytes(out)

def decompress(data: bytes) -> bytes:
    pos = 0
    if data[:4] != b'KOLR':
        raise ValueError('Invalid magic')
    pos = 4
    block_size = struct.unpack_from('<I', data, pos)[0]; pos += 4
    total_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
    nblocks = struct.unpack_from('<H', data, pos)[0]; pos += 2
    # Decoder list aligned with candidate encoders.  Each entry must
    # correspond to the model used during compression.  The order is:
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
    decoders: List[Callable[[bytes, int, Dict[str, Any]], bytes]] = [
        decode_raw,
        decode_xor,
        # bbwt base (flags 0)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":0,"k":2,"length":l,"orig_len":l}),
        # bbwt + bitplane (flags 1)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":1,"k":2,"length":l,"orig_len":l}),
        # bbwt + nibble swap (flags 4)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":4,"k":2,"length":l,"orig_len":l}),
        # bbwt + bit reverse (flags 8)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":8,"k":2,"length":l,"orig_len":l}),
        # bbwt + gray code (flags 16)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":16,"k":2,"length":l,"orig_len":l}),
        lambda p,l,meta=None: decode_lz77(p, l),
        lambda p,l,meta=None: decode_lfsr_predict(p, l),
        lambda p,l,meta=None: repair_decompress(p, l),
        decode_new_pipeline,
    ]
    out = bytearray()
    for _ in range(nblocks):
        method_id = data[pos]; pos += 1
        orig_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
        payload_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
        payload = data[pos:pos + payload_len]; pos += payload_len
        block = decoders[method_id](payload, orig_len, {})
        out += block
    assert len(out) == total_len
    return bytes(out)

###############################################################################
# Built‑in experiment: compare models
###############################################################################

def run_experiment() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # Prepare datasets: text, random, repetitive
    text = (
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
        "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
        "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
        "hobbit‑hole, and that means comfort."
    ).encode('utf-8') * 10
    random_bytes = bytes(random.getrandbits(8) for _ in range(10240))
    repetitive = b'a' * 20480
    datasets = {
        'text': text,
        'random': random_bytes,
        'repetitive': repetitive,
    }
    # candidate names aligned with decoders index ordering above
    # Model names aligned with the candidate list in V2.  We remove
    # invalid BBWT+LFSR variants and add the new V2 pipeline.
    names = [
        'Raw', 'XOR', 'BBWT', 'BBWT+Bitplane',
        'BBWT+Nibble', 'BBWT+BitRev', 'BBWT+Gray',
        'LZ77', 'LFSR predictor', 'Re‑Pair', 'V2 New'
    ]
    # Map names to encoder functions (order matches decoders).  Note
    # that ``encode_bbwt_mtf_rice`` expects flags; we omit LFSR and
    # bitplane+LFSR combinations.
    encoders = [
        encode_raw,
        encode_xor,
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, True, False, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, True, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, True, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, True),
        encode_lz77,
        encode_lfsr_predict,
        repair_compress,
        encode_new_pipeline,
    ]
    ratios: Dict[str, List[float]] = {ds: [] for ds in datasets}
    for ds_name, data in datasets.items():
        for enc in encoders:
            try:
                payload, meta = enc(data)
                size = len(payload)
            except Exception:
                size = len(data)
            ratio = size / len(data) if len(data) else 1.0
            ratios[ds_name].append(ratio)
    # plot as grouped bar chart with English labels to avoid font issues
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    bar_width = 0.08
    for i, name in enumerate(names):
        vals = [ratios[ds][i] for ds in datasets]
        ax.bar(x + i * bar_width, vals, bar_width, label=name)
    ax.set_xticks(x + bar_width * (len(names) - 1) / 2)
    ax.set_xticklabels(list(datasets.keys()))
    ax.set_ylabel('Compressed size / Original size')
    ax.set_title('Model compression ratios across datasets (lower is better)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./kolm_researched_v2_ratios.png')
    print('Experiment completed, plot saved to kolm_researched_v2_ratios.png')

###############################################################################
# CLI
###############################################################################

if __name__ == '__main__':
    import argparse, sys, os
    parser = argparse.ArgumentParser(description='Kolmogorov researched compressor')
    parser.add_argument('input', nargs='?', help='Input file to compress or decompress')
    parser.add_argument('-d', '--decompress', action='store_true', help='Decompress')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-b', '--block', type=int, default=8192, help='Target block size')
    parser.add_argument('--experiment', action='store_true', help='Run built‑in experiment')
    args = parser.parse_args()
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
        blob = compress_blocks(data, block_size=args.block)
        outname = args.output or (args.input + '.kolr')
        with open(outname, 'wb') as f:
            f.write(blob)
        ratio = len(blob) / len(data) if len(data) else 1.0
        print(f'Compressed {len(data)} bytes to {len(blob)} bytes (ratio {ratio:.3f}) → {outname}')
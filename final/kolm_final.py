#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kolm_final.py -- Final prototype compressor based on the "Kolmogorov/MDL and bit‑circuit" design.

This module contains a self‑contained reference implementation of a block
compressor/decompressor designed to showcase the "electrical circuits as
codes" ethos.  Each block of data is evaluated with several lightweight
models and the model yielding the shortest description (model bits +
payload bits) is selected.  The implementation deliberately favours
primitive, bit‑friendly operations (XOR, cyclic shifts, Lyndon factorisation)
over heavy context modelling.  It also retains a fast bijective BWT
(BBWT) forward transform based on Lyndon factorisation and a simple
suffix array on doubled factors.  A naive LZ77 encoder is included for
comparison.  Random or already compressed blocks will automatically
fall back to a RAW copy.

### Container format

The top‑level output begins with a 4‑byte magic string ``b'KOLM'``
followed by:

* ``u32 block_size`` – nominal block target size used when slicing the input.
* ``u64 total_len`` – total length of the original input.
* ``u16 nblocks`` – number of blocks encoded.

Each block then consists of:

* ``u8 method_id`` – selects the decoding routine.
* ``u32 orig_len`` – original block length (for decode).
* ``u32 payload_len`` – length of encoded payload.
* ``payload`` – model‑specific encoding.

This simple container allows the decompressor to reconstruct each
block independently.

### Models

Four candidate models are provided:

* **0 – RAW**: no compression, the payload is the original block.
* **1 – XOR**: differences between successive bytes are encoded via
  unsigned LEB128 integers.  This exploits locality in slowly
  changing sequences while remaining extremely fast.
* **2 – BBWT→MTF**: a bijective Burrows–Wheeler transform is
  computed on the block via Lyndon factorisation.  The resulting
  sequence is move‑to‑front (MTF) encoded and the zero/non‑zero
  elements are coded using Rice or Elias‑γ codes, similar to the
  ``Model 1`` in earlier prototypes.  This model shines on text or
  structured data with lots of small repeats.
* **3 – LZ77**: a naive sliding‑window LZ77 coder which emits either
  literal bytes or (length,distance) pairs encoded as LEB128.  This
  catches medium‑length repetitions and complements the BBWT model.

Each block is passed through the above models in order of
computational cost (cheap to more expensive).  For each candidate
payload the length is measured; the model producing the smallest
payload is selected.  The encoder writes the ``method_id`` and
``orig_len`` for the block along with the chosen payload.  A simple
entropy guard based on sampled Shannon entropy skips heavy models on
high‑entropy blocks.

This code is purely Python for ease of comprehension.  In a real
implementation one would replace the naïve suffix array with a
production‐grade SA/BBWT routine, tighten the LZ77 matcher, and use
SIMD‑friendly bit packing.  Nevertheless, this prototype faithfully
expresses the design philosophy: favouring short, transparent
computations that push low Kolmogorov complexity to the front of the
encoding pipeline.
"""

from __future__ import annotations

import math
import struct
from typing import List, Tuple, Dict, Optional, Callable

###############################################################################
# Utility functions
###############################################################################

def uleb128_encode(n: int) -> bytes:
    """Encode a non‑negative integer into unsigned LEB128.

    Each byte contributes 7 bits of payload with the MSB signalling
    continuation.  Numbers up to 127 produce a single byte, larger
    values use more bytes.  Adapted from the hybrid prototype.
    """
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)

def uleb128_decode_stream(data: bytes, pos: int = 0) -> Tuple[int, int]:
    """Decode a LEB128 value from data starting at position ``pos``.

    Returns a tuple of (value, new_pos).  Raises ``EOFError`` on
    truncated input.
    """
    result = 0
    shift = 0
    i = pos
    while True:
        if i >= len(data):
            raise EOFError("Truncated ULEB128")
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, i

def sample_shannon_entropy(data: bytes, sample_rate: int = 32) -> float:
    """Estimate Shannon entropy over a small sample of bytes.

    The block is subsampled to avoid scanning the entire input when
    deciding whether the block is compressible.  This function
    computes the empirical entropy H(X) = -sum(p_i log2 p_i) over the
    sampled distribution.  Values near 8 bits/byte indicate high
    entropy (likely incompressible) data.
    """
    n = len(data)
    if n == 0:
        return 0.0
    # choose a stride such that at most sample_rate samples are used
    step = max(1, n // sample_rate)
    hist: Dict[int, int] = {}
    for b in data[::step]:
        hist[b] = hist.get(b, 0) + 1
    H = 0.0
    total = len(data[::step])
    for cnt in hist.values():
        p = cnt / total
        H -= p * math.log2(p)
    return H

###############################################################################
# Content defined chunking
###############################################################################

def _gear_table(seed: int = 2025) -> List[int]:
    """Generate a 256‑entry table for a simple gear hash.

    The same table is used at both encode and decode time.  A
    deterministic seed avoids exposing randomness across runs.
    """
    rng = _gear_table._rng  # type: ignore[attr-defined]
    rng.seed(seed)
    return [rng.getrandbits(32) for _ in range(256)]

_gear_table._rng = __import__("random").Random()  # type: ignore[attr-defined]
_GEAR = _gear_table()

def cdc_fast_boundaries(data: bytes, min_size: int = 4096, avg_size: int = 8192,
                        max_size: int = 16384) -> List[Tuple[int, int]]:
    """Identify content defined cut boundaries via a rolling gear hash.

    This is a simplified FastCDC implementation.  It walks the input
    and emits a boundary when the rolling hash satisfies a bitmask
    condition provided the current segment length exceeds ``min_size``.
    If no boundary is found before ``max_size`` bytes, the segment is
    forcibly cut.  Returns a list of (start, end) pairs.
    """
    n = len(data)
    if n == 0:
        return []
    # choose mask bits based on average size (ensuring a target at most one
    # match on average)
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
        # search for a cutpoint between min_size and max_size
        while i < end_max:
            h = ((h << 1) & 0xFFFFFFFF) + _GEAR[data[i]]
            if (h & mask) == 0:
                i += 1
                break
            i += 1
        boundaries.append((start, i))
    return boundaries

###############################################################################
# Bijective BWT (BBWT) via Lyndon factorisation
###############################################################################

def duval_lyndon(s: bytes) -> List[Tuple[int, int]]:
    """Compute the Lyndon factorisation of a byte string.

    Returns a list of (start, end) indices where the substrings form
    non‑increasing Lyndon words.  Based on Duval's linear time
    algorithm.
    """
    n = len(s)
    i = 0
    out: List[Tuple[int, int]] = []
    while i < n:
        j = i + 1
        k = i
        # find the end of the current Lyndon word
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        p = j - k
        # output the Lyndon word repeated as many times as it fits
        while i <= k:
            out.append((i, i + p))
            i += p
    return out

def bbwt_forward(s: bytes) -> bytes:
    """Compute the bijective Burrows–Wheeler transform of a string.

    This implementation uses Lyndon factorisation to split the string
    into factors, then for each factor obtains the order of cyclic
    rotations via a simple prefix‑doubling suffix array on the doubled
    factor.  A k‑way merge under the ω‑order merges the rotations.
    See Bannai et al., "Linear time Lyndon factorization and runs
    theorem" for the underlying theory.  This is a simplified
    demonstrative version and is not optimised for extremely long
    factors.  For production use one should substitute a faster SA
    implementation.
    """
    if not s:
        return b""
    facs = duval_lyndon(s)
    # prepare rotation orders per factor
    def sa_prefix_doubling(t: bytes) -> List[int]:
        n = len(t)
        k = 1
        # initial ranks are byte values
        rank = list(t)
        tmp = [0] * n
        idx = list(range(n))
        while True:
            # sort by 2k length tuples
            idx.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
            tmp[idx[0]] = 0
            for j in range(1, n):
                a, b = idx[j - 1], idx[j]
                tmp[b] = tmp[a] + (
                    (rank[a], rank[a + k] if a + k < n else -1)
                    <
                    (rank[b], rank[b + k] if b + k < n else -1)
                )
            rank, tmp = tmp, rank
            if rank[idx[-1]] == n - 1:
                break
            k <<= 1
        return idx
    factors: List[Tuple[bytes, List[int]]] = []
    for a, b in facs:
        w = s[a:b]
        m = len(w)
        if m == 0:
            continue
        # build suffix array on doubled word
        ww = w + w
        sa = sa_prefix_doubling(ww)
        rot_order = [p for p in sa if p < m]
        factors.append((w, rot_order))
    # define rotation comparator for k‑way merge
    def rot_cmp(u: bytes, i: int, v: bytes, j: int) -> int:
        m, n = len(u), len(v)
        # compare u[i:] + u[:i] vs v[j:] + v[:j]
        p = 0
        while p < m + n:
            cu = u[(i + p) % m]
            cv = v[(j + p) % n]
            if cu != cv:
                return -1 if cu < cv else 1
            p += 1
        return 0
    # k‑way merge using a heap
    import heapq
    class Node:
        __slots__ = ("fi", "k", "w", "order")
        def __init__(self, fi: int, k: int, w: bytes, order: List[int]):
            self.fi = fi
            self.k = k
            self.w = w
            self.order = order
        def __lt__(self, other: "Node") -> bool:
            i = self.order[self.k]
            j = other.order[other.k]
            c = rot_cmp(self.w, i, other.w, j)
            if c != 0:
                return c < 0
            # deterministic tie break: by factor index then rotation index
            if self.fi != other.fi:
                return self.fi < other.fi
            return i < j
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
        # append the last character of this rotation
        out.append(w[(i - 1) % m])
        nd.k += 1
        if nd.k < len(nd.order):
            heapq.heappush(heap, nd)
    return bytes(out)

def bbwt_inverse(L: bytes) -> bytes:
    """Inverse of the bijective Burrows–Wheeler transform.

    Given the sequence of last characters ``L`` produced by ``bbwt_forward``,
    reconstruct the original string.  This implementation follows the
    cycle‑decomposition approach: the ``next`` array is the permutation
    mapping each position to its successor in the rotation, and cycles
    are collected to form factors.  The factors are concatenated in
    reverse of the lexicographically sorted order of their minima.
    """
    n = len(L)
    if n == 0:
        return b""
    # stable sort positions by (symbol, index)
    order = sorted(range(n), key=lambda idx: (L[idx], idx))
    pi = order[:]  # next‑index permutation
    # gather disjoint cycles
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
    # sort cycles by the minimum index to obtain canonical order
    cycles.sort(key=lambda cyc: min(cyc))
    factors: List[bytes] = []
    for cyc in cycles:
        # reconstruct each factor via following the cycle one character back
        i0 = min(cyc)
        d = len(cyc)
        cur = i0
        seq: List[int] = []
        for _ in range(d):
            cur = pi[cur]
            seq.append(L[cur])
        factors.append(bytes(seq))
    # concatenated in reverse order
    return b"".join(reversed(factors))

###############################################################################
# Move‑to‑front (MTF) coding
###############################################################################

def mtf_encode(data: bytes) -> List[int]:
    """Encode ``data`` into move‑to‑front positions.

    The list of 256 symbols is initialised to [0,1,2,...,255].  For each
    byte, the index of that byte in the list is emitted and the byte is
    moved to the front.  Output values range from 0..255.
    """
    table = list(range(256))
    out: List[int] = []
    for b in data:
        i = table.index(b)
        out.append(i)
        # move to front
        table.pop(i)
        table.insert(0, b)
    return out

def mtf_decode(seq: List[int]) -> bytes:
    """Inverse of ``mtf_encode``.

    Given a list of positions, reconstruct the original byte sequence.
    """
    table = list(range(256))
    out = bytearray()
    for pos in seq:
        b = table[pos]
        out.append(b)
        # move to front
        table.pop(pos)
        table.insert(0, b)
    return bytes(out)

###############################################################################
# Simple Rice coder for non‑negative integers
###############################################################################

class BitWriter:
    """Bitstream writer supporting unary, gamma and Rice encoding."""
    __slots__ = ("buf", "acc", "nbits")
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0
    def push_bit(self, bit: int) -> None:
        self.acc = (self.acc << 1) | (1 if bit else 0)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0
    def push_bits(self, val: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            self.push_bit((val >> i) & 1)
    def push_gamma(self, x: int) -> None:
        assert x >= 1
        b = x.bit_length()
        # unary prefix: b-1 zeros
        for _ in range(b - 1):
            self.push_bit(0)
        # binary suffix of length b
        self.push_bits(x, b)
    def push_rice(self, x: int, k: int) -> None:
        assert x >= 0
        q = x >> k
        # write q ones then a zero
        for _ in range(q):
            self.push_bit(1)
        self.push_bit(0)
        # remainder bits
        if k:
            self.push_bits(x & ((1 << k) - 1), k)
    def getbytes(self) -> bytes:
        # flush remaining bits
        if self.nbits:
            self.acc <<= (8 - self.nbits)
            self.buf.append(self.acc & 0xFF)
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf)

class BitReader:
    """Bitstream reader supporting unary, gamma and Rice decoding."""
    __slots__ = ("data", "idx", "acc", "nbits")
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.idx = 0
        self.acc = 0
        self.nbits = 0
    def _fill(self) -> None:
        if self.idx < len(self.data):
            self.acc = (self.acc << 8) | self.data[self.idx]
            self.idx += 1
            self.nbits += 8
    def read_bit(self) -> int:
        if self.nbits == 0:
            self._fill()
            if self.nbits == 0:
                raise EOFError("Bit underflow")
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1
    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v
    def read_gamma(self) -> int:
        z = 0
        while True:
            b = self.read_bit()
            if b == 0:
                z += 1
            else:
                break
        val = 1
        if z:
            val = (1 << z) | self.read_bits(z)
        return val
    def read_rice(self, k: int) -> int:
        q = 0
        while self.read_bit() == 1:
            q += 1
        r = self.read_bits(k) if k else 0
        return (q << k) | r

def cost_gamma(vals: List[int]) -> int:
    """Estimate bit cost of Elias‑γ encoding for positive integers."""
    c = 0
    for x in vals:
        v = max(1, x)
        b = v.bit_length()
        c += (b - 1) + b
    return c

def cost_rice(vals: List[int], k: int) -> int:
    """Estimate bit cost of Rice(k) encoding for non‑negative integers."""
    c = 0
    for x in vals:
        q = x >> k
        c += q + 1 + k
    return c

def choose_rice_grid(vals: List[int], kmax: int = 6) -> Tuple[int, int]:
    """Pick the Rice parameter yielding the fewest bits on a grid of k in [0..kmax].

    Returns the selected ``k`` and the corresponding bit cost.
    """
    if not vals:
        return 0, 0
    best_k = 0
    best_c = cost_rice(vals, 0)
    for k in range(1, kmax + 1):
        c = cost_rice(vals, k)
        if c < best_c:
            best_c, best_k = c, k
    return best_k, best_c

###############################################################################
# Model encoders
###############################################################################

def encode_model_raw(block: bytes) -> Tuple[bytes, Dict[str, object]]:
    """Model 0: no compression.

    Returns the payload (identical to the original block) and an empty
    metadata dict.  The MDL cost for this model is simply the length
    of the payload.  This acts as the fallback when other models do
    not produce a more compact representation.
    """
    return block, {}

def encode_model_xor(block: bytes) -> Tuple[bytes, Dict[str, object]]:
    """Model 1: XOR predictor with ULEB128 coded residuals.

    Each byte is XORed with the previous byte (starting from zero) to
    produce a residual in the range 0..255.  The sequence of
    residuals is encoded as a stream of unsigned LEB128 integers.  The
    resulting payload is typically shorter than the raw block when the
    input exhibits small changes between successive bytes.  Metadata
    contains no extra parameters.
    """
    # compute residuals
    residuals: List[int] = []
    prev = 0
    for b in block:
        residuals.append(b ^ prev)
        prev = b
    # encode residuals via ULEB128
    out = bytearray()
    for r in residuals:
        out += uleb128_encode(r)
    return bytes(out), {}

def encode_model_lz77(block: bytes) -> Tuple[bytes, Dict[str, object]]:
    """Model 3: naive LZ77 encoding with ULEB128 coded (length,distance).

    A simple sliding window dictionary coder.  Matches of length at
    least 3 bytes are encoded as a flag byte 1 followed by LEB128
    encoded length and distance (distance >= 1).  Literal bytes are
    encoded as flag byte 0 followed by the literal.  The payload is
    the concatenated stream of flags and encoded data.  This model
    works well on medium length repetitions.  Metadata contains no
    additional parameters.
    """
    i = 0
    n = len(block)
    out = bytearray()
    # window and lookahead sizes tuned for small Python prototype
    window_size = 255  # maximum backward distance stored in one byte ULEB128
    lookahead = 127    # limit match length to avoid long searches
    while i < n:
        best_len = 0
        best_dist = 0
        # define search window
        win_start = max(0, i - window_size)
        # search for the longest match in the window
        # using naive scanning suffices for a prototype
        for dist in range(1, i - win_start + 1):
            # starting index of the candidate match in the window
            j = i - dist
            # compute match length
            length = 0
            while (length < lookahead and
                   i + length < n and
                   block[j + length] == block[i + length]):
                length += 1
            if length >= 3 and length > best_len:
                best_len = length
                best_dist = dist
                # early exit if we found the maximum possible
                if best_len == lookahead:
                    break
        if best_len >= 3:
            # emit match (flag=1, length, distance)
            out.append(1)
            out += uleb128_encode(best_len)
            out += uleb128_encode(best_dist)
            i += best_len
        else:
            # emit literal (flag=0, byte)
            out.append(0)
            out.append(block[i])
            i += 1
    return bytes(out), {}

def encode_model_bbwt_mtf(block: bytes) -> Tuple[bytes, Dict[str, object]]:
    """Model 2: bijective BWT followed by MTF and run coding using Rice/Gamma.

    This model applies the BBWT forward transform to the block to
    amplify runs of identical symbols.  The BBWT output is then
    move‑to‑front encoded.  Runs of zeros and the non‑zero values
    (offset by one) are encoded via Rice coding with small grid search
    for the Rice parameter, falling back to Elias‑γ when beneficial.
    The returned payload is a bitstream represented as bytes.  The
    metadata records whether Rice was used for zeros and non‑zeros
    respectively along with the selected k parameters.  The decoder
    needs the original block length to reconstruct the MTF sequence.
    """
    # compute BBWT of the block
    L = bbwt_forward(block)
    # MTF encode
    seq = mtf_encode(L)
    # separate zero runs and non‑zero values (subtract 1)
    zero_runs: List[int] = []
    nonzeros: List[int] = []
    tags: List[Tuple[int, int]] = []  # (tag, value)
    i = 0
    n = len(seq)
    while i < n:
        if seq[i] == 0:
            j = i + 1
            while j < n and seq[j] == 0:
                j += 1
            run = j - i
            zero_runs.append(run)
            tags.append((0, run))
            i = j
        else:
            val = seq[i]
            nonzeros.append(val - 1)
            tags.append((1, val - 1))
            i += 1
    # choose Rice parameters
    k0, c0 = choose_rice_grid(zero_runs)
    k1, c1 = choose_rice_grid(nonzeros)
    # estimate gamma costs to decide between Rice and Gamma
    use_rice_zero = (c0 + 3) < (cost_gamma(zero_runs) + 3)
    use_rice_nz = (c1 + 3) < (cost_gamma([v + 1 for v in nonzeros]) + 3)
    bw = BitWriter()
    # flags: bit0 = use_rice_zero, bit1 = use_rice_nz
    flags = (1 if use_rice_zero else 0) | ((1 if use_rice_nz else 0) << 1)
    bw.push_bits(flags, 2)
    # store k0 and k1 in 4 bits each (0..15)
    bw.push_bits(k0 & 0xF, 4)
    bw.push_bits(k1 & 0xF, 4)
    # encode the sequence of tags
    for tag, val in tags:
        bw.push_bit(tag)
        if tag == 0:
            # zero run
            if use_rice_zero:
                bw.push_rice(val, k0)
            else:
                bw.push_gamma(val)
        else:
            # non‑zero
            if use_rice_nz:
                bw.push_rice(val, k1)
            else:
                bw.push_gamma(val + 1)
    payload = bw.getbytes()
    meta = {
        "k0": k0,
        "k1": k1,
        "use_rice_zero": use_rice_zero,
        "use_rice_nz": use_rice_nz,
    }
    return payload, meta

###############################################################################
# Decoders for each model
###############################################################################

def decode_model_raw(payload: bytes, orig_len: int) -> bytes:
    """Decode RAW model: payload is original data."""
    assert len(payload) == orig_len, "Payload length mismatch for RAW"
    return payload

def decode_model_xor(payload: bytes, orig_len: int) -> bytes:
    """Decode XOR model.

    Reads ULEB128‑encoded residuals until ``orig_len`` values have been
    recovered, then reconstructs the original bytes by XORing with the
    running previous byte value.
    """
    residuals: List[int] = []
    pos = 0
    while len(residuals) < orig_len:
        r, pos = uleb128_decode_stream(payload, pos)
        residuals.append(r)
    # reconstruct original bytes
    out = bytearray()
    prev = 0
    for r in residuals:
        b = r ^ prev
        out.append(b)
        prev = b
    return bytes(out)

def decode_model_lz77(payload: bytes, orig_len: int) -> bytes:
    """Decode naive LZ77 model.

    Reads a stream of flag bytes followed by either a literal byte or a
    (length,distance) pair encoded as ULEB128.  Reconstruction stops
    once ``orig_len`` bytes have been produced.  Extra input beyond
    that point (if present) is ignored.
    """
    i = 0
    out = bytearray()
    while i < len(payload) and len(out) < orig_len:
        flag = payload[i]
        i += 1
        if flag == 0:
            # literal
            if i >= len(payload):
                raise EOFError("Truncated LZ77 literal")
            out.append(payload[i])
            i += 1
        elif flag == 1:
            # match
            length, i = uleb128_decode_stream(payload, i)
            dist, i = uleb128_decode_stream(payload, i)
            # copy bytes from distance back
            for _ in range(length):
                if len(out) >= orig_len:
                    break
                # ensure distance is valid
                if dist > len(out):
                    raise ValueError("Invalid LZ77 distance")
                out.append(out[-dist])
        else:
            raise ValueError(f"Invalid LZ77 flag: {flag}")
    # ensure we produced the expected length
    if len(out) != orig_len:
        # Incomplete decode indicates malformed payload
        raise ValueError(f"LZ77 decode length mismatch: expected {orig_len}, got {len(out)}")
    return bytes(out)

def decode_model_bbwt_mtf(payload: bytes, orig_len: int) -> bytes:
    """Decode BBWT→MTF model.

    Interprets ``payload`` as a bitstream containing flags and Rice/Gamma
    coded run values.  The original MTF sequence length is provided by
    ``orig_len`` and guides how many symbols to decode.  The MTF
    sequence is then inversed to recover the BBWT output, which is
    inverted via ``bbwt_inverse``.  Returns the reconstructed block.
    """
    br = BitReader(payload)
    # read flags and k parameters
    flags = br.read_bits(2)
    k0 = br.read_bits(4)
    k1 = br.read_bits(4)
    use_rice_zero = (flags & 1) != 0
    use_rice_nz = (flags >> 1) != 0
    seq: List[int] = []
    # decode until orig_len elements are produced
    while len(seq) < orig_len:
        tag = br.read_bit()
        if tag == 0:
            if use_rice_zero:
                run = br.read_rice(k0)
            else:
                run = br.read_gamma()
            seq.extend([0] * run)
        else:
            if use_rice_nz:
                val = br.read_rice(k1)
            else:
                val = br.read_gamma() - 1
            seq.append(val + 1)
    seq = seq[:orig_len]
    # inverse MTF
    L = mtf_decode(seq)
    # inverse BBWT
    return bbwt_inverse(L)

###############################################################################
# MDL selection and container encode/decode
###############################################################################

ModelEncoder = Callable[[bytes], Tuple[bytes, Dict[str, object]]]
ModelDecoder = Callable[[bytes, int], bytes]

# Map method IDs to encoders and decoders
_ENCODERS: Dict[int, ModelEncoder] = {
    0: encode_model_raw,
    1: encode_model_xor,
    2: encode_model_bbwt_mtf,
    3: encode_model_lz77,
}
_DECODERS: Dict[int, ModelDecoder] = {
    0: decode_model_raw,
    1: decode_model_xor,
    2: decode_model_bbwt_mtf,
    3: decode_model_lz77,
}

def _encode_block(block: bytes) -> Tuple[int, bytes, int]:
    """Encode a single block using MDL selection.

    For the given ``block``, each registered model encoder is invoked in
    increasing order of model ID.  A very cheap entropy guard first
    measures the sampled Shannon entropy; if it exceeds 7.8 bits/byte
    (nearly uniform), heavy models beyond RAW are skipped to avoid
    wasting time.  The total code length (payload length) is
    measured for each candidate.  The model with the smallest payload
    length is selected; in case of a tie the lowest model ID wins.
    Returns a tuple of ``(method_id, payload, payload_length)``.
    """
    # cheap guard: high entropy suggests raw may be best
    H = sample_shannon_entropy(block, sample_rate=64)
    candidate_ids = list(_ENCODERS.keys())
    # if very high entropy, only try RAW (0) and XOR (1) since others
    # are unlikely to help.  XOR is extremely cheap; BBWT and LZ77 are
    # skipped to save time.
    if H > 7.8:
        candidate_ids = [0, 1]
    best_id: Optional[int] = None
    best_payload: Optional[bytes] = None
    best_len: int = 2**31 - 1
    for mid in candidate_ids:
        encoder = _ENCODERS[mid]
        try:
            payload, meta = encoder(block)
        except Exception:
            # skip models that raise
            continue
        # payload length in bytes; we do not account for descriptor
        # overhead here because it is equal across models when compared
        # within a block (all models require a one‑byte method id and
        # u32 lengths which we add later).  A real MDL scheme would
        # include model description bits too.
        plen = len(payload)
        if plen < best_len or (plen == best_len and (best_id is None or mid < best_id)):
            best_id = mid
            best_payload = payload
            best_len = plen
    # fallback if all encoders failed
    if best_id is None or best_payload is None:
        best_id, best_payload, best_len = 0, block, len(block)
    return best_id, best_payload, best_len

def compress(data: bytes, target_block: int = 8192) -> bytes:
    """Compress ``data`` using content defined chunking and MDL selection.

    The input is first partitioned into blocks via a FastCDC‑like
    routine with the provided ``target_block`` as the average size.
    Each block is compressed by choosing the shortest description
    among the registered models.  The compressed stream is prefaced
    with a container header and per‑block metadata.  Returns the
    binary container.
    """
    # determine cutpoints
    cuts = cdc_fast_boundaries(data, min_size=target_block // 2,
                               avg_size=target_block,
                               max_size=target_block * 2)
    blocks = [data[a:b] for a, b in cuts]
    out = bytearray()
    # write header
    out += b'KOLM'
    # record nominal block size used (u32)
    out += struct.pack('<I', target_block & 0xFFFFFFFF)
    # total input length (u64)
    out += struct.pack('<Q', len(data))
    # number of blocks (u16, clamp to max)
    nblocks = len(blocks)
    out += struct.pack('<H', nblocks & 0xFFFF)
    # encode each block
    for block in blocks:
        orig_len = len(block)
        method_id, payload, plen = _encode_block(block)
        # write method id
        out.append(method_id & 0xFF)
        # write orig_len and payload length as little endian u32
        out += struct.pack('<I', orig_len & 0xFFFFFFFF)
        out += struct.pack('<I', plen & 0xFFFFFFFF)
        # append payload
        out += payload
    return bytes(out)

def decompress(blob: bytes) -> bytes:
    """Decompress a container produced by ``compress``.

    Parses the container header to recover the number of blocks and
    iterates over them, invoking the appropriate decoder based on the
    stored ``method_id``.  Each block's original length guides the
    decoder when needed (e.g. for BBWT→MTF).  The reconstructed
    blocks are concatenated to yield the original data.  Raises
    ``ValueError`` on malformed input.
    """
    p = 0
    if blob[p:p + 4] != b'KOLM':
        raise ValueError("Bad magic header")
    p += 4
    target_block = struct.unpack_from('<I', blob, p)[0]
    p += 4
    total_len = struct.unpack_from('<Q', blob, p)[0]
    p += 8
    nblocks = struct.unpack_from('<H', blob, p)[0]
    p += 2
    out = bytearray()
    for _ in range(nblocks):
        if p >= len(blob):
            raise EOFError("Truncated block header")
        method_id = blob[p]
        p += 1
        if method_id not in _DECODERS:
            raise ValueError(f"Unknown method id {method_id}")
        # read lengths
        if p + 8 > len(blob):
            raise EOFError("Truncated block lengths")
        orig_len = struct.unpack_from('<I', blob, p)[0]
        p += 4
        payload_len = struct.unpack_from('<I', blob, p)[0]
        p += 4
        if p + payload_len > len(blob):
            raise EOFError("Truncated payload")
        payload = blob[p:p + payload_len]
        p += payload_len
        # decode
        decoder = _DECODERS[method_id]
        block = decoder(payload, orig_len)
        # sanity check length
        if len(block) != orig_len:
            raise ValueError(
                f"Decoded length mismatch: expected {orig_len}, got {len(block)}"
            )
        out += block
    # optional: verify total_len matches
    if len(out) != total_len:
        raise ValueError(
            f"Total decoded length mismatch: expected {total_len}, got {len(out)}"
        )
    return bytes(out)

###############################################################################
# Demo / CLI
###############################################################################

if __name__ == '__main__':
    import argparse, sys, os
    parser = argparse.ArgumentParser(description="Kolmogorov/bit‑circuit compressor prototype")
    parser.add_argument('input', help="Input file to compress or decompress")
    parser.add_argument('-d', '--decompress', action='store_true', help="Decompress instead of compress")
    parser.add_argument('-o', '--output', help="Output file")
    parser.add_argument('-b', '--block', type=int, default=8192, help="Target block size for compression (default 8192)")
    args = parser.parse_args()
    if args.decompress:
        data = open(args.input, 'rb').read()
        out = decompress(data)
        outname = args.output or (os.path.splitext(args.input)[0] + '.out')
        open(outname, 'wb').write(out)
        print(f"Decompressed {len(data)} bytes to {len(out)} bytes → {outname}")
    else:
        data = open(args.input, 'rb').read()
        blob = compress(data, target_block=args.block)
        outname = args.output or (args.input + '.kolm')
        open(outname, 'wb').write(blob)
        # compute ratio
        ratio = len(blob) / len(data) if len(data) else 1.0
        print(f"Compressed {len(data)} bytes to {len(blob)} bytes (ratio {ratio:.3f}) → {outname}")
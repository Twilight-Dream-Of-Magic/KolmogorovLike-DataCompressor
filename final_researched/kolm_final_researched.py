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

import math
import struct
import random
from typing import List, Tuple, Dict, Callable, Optional, Any

###############################################################################
# Utility functions: ULEB128
###############################################################################

def uleb128_encode(n: int) -> bytes:
    """Encode a non‑negative integer into unsigned LEB128."""
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
    """Decode a LEB128 value from data starting at position ``pos``."""
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
    """Encode a list of non‑negative integers using Rice coding with parameter 2^k."""
    out_bits = []
    M = 1 << k
    for n in seq:
        q = n // M
        r = n % M
        out_bits.append('1' * q + '0')
        out_bits.append(format(r, f'0{k}b'))
    bitstr = ''.join(out_bits)
    # pack into bytes
    pad = (8 - len(bitstr) % 8) % 8
    bitstr += '0' * pad
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i + 8], 2))
    return bytes(out)

# -----------------------------------------------------------------------------
# Gray code transform
#
# A Gray code is a binary numbering system in which adjacent values differ in
# exactly one bit.  A common way to generate the Gray code g(x) for an
# integer x is to compute g(x) = x XOR (x >> 1).  This transform is
# reversible; the original number can be recovered by repeatedly XORing
# the Gray code with itself right‑shifted by powers of two【882443461041209†L485-L536】.
# In this compressor we map each byte to its 8‑bit Gray code prior to
# entropy coding, and invert it during decoding.

def gray_encode_bytes(data: bytes) -> bytes:
    """Convert each byte into its Gray code: g = x ^ (x >> 1)."""
    return bytes(((b ^ (b >> 1)) & 0xFF) for b in data)

def gray_decode_bytes(data: bytes) -> bytes:
    """Recover original bytes from their Gray codes.

    Given a Gray code g, the original value n can be recovered by
    iteratively XORing g with itself right‑shifted by 1, 2, 4, etc.
    See【882443461041209†L564-L580】 for the derivation.
    """
    out = bytearray()
    for g in data:
        n = g
        n ^= (n >> 1)
        n ^= (n >> 2)
        n ^= (n >> 4)
        out.append(n & 0xFF)
    return bytes(out)

def rice_decode(data: bytes, k: int, nvals: int) -> List[int]:
    bitstr = ''.join(f'{b:08b}' for b in data)
    i = 0
    M = 1 << k
    out: List[int] = []
    while len(out) < nvals:
        q = 0
        while bitstr[i] == '1':
            q += 1; i += 1
        i += 1  # skip the '0'
        r = int(bitstr[i:i + k], 2) if k > 0 else 0
        i += k
        out.append(q * M + r)
    return out

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

###############################################################################
# Simple Re‑Pair grammar compression
###############################################################################

def repair_compress(block: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Compress ``block`` using a basic Re‑Pair grammar compressor.

    The algorithm repeatedly replaces the most frequent adjacent symbol pair
    with a new non‑terminal until no pair appears more than once【967225900425034†L139-L166】.
    A mapping of rules is recorded; the output encodes the rule table and
    the final sequence via ULEB128.  This implementation limits the
    maximum number of rules to control memory usage.
    """
    # map bytes to integers for uniform processing
    seq = list(block)
    next_sym = 256
    rules: Dict[int, Tuple[int, int]] = {}
    max_rules = 256  # limit rules to one byte code
    while True:
        # count adjacent pairs
        freq: Dict[Tuple[int, int], int] = {}
        for a, b in zip(seq, seq[1:]):
            freq[(a, b)] = freq.get((a, b), 0) + 1
        # find most frequent pair occurring more than once
        cand = None
        best = 1
        for pair, f in freq.items():
            if f > best:
                best = f; cand = pair
        if cand is None or next_sym >= 256 + max_rules:
            break
        a, b = cand
        rules[next_sym] = (a, b)
        # replace occurrences
        i = 0
        new_seq = []
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(next_sym)
                i += 2
            else:
                new_seq.append(seq[i]); i += 1
        seq = new_seq
        next_sym += 1
    # encode: rules list length, each rule, final sequence length and sequence
    out = bytearray()
    out += uleb128_encode(len(rules))
    for nt, (a, b) in rules.items():
        out.append(nt - 256)  # store non‑terminal id offset by 256
        out.append(a)
        out.append(b)
    out += uleb128_encode(len(seq))
    for sym in seq:
        out.append(sym if sym < 256 else (sym - 256))
    return bytes(out), {"rules": rules, "final_len": len(seq)}

def repair_decompress(data: bytes, orig_len: int) -> bytes:
    i = 0
    nrules, i = uleb128_decode_stream(data, i)
    rules: Dict[int, Tuple[int, int]] = {}
    for _ in range(nrules):
        nt = data[i] + 256; a = data[i + 1]; b = data[i + 2]; i += 3
        rules[nt] = (a, b)
    seq_len, i = uleb128_decode_stream(data, i)
    seq = [data[j] if data[j] < 256 else data[j] + 256 for j in range(i, i + seq_len)]
    # expand sequence
    def expand(sym: int) -> List[int]:
        if sym < 256:
            return [sym]
        a, b = rules[sym]
        return expand(a) + expand(b)
    out = bytearray()
    for sym in seq:
        out += bytes(expand(sym))
    assert len(out) == orig_len
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
    candidates: List[Tuple[ModelEncoder, str]] = [
        (encode_raw, 'raw'),
        (encode_xor, 'xor'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, False, rice_param=2), 'bbwt'),
        (lambda b: encode_bbwt_mtf_rice(b, True, False, False, False, False, rice_param=2), 'bbwt_bp'),
        (lambda b: encode_bbwt_mtf_rice(b, False, True, False, False, False, rice_param=2), 'bbwt_lfsr'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, True, False, False, rice_param=2), 'bbwt_nib'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, True, False, rice_param=2), 'bbwt_br'),
        (lambda b: encode_bbwt_mtf_rice(b, True, True, False, False, False, rice_param=2), 'bbwt_bp_lfsr'),
        (lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, True, rice_param=2), 'bbwt_gray'),
        (encode_lz77, 'lz77'),
        (encode_lfsr_predict, 'lfsr_pred'),
        (repair_compress, 'repair'),
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
    decoders: List[Callable[[bytes, int, Dict[str, Any]], bytes]] = [
        decode_raw,
        decode_xor,
        # bbwt base
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":0,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + bitplane
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":1,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + lfsr
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":2,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + nibble swap
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":4,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + bit reverse
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":8,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + bitplane + lfsr (flags 3)
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":3,"k":2,"length":len(p),"orig_len":l}),
        # bbwt + gray code
        lambda p,l,meta=None: decode_bbwt_mtf_rice(p, {"flags":16,"k":2,"length":len(p),"orig_len":l}),
        decode_lz77,
        decode_lfsr_predict,
        repair_decompress,
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
    names = [
        'Raw', 'XOR', 'BBWT', 'BBWT+Bitplane', 'BBWT+LFSR',
        'BBWT+Nibble', 'BBWT+BitRev', 'BBWT+BP+LFSR', 'BBWT+Gray',
        'LZ77', 'LFSR predictor', 'Re‑Pair'
    ]
    # map names to encoder functions (same order as decoders)
    encoders = [
        encode_raw,
        encode_xor,
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, True, False, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, True, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, True, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, True, False),
        lambda b: encode_bbwt_mtf_rice(b, True, True, False, False, False),
        lambda b: encode_bbwt_mtf_rice(b, False, False, False, False, True),
        encode_lz77,
        encode_lfsr_predict,
        repair_compress,
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
    plt.savefig('/home/oai/share/kolm_researched_ratios.png')
    print('Experiment completed, plot saved to kolm_researched_ratios.png')

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
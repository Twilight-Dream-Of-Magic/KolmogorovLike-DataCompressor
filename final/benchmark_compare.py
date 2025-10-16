#!/usr/bin/env python3
"""
benchmark_compare.py -- Compare the custom Kolmogorov/MDL compressor against
baseline algorithms implemented in pure Python.

This utility script exercises several compressors on a small suite of
hand-crafted test data sets. Each compressor is invoked to compress
and then decompress the data. The script measures the total bytes
produced (compression ratio), as well as the time taken to encode and
decode. Results are collected into a pandas DataFrame and plotted
using matplotlib.

The custom compressor is imported from ``kolm_final``; it uses a
combination of BBWT, XOR and LZ77 models with an MDL-based chooser.
Two baselines are provided here:

* ``baseline_bbwt_mtf_rle`` – A very simple block transform coder that
 applies the bijective Burrows–Wheeler transform (via the ``bbwt_forward``
 routine from ``kolm_final``) followed by Move-to-Front (MTF) coding
 and run-length encoding. Runs of zeros and individual non-zero
 values are stored using unsigned LEB128 integers. Decoding is the
 obvious inverse: parse the run-length stream, reconstruct the MTF
 sequence, invert BBWT.

* ``baseline_lz77`` – A naive LZ77 encoder/decoder pair imported
directly from ``kolm_final``. This implementation scans a sliding
window for matches and encodes length/distance pairs with ULEB128.
It serves as a baseline because it mirrors the basic idea of
dictionary coding without relying on any external C libraries.

The intention of these baselines is to provide a fair comparison
against other known compression paradigms (transform coding and
dictionary coding) without invoking compiled libraries like zlib.

Run this script directly to print a table of metrics and output a
PNG chart named ``kolm_comparison_plot.png`` into the working directory.
"""

import os
import time
import struct
import math
from typing import Dict, Tuple, List

import matplotlib
matplotlib.use('Agg') # headless backend
import matplotlib.pyplot as plt
import pandas as pd

import kolm_final # custom compressor module

# Short aliases for functions used from kolm_final
from kolm_final import (
    compress as kolm_compress,
    decompress as kolm_decompress,
    bbwt_forward,
    bbwt_inverse,
    mtf_encode,
    mtf_decode,
    uleb128_encode,
    uleb128_decode_stream,
    encode_model_lz77,
    decode_model_lz77,
)

def baseline_bbwt_mtf_rle_encode(block: bytes) -> bytes:
    """Encode a block using BBWT → MTF → run-length coding.

    The bijective BWT (BBWT) is applied to the block, resulting in a
    permuted block of equal length. The BBWT output is then MTF
    encoded to produce a sequence of small integers. This sequence
    contains many zeros for repeated symbols. We emit a stream of
    (tag, value) pairs: if tag=0 the value is a run of zeros; if
    tag=1 the value is a non-zero symbol minus one. Both values are
    stored as ULEB128. The decoder will need to know the original
    block length (passed in separately) to reconstruct the full MTF
    sequence.
    """
    # transform via BBWT
    L = bbwt_forward(block)
    # MTF encode
    seq = mtf_encode(L)
    out = bytearray()
    run = 0
    for v in seq:
        if v == 0:
            run += 1
        else:
            # flush any pending zero run
            if run > 0:
                out.append(0) # zero-run tag
                out += uleb128_encode(run)
                run = 0
            # emit non-zero tag and value-1
            out.append(1)
            out += uleb128_encode(v - 1)
    # flush any remaining zero run
    if run > 0:
        out.append(0)
        out += uleb128_encode(run)
    return bytes(out)


def baseline_bbwt_mtf_rle_decode(payload: bytes, orig_len: int) -> bytes:
    """Decode a payload encoded by ``baseline_bbwt_mtf_rle_encode``.

    The decoder reads a stream of (tag, value) pairs: tag=0 denotes a
    run of zeros of length ``value``; tag=1 denotes a single MTF
    symbol equal to ``value+1``. The resulting sequence must have
    length equal to ``orig_len``. It is then MTF decoded to recover
    the BBWT output, and finally inverted via ``bbwt_inverse`` to
    obtain the original block.
    """

    # parse payload into MTF sequence
    seq: List[int] = []
    i = 0
    n = len(payload)
    while i < n:
        tag = payload[i]
        i += 1
        value, i = uleb128_decode_stream(payload, i)
        if tag == 0:
            seq.extend([0] * value)
        elif tag == 1:
            seq.append(value + 1)
        else:
            raise ValueError(f"unknown tag {tag} in baseline decode")
    # pad/truncate sequence to original length
    if len(seq) != orig_len:
        # If the decoded sequence length differs, raise an error; this
        # should not happen if encoding/decoding are consistent.
        raise ValueError(
            f"baseline decode produced {len(seq)} symbols, expected {orig_len}")
    # reconstruct BBWT output
    mtf_bytes = bytes(seq)
    L = mtf_decode(mtf_bytes)
    # invert BBWT to recover original block
    return bbwt_inverse(L)

def baseline_lz77_encode(block: bytes) -> bytes:
    """Encode a block using the naive LZ77 model from kolm_final.

    Returns only the payload (does not include model id or metadata).
    """
    payload, _meta = encode_model_lz77(block)
    return payload


def baseline_lz77_decode(payload: bytes, orig_len: int) -> bytes:
    """Decode a payload encoded by ``baseline_lz77_encode``.

    Calls the corresponding decoder in ``kolm_final``.
    """
    return decode_model_lz77(payload, orig_len)

def run_benchmarks():
    """Run compression benchmarks on a suite of test data sets.

    Returns a pandas DataFrame with results for each combination of
    dataset and compressor. Also writes a PNG plot to disk.
    """
    import random

    # Assemble test data sets
    data_sets: Dict[str, bytes] = {
        "repetitive_text": b"A" * 2000 + b"B" * 1000 + (b"CD" * 500),
        "english_like": (b"In compression we favor short programs and transparent circuits. " * 20),
        "source_code": open(__file__, 'rb').read()[:4096], # first 4 KiB of this script as code sample
        "byte_counter": bytes([i % 256 for i in range(4096)]),
        # use a deterministic random seed for reproducibility
        "random_bytes": bytes(random.Random(42).getrandbits(8) for _ in range(4096)),
    }
    results: List[Dict[str, object]] = []

    for name, data in data_sets.items():
        orig_len = len(data)
        # Custom compressor (kolm_final)
        t0 = time.perf_counter()
        cdata = kolm_compress(data)
        comp_time = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        ddata = kolm_decompress(cdata)
        decomp_time = (time.perf_counter() - t0) * 1000.0
        ratio = len(cdata) / orig_len
        ok = (ddata == data)
        results.append({
            'dataset': name,
            'algorithm': 'kolm_final',
            'ratio': ratio,
            'comp_ms': comp_time,
            'decomp_ms': decomp_time,
            'valid': ok,
        })

        # Baseline 1: BBWT→MTF→RLE
        t0 = time.perf_counter()
        payload = baseline_bbwt_mtf_rle_encode(data)
        comp_time = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        try:
            decoded = baseline_bbwt_mtf_rle_decode(payload, orig_len)
            ok_baseline = decoded == data
        except Exception:
            ok_baseline = False
        decomp_time = (time.perf_counter() - t0) * 1000.0
        ratio = len(payload) / orig_len
        results.append({
            'dataset': name,
            'algorithm': 'baseline_bbwt_mtf_rle',
            'ratio': ratio,
            'comp_ms': comp_time,
            'decomp_ms': decomp_time,
            'valid': ok_baseline,
        })
        # Baseline 2: naive LZ77
        t0 = time.perf_counter()
        payload2 = baseline_lz77_encode(data)
        comp_time2 = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        decoded2 = baseline_lz77_decode(payload2, orig_len)
        decomp_time2 = (time.perf_counter() - t0) * 1000.0
        ratio2 = len(payload2) / orig_len
        ok2 = decoded2 == data
        results.append({
            'dataset': name,
            'algorithm': 'baseline_lz77',
            'ratio': ratio2,
            'comp_ms': comp_time2,
            'decomp_ms': decomp_time2,
            'valid': ok2,
        })
    df = pd.DataFrame(results)
    # Create a bar chart comparing ratios and times
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    for ax, metric, title in zip(
        axs,
        ['ratio', 'comp_ms', 'decomp_ms'],
        ['Compression Ratio (lower is better)',
         'Compression Time (ms)',
         'Decompression Time (ms)']):
        subset = df.pivot(index='dataset', columns='algorithm', values=metric)
        subset.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plot_path = 'kolm_comparison_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(df)
    print(f"Plot written to {plot_path}")
    return df, plot_path


if __name__ == '__main__':
    run_benchmarks()

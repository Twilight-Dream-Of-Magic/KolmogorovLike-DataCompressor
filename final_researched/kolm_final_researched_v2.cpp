// kolm_final_researched_v2.cpp
// ===================================
//
// This translation of the original Python module implements a prototype
// compressor/decompressor that follows the “boolean‑circuit first, probability
// last” philosophy.  The code is written in ISO C++20 using only the
// standard library.  Each function corresponds closely to its Python
// counterpart with clear variable names and detailed comments explaining the
// algorithmic intent.  Care has been taken to avoid arithmetic overflow by
// consistently masking to fixed bit‑widths where appropriate.  All
// transformations are lossless and reversible; when paired with the
// appropriate decoder they reconstruct the original data exactly.

#include <cassert>
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
// Additional includes for timing and formatted output in run_self_test.  These
// headers provide std::chrono for wall‑clock measurements and std::setw,
// std::setprecision, etc. for nicely formatted tables.
#include <chrono>
#include <iomanip>

// -----------------------------------------------------------------------------
// PairHash helper
//
// The Re‑Pair compressor counts the frequency of adjacent symbol pairs using an
// unordered_map with keys of type std::pair<int,int>.  The standard library
// does not provide a noexcept default‑constructible hash for std::pair in all
// implementations (and even when it does, the resulting unordered_map may
// assert that the hash functor is not noexcept).  We define our own hash
// functor here that combines the two 32‑bit integers into a 64‑bit value and
// hashes that using std::hash<std::uint64_t>.  This ensures the hash type is
// properly constructible and noexcept.
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::uint64_t combined = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.first)) << 32)
                                 ^ static_cast<std::uint32_t>(p.second);
        return std::hash<std::uint64_t>{}(combined);
    }
};

// Forward declarations for functions defined later in the file.  The
// compressor uses a wide variety of helper routines and data structures
// defined throughout this file; listing prototypes up front makes the
// ordering of definitions less critical.

// ULEB128 coding
std::vector<std::uint8_t> uleb128_encode(std::uint64_t n);
std::pair<std::uint64_t, std::size_t>
uleb128_decode_stream(const std::vector<std::uint8_t>& data, std::size_t pos);

// Content defined chunking (FastCDC)
using BoundaryList = std::vector<std::pair<std::size_t, std::size_t>>;
BoundaryList cdc_fast_boundaries(const std::vector<std::uint8_t>& data,
                                 std::size_t min_size = 4096,
                                 std::size_t avg_size = 8192,
                                 std::size_t max_size = 16384);

// Bijective Burrows–Wheeler transform and Lyndon factorisation
using RangeList = std::vector<std::pair<std::size_t, std::size_t>>;
RangeList duval_lyndon(const std::vector<std::uint8_t>& s);
std::vector<std::uint8_t> bbwt_forward(const std::vector<std::uint8_t>& s);
std::vector<std::uint8_t> bbwt_inverse(const std::vector<std::uint8_t>& L);

// Move‑to‑front coding
std::vector<std::uint8_t> mtf_encode(const std::vector<std::uint8_t>& data);
std::vector<std::uint8_t> mtf_decode(const std::vector<std::uint8_t>& seq);

// Bitwise reversible circuit modules
std::vector<std::uint8_t> bitplane_interleave(const std::vector<std::uint8_t>& data);
std::vector<std::uint8_t> bitplane_deinterleave(const std::vector<std::uint8_t>& data,
                                                std::size_t orig_len);
std::vector<std::uint8_t> lfsr_whiten(const std::vector<std::uint8_t>& data,
                                      std::uint8_t taps = 0b10010110,
                                      std::uint8_t seed = 1);
std::vector<std::uint8_t> nibble_swap(const std::vector<std::uint8_t>& data);
std::vector<std::uint8_t> bit_reverse(const std::vector<std::uint8_t>& data);

// Boolean gate primitives and automata
inline std::uint8_t gate_and(std::uint8_t a, std::uint8_t b) { return a & b; }
inline std::uint8_t gate_or(std::uint8_t a, std::uint8_t b) { return a | b; }
std::uint8_t gate_not(std::uint8_t a, std::uint8_t width = 8);
std::uint8_t gate_xor(std::uint8_t a, std::uint8_t b, std::uint8_t width = 8);
std::uint8_t prefix_or_word(std::uint8_t x, std::uint8_t width);
std::uint8_t left_band(std::uint8_t beta, std::uint8_t L, std::uint8_t width);
std::uint8_t byte_eq(std::uint8_t a, std::uint8_t b);
std::uint8_t mux_byte(std::uint8_t m, std::uint8_t a,
                      std::uint8_t b, std::uint8_t width = 8);
std::uint8_t gray_pred(std::uint8_t v);

// Four automaton modes and selection
std::vector<std::uint8_t> mode1_forward(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> mode1_inverse(const std::vector<std::uint8_t>& residual);
std::vector<std::uint8_t> mode2_forward(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> mode2_inverse(const std::vector<std::uint8_t>& residual);
std::vector<std::uint8_t> mode3_forward(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> mode3_inverse(const std::vector<std::uint8_t>& residual);
std::vector<std::uint8_t> mode4_forward(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> mode4_inverse(const std::vector<std::uint8_t>& residual);

// Circuit automaton selector
struct AutomatonResult {
    std::vector<std::uint8_t> data;
    std::uint8_t mode;
};
AutomatonResult circuit_map_automaton_forward(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t>
circuit_map_automaton_inverse(const std::vector<std::uint8_t>& mapped,
                              std::uint8_t mode);

// Helper functions for bitplane and entropy calculation
double first_order_bit_entropy(const std::vector<std::uint8_t>& block);
std::tuple<std::vector<std::vector<int>>, std::size_t> bytes_to_bitplanes(
    const std::vector<std::uint8_t>& data);
std::vector<std::uint8_t> bitplanes_to_bytes(
    const std::vector<std::vector<int>>& planes);
double avg_run_bits(const std::vector<int>& bits);
double H0_bits(const std::vector<int>& bits);
std::pair<int, std::vector<int>> rle_binary(const std::vector<int>& bits);
std::vector<int> unrle_binary(int first_bit, const std::vector<int>& runs);
std::vector<std::uint8_t> pack_bits_to_bytes(const std::vector<int>& bits);
std::vector<int> unpack_bits_from_bytes(const std::vector<std::uint8_t>& buf,
                                        std::size_t nbits);

// Rice/Golomb coding for non‑negative integers
std::vector<std::uint8_t> rice_encode(const std::vector<std::uint64_t>& seq,
                                      std::uint8_t k);
std::vector<std::uint64_t> rice_decode(const std::vector<std::uint8_t>& data,
                                       std::uint8_t k,
                                       std::size_t nvals);

// Gray code helpers
std::vector<std::uint8_t> gray_encode_bytes(const std::vector<std::uint8_t>& data);
std::vector<std::uint8_t> gray_decode_bytes(const std::vector<std::uint8_t>& data);

// LZ77 and Re‑Pair coders
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_lz77(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> decode_lz77(const std::vector<std::uint8_t>& data,
                                      std::size_t orig_len);

std::unordered_map<std::pair<int,int>, int, PairHash>
count_pairs(const std::vector<int>& seq);

std::pair<std::vector<int>, int>
replace_non_overlapping(const std::vector<int>& seq, const std::pair<int,int>& target, int new_sym);
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
repair_compress(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> repair_decompress(const std::vector<std::uint8_t>& data,
                                            std::size_t orig_len);

// LFSR predictor
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_lfsr_predict(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> decode_lfsr_predict(const std::vector<std::uint8_t>& data,
                                              std::size_t orig_len);

// BBWT→MTF→Rice model with optional bitwise modules
struct BBWTMeta {
    std::uint8_t flags;
    std::uint8_t k;
    std::size_t length;
    std::size_t orig_len;
};
std::pair<std::vector<std::uint8_t>, BBWTMeta>
encode_bbwt_mtf_rice(const std::vector<std::uint8_t>& block,
                     bool use_bitplane = false,
                     bool use_lfsr = false,
                     bool use_nibble = false,
                     bool use_bitrev = false,
                     bool use_gray = false,
                     std::uint8_t rice_param = 2);
std::vector<std::uint8_t> decode_bbwt_mtf_rice(const std::vector<std::uint8_t>& payload,
                                               const BBWTMeta& meta);

// New pipeline (V2) using automaton + per‑plane BBWT + RLE + Rice
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_new_pipeline(const std::vector<std::uint8_t>& block);
std::vector<std::uint8_t> decode_new_pipeline(const std::vector<std::uint8_t>& payload,
                                              std::size_t orig_len,
                                              const std::unordered_map<std::string, std::string>& meta);

// Top level MDL compressor and decompressor
std::vector<std::uint8_t> compress_blocks(const std::vector<std::uint8_t>& data,
                                          std::size_t block_size = 8192);
std::vector<std::uint8_t> decompress(const std::vector<std::uint8_t>& data);

// -----------------------------------------------------------------------------
// Implementation
//
// The following section contains the actual function definitions.  Each
// function includes explanatory comments describing the logic behind the
// translation.  Where the original Python relied on dynamic typing or
// built‑in convenience functions, the C++ version uses explicit types
// (std::vector, std::uint8_t, etc.) and iterators.  Helper lambdas are used
// sparingly; readability and clarity were prioritised over terseness.

// ULEB128 encoding: encode a non‑negative integer into unsigned LEB128.
std::vector<std::uint8_t> uleb128_encode(std::uint64_t n) {
    std::vector<std::uint8_t> out;
    // Repeatedly emit 7 bits per iteration; set high bit if more remain.
    while (true) {
        std::uint8_t byte = n & 0x7F;
        n >>= 7;
        if (n != 0) {
            // More data remains: set continuation bit.
            out.push_back(static_cast<std::uint8_t>(byte | 0x80));
        } else {
            out.push_back(byte);
            break;
        }
    }
    return out;
}

// ULEB128 decoding: decode a value from data starting at position pos.
std::pair<std::uint64_t, std::size_t>
uleb128_decode_stream(const std::vector<std::uint8_t>& data, std::size_t pos) {
    std::uint64_t result = 0;
    std::uint32_t shift = 0;
    std::size_t i = pos;
    while (true) {
        if (i >= data.size()) {
            throw std::runtime_error("Truncated ULEB128");
        }
        std::uint8_t b = data[i++];
        result |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) {
            break;
        }
        shift += 7;
    }
    return {result, i};
}

// Fast gear table used by content defined chunking.  A deterministic table
// seeded with a constant ensures reproducibility of boundaries across runs.
static std::array<std::uint32_t, 256> build_gear_table(std::uint32_t seed = 2025) {
    std::mt19937 gen(seed);
    std::array<std::uint32_t, 256> tbl{};
    for (auto& entry : tbl) {
        entry = gen();
    }
    return tbl;
}

static const std::array<std::uint32_t, 256> GEAR_TABLE = build_gear_table();

// Simplified FastCDC: identify cut boundaries based on gear hash.  The
// algorithm rolls a 32‑bit hash across the data stream and cuts whenever
// certain low bits of the hash are zero.  If no cut is found before
// max_size, we forcibly split to prevent giant chunks.
BoundaryList cdc_fast_boundaries(const std::vector<std::uint8_t>& data,
                                 std::size_t min_size,
                                 std::size_t avg_size,
                                 std::size_t max_size) {
    std::size_t n = data.size();
    BoundaryList boundaries;
    if (n == 0) {
        return boundaries;
    }
    // Choose mask bits such that 2^k ≈ average size.  We clamp k into [6,20].
    std::uint32_t k = std::max<std::uint32_t>(6, std::min<std::uint32_t>(20,
                                          static_cast<std::uint32_t>(std::bit_width(avg_size) - 1)));
    std::uint32_t mask = (1u << k) - 1u;
    std::size_t i = 0;
    while (i < n) {
        std::size_t start = i;
        std::uint32_t h = 0;
        std::size_t end_min = std::min(n, start + min_size);
        std::size_t end_max = std::min(n, start + max_size);
        i = end_min;
        // Search for a zeroed mask within the allowed window.
        while (i < end_max) {
            h = ((h << 1) & 0xFFFFFFFFu) + GEAR_TABLE[data[i]];
            if ((h & mask) == 0u) {
                ++i;
                break;
            }
            ++i;
        }
        boundaries.emplace_back(start, i);
    }
    return boundaries;
}

// Duval's Lyndon factorisation: decompose the input into a sequence of
// non‑increasing Lyndon words.  Each pair represents the half‑open interval
// [start, end) of the factor within the original string.  The algorithm is
// linear in the length of the input.
RangeList duval_lyndon(const std::vector<std::uint8_t>& s) {
    std::size_t n = s.size();
    RangeList out;
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i + 1;
        std::size_t k = i;
        // Determine the end of the smallest Lyndon word starting at i.
        while (j < n && s[k] <= s[j]) {
            if (s[k] < s[j]) {
                k = i;
            } else {
                ++k;
            }
            ++j;
        }
        std::size_t p = j - k;
        // Output repeats of this factor as long as they fit.
        while (i <= k) {
            out.emplace_back(i, i + p);
            i += p;
        }
    }
    return out;
}

// Construct suffix array using prefix doubling.  Because the input size for
// each factor (up to 2*m for a factor of length m) is small compared to a
// typical block, the O(n log n) complexity is acceptable.  Returns a vector
// of starting positions sorted lexicographically.
static std::vector<std::size_t> sa_prefix_doubling(const std::vector<std::uint8_t>& t) {
    std::size_t n = t.size();
    if (n == 0) {
        return {};
    }
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<int> rank(n);
    for (std::size_t i = 0; i < n; ++i) {
        rank[i] = static_cast<int>(t[i]);
    }
    std::vector<int> tmp(n);
    std::size_t k = 1;
    while (true) {
        // Sort by (rank[i], rank[i+k]) using lambda comparator.
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            int ra = rank[a];
            int rb = rank[b];
            int ra_k = (a + k < n) ? rank[a + k] : -1;
            int rb_k = (b + k < n) ? rank[b + k] : -1;
            if (ra != rb) return ra < rb;
            return ra_k < rb_k;
        });
        // Recompute temporary ranks.
        tmp[idx[0]] = 0;
        for (std::size_t i = 1; i < n; ++i) {
            std::size_t a = idx[i - 1];
            std::size_t b = idx[i];
            int ra = rank[a];
            int rb = rank[b];
            int ra_k = (a + k < n) ? rank[a + k] : -1;
            int rb_k = (b + k < n) ? rank[b + k] : -1;
            tmp[b] = tmp[a] + ((ra != rb || ra_k != rb_k) ? 1 : 0);
        }
        rank = tmp;
        if (rank[idx.back()] == static_cast<int>(n) - 1) {
            break;
        }
        k <<= 1;
    }
    return idx;
}

// Forward bijective BWT: factorise s into Lyndon words, compute rotation order
// per factor and merge them using a priority queue.  The output is the last
// column of the sorted rotations (as in the classical BWT) but forms a
// bijection without the need for an end‑of‑file marker.
std::vector<std::uint8_t> bbwt_forward(const std::vector<std::uint8_t>& s) {
    if (s.empty()) {
        return {};
    }
    RangeList facs = duval_lyndon(s);
    // Compute rotation order for each factor using doubling suffix array.
    struct FactorData {
        std::vector<std::uint8_t> word;
        std::vector<std::size_t> order;
    };
    std::vector<FactorData> factors;
    factors.reserve(facs.size());
    for (auto [a, b] : facs) {
        std::vector<std::uint8_t> w(s.begin() + a, s.begin() + b);
        std::size_t m = w.size();
        // Double the factor to model cyclic rotations.
        std::vector<std::uint8_t> ww = w;
        ww.insert(ww.end(), w.begin(), w.end());
        std::vector<std::size_t> sa = sa_prefix_doubling(ww);
        // Select starting positions less than m to obtain rotation order.
        std::vector<std::size_t> rot_order;
        rot_order.reserve(m);
        for (std::size_t p : sa) {
            if (p < m) rot_order.push_back(p);
        }
        factors.push_back({std::move(w), std::move(rot_order)});
    }
    // Node used in the heap merge compares two rotations lexicographically.
    struct Node {
        std::size_t fi;      // index of factor
        std::size_t k;       // index within rotation order
        const std::vector<std::uint8_t>* w;
        const std::vector<std::size_t>* order;
    };
    // Custom comparator for std::priority_queue (min‑heap).  The Python
    // implementation compares up to m+n bytes of two rotations; here we
    // implement a deterministic ordering consistent with the original code.
    auto cmp = [](const Node& a, const Node& b) {
        const auto& wA = *a.w;
        const auto& wB = *b.w;
        std::size_t i = (*a.order)[a.k];
        std::size_t j = (*b.order)[b.k];
        std::size_t m = wA.size();
        std::size_t n = wB.size();
        // Compare rotations u[i:] + u[:i] and v[j:] + v[:j].  We only need
        // to inspect up to m+n bytes to distinguish them.
        std::size_t p = 0;
        while (p < m + n) {
            std::uint8_t cu = wA[(i + p) % m];
            std::uint8_t cv = wB[(j + p) % n];
            if (cu != cv) return cu > cv;  // reverse for min‑heap
            ++p;
        }
        // Ties broken by factor index and rotation index.
        return std::tie(a.fi, i) > std::tie(b.fi, j);
    };
    // Build initial heap: one node per factor at k=0.
    std::vector<Node> heap;
    heap.reserve(factors.size());
    for (std::size_t fi = 0; fi < factors.size(); ++fi) {
        if (!factors[fi].order.empty()) {
            Node nd{fi, 0, &factors[fi].word, &factors[fi].order};
            heap.push_back(nd);
        }
    }
    // Convert vector into a heap.
    std::make_heap(heap.begin(), heap.end(), cmp);
    std::vector<std::uint8_t> out;
    out.reserve(s.size());
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        Node nd = heap.back();
        heap.pop_back();
        std::size_t i = (*nd.order)[nd.k];
        const auto& w = *nd.w;
        std::size_t m = w.size();
        // Append the character preceding the rotation start (cyclically).
        out.push_back(w[(i + m - 1) % m]);
        // Advance within this factor; push back into heap if more rotations remain.
        nd.k += 1;
        if (nd.k < nd.order->size()) {
            std::push_heap(heap.begin(), heap.end(), cmp);
            heap.push_back(nd);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    return out;
}

// Invert the bijective BWT.  The inverse recovers factors by computing
// permutation cycles over the sorted indices of L.  The method follows the
// algorithm described in the Python code and reconstructs the original
// concatenation of Lyndon words in reverse order.
std::vector<std::uint8_t> bbwt_inverse(const std::vector<std::uint8_t>& L) {
    std::size_t n = L.size();
    if (n == 0) return {};
    // Sort indices by (L[idx], idx) to build the permutation.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::make_pair(L[a], a) < std::make_pair(L[b], b);
    });
    // pi[i] points to the next index in the cycle.
    std::vector<std::size_t> pi = order;
    std::vector<bool> seen(n, false);
    std::vector<std::vector<std::size_t>> cycles;
    for (std::size_t i = 0; i < n; ++i) {
        if (!seen[i]) {
            std::vector<std::size_t> cyc;
            std::size_t cur = i;
            while (!seen[cur]) {
                seen[cur] = true;
                cyc.push_back(cur);
                cur = pi[cur];
            }
            cycles.push_back(std::move(cyc));
        }
    }
    // Sort cycles by the minimal index to preserve order.
    std::sort(cycles.begin(), cycles.end(), [](const auto& a, const auto& b) {
        return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end());
    });
    // Reconstruct factors; each cycle yields one Lyndon word.
    std::vector<std::vector<std::uint8_t>> factors;
    factors.reserve(cycles.size());
    for (const auto& cyc : cycles) {
        std::size_t i0 = *std::min_element(cyc.begin(), cyc.end());
        std::size_t d = cyc.size();
        std::size_t cur = i0;
        std::vector<std::uint8_t> seq;
        seq.reserve(d);
        for (std::size_t k = 0; k < d; ++k) {
            cur = pi[cur];
            seq.push_back(L[cur]);
        }
        factors.push_back(seq);
    }
    // Concatenate factors in reverse order.
    std::vector<std::uint8_t> out;
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        out.insert(out.end(), it->begin(), it->end());
    }
    return out;
}

// Move‑to‑front encoding: maintain a list of 256 symbols; for each input byte
// output its index and move it to the front.  Note that the output values
// may exceed 255 if the original data contains bytes ≥256, but here we
// restrict to 8‑bit bytes so indices lie in [0,255].
std::vector<std::uint8_t> mtf_encode(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> table(256);
    std::iota(table.begin(), table.end(), 0);
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (std::uint8_t b : data) {
        // Find the index of b in the current table.
        auto it = std::find(table.begin(), table.end(), b);
        std::uint8_t idx = static_cast<std::uint8_t>(std::distance(table.begin(), it));
        out.push_back(idx);
        // Move b to the front.
        table.erase(it);
        table.insert(table.begin(), b);
    }
    return out;
}

// Move‑to‑front decoding: invert the MTF transform by maintaining the same
// table.  Each index refers to the current table; the corresponding symbol
// is output and moved to the front.
std::vector<std::uint8_t> mtf_decode(const std::vector<std::uint8_t>& seq) {
    std::vector<std::uint8_t> table(256);
    std::iota(table.begin(), table.end(), 0);
    std::vector<std::uint8_t> out;
    out.reserve(seq.size());
    for (std::uint8_t idx : seq) {
        std::uint8_t b = table[idx];
        out.push_back(b);
        table.erase(table.begin() + idx);
        table.insert(table.begin(), b);
    }
    return out;
}

// Bitplane interleaving: for each block of 8 bytes, emit 8 bytes whose
// positions group together the corresponding bitplane.  The transform is
// self‑inverting (applying it twice restores the original).
std::vector<std::uint8_t> bitplane_interleave(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    auto it = data.begin();
    while (it != data.end()) {
        // Read up to 8 bytes (pad with zeros if fewer remain).
        std::array<std::uint8_t, 8> block{};
        std::size_t count = 0;
        for (; count < 8 && it != data.end(); ++count, ++it) {
            block[count] = *it;
        }
        // Pad remaining entries with zeros.
        for (std::size_t bit = 0; bit < 8; ++bit) {
            std::uint8_t v = 0;
            for (std::size_t i = 0; i < 8; ++i) {
                std::uint8_t b = block[i];
                v |= ((b >> (7 - bit)) & 1u) << (7 - i);
            }
            out.push_back(v);
        }
    }
    return out;
}

// Bitplane de‑interleaving: invert bitplane_interleave by reconstructing
// original bytes.  orig_len specifies the length of the original data before
// padding.  Blocks are processed in groups of 8 interleaved bytes.
std::vector<std::uint8_t> bitplane_deinterleave(const std::vector<std::uint8_t>& data,
                                                std::size_t orig_len) {
    std::vector<std::uint8_t> out;
    out.reserve(orig_len);
    auto it = data.begin();
    while (it != data.end()) {
        std::array<std::uint8_t, 8> block{};
        for (std::size_t i = 0; i < 8 && it != data.end(); ++i, ++it) {
            block[i] = *it;
        }
        // Initialize eight output bytes to zero.
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t bit = 0; bit < 8; ++bit) {
            std::uint8_t byte = block[bit];
            for (std::size_t i = 0; i < 8; ++i) {
                bytes[i] |= ((byte >> (7 - i)) & 1u) << (7 - bit);
            }
        }
        for (std::size_t i = 0; i < 8; ++i) {
            out.push_back(bytes[i]);
        }
    }
    // Trim to original length (the last block may have contained padding).
    if (out.size() > orig_len) out.resize(orig_len);
    return out;
}

// LFSR whitening: produce a pseudo‑random stream via an 8‑bit linear feedback
// shift register (LFSR) with the given taps and seed.  Each input byte is
// XORed with the current state; the same function applied twice yields the
// original sequence.  The taps and seed defaults match the Python code.
std::vector<std::uint8_t> lfsr_whiten(const std::vector<std::uint8_t>& data,
                                      std::uint8_t taps,
                                      std::uint8_t seed) {
    std::uint8_t state = seed & 0xFFu;
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (std::uint8_t b : data) {
        out.push_back(static_cast<std::uint8_t>(b ^ state));
        // Compute feedback bit as XOR of tapped bits of state.
        std::uint8_t fb = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if ((taps >> bit) & 1u) {
                fb ^= (state >> bit) & 1u;
            }
        }
        state = static_cast<std::uint8_t>((state << 1) | fb);
    }
    return out;
}

// Swap high and low 4‑bit nibbles of each byte.  Applying nibble_swap twice
// returns the original data.
std::vector<std::uint8_t> nibble_swap(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (std::uint8_t b : data) {
        std::uint8_t hi = (b & 0xF0u) >> 4;
        std::uint8_t lo = (b & 0x0Fu) << 4;
        out.push_back(static_cast<std::uint8_t>(hi | lo));
    }
    return out;
}

// Bit reversal lookup table for 8‑bit values.  Precompute once to speed up
// bit_reverse().
static const std::array<std::uint8_t, 256> BIT_REVERSE_TABLE = []{
    std::array<std::uint8_t, 256> table{};
    for (int i = 0; i < 256; ++i) {
        std::bitset<8> bs(i);
        // Reverse the bits and convert back to integer.
        std::bitset<8> reversed;
        for (int k = 0; k < 8; ++k) {
            reversed[7 - k] = bs[k];
        }
        table[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(reversed.to_ulong());
    }
    return table;
}();

// Reverse the bit order of each byte using the lookup table above.
std::vector<std::uint8_t> bit_reverse(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (std::uint8_t b : data) {
        out.push_back(BIT_REVERSE_TABLE[b]);
    }
    return out;
}

// Gate NOT limited to a specific bit‑width.  We mask the complement so that
// only the lowest width bits are retained (two's complement semantics).
std::uint8_t gate_not(std::uint8_t a, std::uint8_t width) {
    std::uint8_t lane_mask = static_cast<std::uint8_t>((1u << width) - 1u);
    return static_cast<std::uint8_t>((~a) & lane_mask);
}

// Gate XOR expressed in terms of OR, AND and NOT.  We mask intermediate
// results to prevent overflow outside the active lane.  See Python comment
// for derivation: XOR = (a OR b) AND NOT(a AND b).
std::uint8_t gate_xor(std::uint8_t a, std::uint8_t b, std::uint8_t width) {
    std::uint8_t or_result = static_cast<std::uint8_t>(a | b);
    std::uint8_t and_result = static_cast<std::uint8_t>(a & b);
    std::uint8_t not_and = gate_not(and_result, width);
    return static_cast<std::uint8_t>(or_result & not_and);
}

// Intra‑word prefix OR using a log‑doubling schedule.  Starting from the
// least significant bit, propagate ones forward so that each bit becomes the
// OR of all lower (including itself) bits.  width must be ≤8 for bytes.
std::uint8_t prefix_or_word(std::uint8_t x, std::uint8_t width) {
    std::uint8_t result = static_cast<std::uint8_t>(x & ((1u << width) - 1u));
    std::uint8_t lane_mask = static_cast<std::uint8_t>((1u << width) - 1u);
    std::uint8_t distance = 1;
    while (distance < width) {
        // Shift result left by distance and OR; mask to width bits.
        result = static_cast<std::uint8_t>(result | ((result << distance) & lane_mask));
        distance <<= 1;
    }
    return result;
}

// left_band implements a left‑closed, right‑open band of ones of length L
// beginning at the least significant set bit of beta.  Computed purely via
// prefix ORs and masking.
std::uint8_t left_band(std::uint8_t beta, std::uint8_t L, std::uint8_t width) {
    std::uint8_t lane_mask = static_cast<std::uint8_t>((1u << width) - 1u);
    std::uint8_t pref = prefix_or_word(static_cast<std::uint8_t>(beta & lane_mask), width);
    std::uint8_t cut = prefix_or_word(static_cast<std::uint8_t>((beta << L) & lane_mask), width);
    return static_cast<std::uint8_t>((pref & gate_not(cut, width)) & lane_mask);
}

// Branch‑free byte equality: returns 0xFF if a == b else 0x00.  Based on
// gate logic: compute x = a XOR b, reduce to a single bit any_one, invert
// this bit to get equality and replicate across the lane.
std::uint8_t byte_eq(std::uint8_t a, std::uint8_t b) {
    std::uint8_t width = 8;
    std::uint8_t lane_mask = static_cast<std::uint8_t>((1u << width) - 1u);
    // x = a XOR b within the lane.
    std::uint8_t x = gate_xor(static_cast<std::uint8_t>(a & lane_mask), static_cast<std::uint8_t>(b & lane_mask), width);
    // OR‑fold to detect any set bit.
    std::uint8_t x1 = static_cast<std::uint8_t>(x | ((x >> 4) & 0x0Fu));
    x1 = static_cast<std::uint8_t>(x1 | ((x1 >> 2) & 0x3Fu));
    x1 = static_cast<std::uint8_t>(x1 | ((x1 >> 1) & 0x7Fu));
    std::uint8_t any_one = static_cast<std::uint8_t>(x1 & 1u);
    // eq_bit = NOT(any_one) in a 1‑bit lane.
    std::uint8_t eq_bit = static_cast<std::uint8_t>(gate_not(any_one, 1) & 1u);
    // Replicate eq_bit across all bits of a byte.
    std::uint8_t mask = 0;
    for (int i = 0; i < 8; ++i) {
        mask |= static_cast<std::uint8_t>((eq_bit << i) & 0xFFu);
    }
    return mask;
}

// 2:1 multiplexer on a byte lane.  Canonical masks are 0xFF (select a) or
// 0x00 (select b).  Non‑canonical masks are supported but may blend bits.
std::uint8_t mux_byte(std::uint8_t m, std::uint8_t a, std::uint8_t b, std::uint8_t width) {
    return static_cast<std::uint8_t>(((m & a) | (gate_not(m, width) & b)) & ((1u << width) - 1u));
}

// Gray predictor: compute g = v XOR (v >> 1).  Use gate_xor to preserve lane.
std::uint8_t gray_pred(std::uint8_t v) {
    return gate_xor(v, static_cast<std::uint8_t>(v >> 1), 8);
}

// Mode 1 forward: order‑1 residual (y[i] = x[i] XOR x[i‑1]).  The first
// symbol seeds the recursion.
std::vector<std::uint8_t> mode1_forward(const std::vector<std::uint8_t>& block) {
    if (block.empty()) return {};
    std::size_t length = block.size();
    std::vector<std::uint8_t> out(length);
    out[0] = block[0];
    for (std::size_t i = 1; i < length; ++i) {
        out[i] = gate_xor(block[i], block[i - 1], 8);
    }
    return out;
}

// Mode 1 inverse: reconstruct x from residuals and history.
std::vector<std::uint8_t> mode1_inverse(const std::vector<std::uint8_t>& residual) {
    if (residual.empty()) return {};
    std::size_t length = residual.size();
    std::vector<std::uint8_t> out(length);
    out[0] = residual[0];
    for (std::size_t i = 1; i < length; ++i) {
        out[i] = gate_xor(residual[i], out[i - 1], 8);
    }
    return out;
}

// Mode 2 forward: Gray predictor residual (y[i] = x[i] XOR Gray(x[i‑1])).
std::vector<std::uint8_t> mode2_forward(const std::vector<std::uint8_t>& block) {
    if (block.empty()) return {};
    std::size_t length = block.size();
    std::vector<std::uint8_t> out(length);
    out[0] = block[0];
    for (std::size_t i = 1; i < length; ++i) {
        std::uint8_t predictor = gray_pred(block[i - 1]);
        out[i] = gate_xor(block[i], predictor, 8);
    }
    return out;
}

// Mode 2 inverse: recover original bytes using Gray predictor from decoded history.
std::vector<std::uint8_t> mode2_inverse(const std::vector<std::uint8_t>& residual) {
    if (residual.empty()) return {};
    std::size_t length = residual.size();
    std::vector<std::uint8_t> out(length);
    out[0] = residual[0];
    for (std::size_t i = 1; i < length; ++i) {
        std::uint8_t predictor = gray_pred(out[i - 1]);
        out[i] = gate_xor(residual[i], predictor, 8);
    }
    return out;
}

// Mode 3 forward: order‑2 residual: y[0] = x[0], y[1] = x[1] XOR x[0],
// for i≥2: y[i] = x[i] XOR x[i‑2].
std::vector<std::uint8_t> mode3_forward(const std::vector<std::uint8_t>& block) {
    std::size_t length = block.size();
    if (length == 0) return {};
    if (length == 1) return {block[0]};
    std::vector<std::uint8_t> out(length);
    out[0] = block[0];
    out[1] = gate_xor(block[1], block[0], 8);
    for (std::size_t i = 2; i < length; ++i) {
        out[i] = gate_xor(block[i], block[i - 2], 8);
    }
    return out;
}

// Mode 3 inverse: invert order‑2 residual by reconstructing from history.
std::vector<std::uint8_t> mode3_inverse(const std::vector<std::uint8_t>& residual) {
    std::size_t length = residual.size();
    if (length == 0) return {};
    if (length == 1) return {residual[0]};
    std::vector<std::uint8_t> out(length);
    out[0] = residual[0];
    out[1] = gate_xor(residual[1], out[0], 8);
    for (std::size_t i = 2; i < length; ++i) {
        out[i] = gate_xor(residual[i], out[i - 2], 8);
    }
    return out;
}

// Mode 4 forward: run‑segment selector.  For i>=2 choose predictor = x[i‑1]
// (if x[i‑1]==x[i‑2]) else Gray(x[i‑1]).  Residual is XOR with predictor.
std::vector<std::uint8_t> mode4_forward(const std::vector<std::uint8_t>& block) {
    std::size_t length = block.size();
    if (length == 0) return {};
    if (length == 1) return {block[0]};
    std::vector<std::uint8_t> out(length);
    out[0] = block[0];
    out[1] = gate_xor(block[1], block[0], 8);
    for (std::size_t i = 2; i < length; ++i) {
        std::uint8_t selector_mask = byte_eq(block[i - 1], block[i - 2]);
        std::uint8_t pred_run = block[i - 1];
        std::uint8_t pred_alt = gray_pred(block[i - 1]);
        std::uint8_t predictor = mux_byte(selector_mask, pred_run, pred_alt);
        out[i] = gate_xor(block[i], predictor, 8);
    }
    return out;
}

// Mode 4 inverse: recompute predictor from already decoded bytes.
std::vector<std::uint8_t> mode4_inverse(const std::vector<std::uint8_t>& residual) {
    std::size_t length = residual.size();
    if (length == 0) return {};
    if (length == 1) return {residual[0]};
    std::vector<std::uint8_t> out(length);
    out[0] = residual[0];
    out[1] = gate_xor(residual[1], out[0], 8);
    for (std::size_t i = 2; i < length; ++i) {
        std::uint8_t selector_mask = byte_eq(out[i - 1], out[i - 2]);
        std::uint8_t pred_run = out[i - 1];
        std::uint8_t pred_alt = gray_pred(out[i - 1]);
        std::uint8_t predictor = mux_byte(selector_mask, pred_run, pred_alt);
        out[i] = gate_xor(residual[i], predictor, 8);
    }
    return out;
}

// Approximate first‑order (Markov) bit entropy of a byte block.  We treat
// each byte as 8 bits and estimate H(bit_t | bit_{t‑1}).  For short
// sequences return maximal entropy of 1.0.
double first_order_bit_entropy(const std::vector<std::uint8_t>& block) {
    std::size_t nbits = block.size() * 8;
    if (nbits < 2) return 1.0;
    // counts[x][y] counts transitions from bit x to bit y.
    std::array<std::array<std::uint64_t, 2>, 2> counts{};
    // Initialize prev bit to the MSB of first byte.
    std::uint8_t prev = (block[0] >> 7) & 1u;
    std::size_t bit_idx = 1;
    bool skip_first = true;
    for (std::uint8_t b : block) {
        for (int k = 0; k < 8; ++k) {
            if (skip_first) {
                // Skip the very first bit; it has no predecessor.
                skip_first = false;
                continue;
            }
            std::uint8_t bit = (b >> (7 - k)) & 1u;
            counts[prev][bit] += 1;
            prev = bit;
            ++bit_idx;
        }
    }
    double total_trans = static_cast<double>(nbits - 1);
    double H = 0.0;
    for (int x = 0; x <= 1; ++x) {
        double px = static_cast<double>(counts[x][0] + counts[x][1]);
        if (px == 0) continue;
        for (int y = 0; y <= 1; ++y) {
            double pxy = static_cast<double>(counts[x][y]);
            if (pxy == 0) continue;
            double p_y_given_x = pxy / px;
            H -= (pxy / total_trans) * std::log2(p_y_given_x);
        }
    }
    return H;
}

// Compute 0th‑order entropy of a byte array.  Used by the automaton selector.
static double zero_order_entropy(const std::vector<std::uint8_t>& data) {
    if (data.empty()) return 0.0;
    std::array<std::uint64_t, 256> counts{};
    for (auto b : data) counts[b]++;
    double h = 0.0;
    double n = static_cast<double>(data.size());
    for (auto c : counts) {
        if (c == 0) continue;
        double p = static_cast<double>(c) / n;
        h -= p * std::log2(p);
    }
    return h;
}

// Select the best reversible automaton by minimum 0th‑order entropy.  Modes
// considered: identity (0), mode1, mode2, mode3, mode4.  Ties favour
// smaller mode index to ensure deterministic behaviour.  Returns both
// transformed data and the chosen mode identifier.
AutomatonResult circuit_map_automaton_forward(const std::vector<std::uint8_t>& block) {
    double best_entropy = zero_order_entropy(block);
    std::uint8_t best_mode = 0;
    std::vector<std::uint8_t> best_bytes = block;
    // Candidates for modes 1..4.
    std::vector<std::pair<std::uint8_t, std::vector<std::uint8_t>>> candidates;
    candidates.emplace_back(1, mode1_forward(block));
    candidates.emplace_back(2, mode2_forward(block));
    candidates.emplace_back(3, mode3_forward(block));
    candidates.emplace_back(4, mode4_forward(block));
    for (const auto& [mode_id, mapped] : candidates) {
        double ent = zero_order_entropy(mapped);
        if (ent < best_entropy - 1e-9 ||
            (std::abs(ent - best_entropy) <= 1e-9 && mode_id < best_mode)) {
            best_entropy = ent;
            best_mode = mode_id;
            best_bytes = mapped;
        }
    }
    return {best_bytes, best_mode};
}

// Inverse of the selected automaton.  Given the mode id, dispatch to the
// appropriate inverse function.  Unknown modes fall back to identity.
std::vector<std::uint8_t>
circuit_map_automaton_inverse(const std::vector<std::uint8_t>& mapped,
                              std::uint8_t mode) {
    switch (mode) {
        case 0: return mapped;
        case 1: return mode1_inverse(mapped);
        case 2: return mode2_inverse(mapped);
        case 3: return mode3_inverse(mapped);
        case 4: return mode4_inverse(mapped);
        default: return mapped;
    }
}

// Split bytes -> 8 MSB-first bit-planes.
// planes[j][t] is bit j (0..7, MSB-first) of data[t] as 0/1.
std::tuple<std::vector<std::vector<int>>, std::size_t>
bytes_to_bitplanes(const std::vector<std::uint8_t>& data) {
    std::size_t L = data.size();
    std::vector<std::vector<int>> planes(8, std::vector<int>(L));
    for (std::size_t t = 0; t < L; ++t) {
        std::uint8_t b = data[t];
        // MSB-first: j=0 => bit7, j=7 => bit0
        planes[0][t] = (b >> 7) & 1;
        planes[1][t] = (b >> 6) & 1;
        planes[2][t] = (b >> 5) & 1;
        planes[3][t] = (b >> 4) & 1;
        planes[4][t] = (b >> 3) & 1;
        planes[5][t] = (b >> 2) & 1;
        planes[6][t] = (b >> 1) & 1;
        planes[7][t] = (b >> 0) & 1;
    }
    return {planes, L};
}

// Reconstruct bytes from 8 MSB-first bit-planes.
// Expects exactly 8 planes and equal lengths.
std::vector<std::uint8_t> bitplanes_to_bytes(const std::vector<std::vector<int>>& planes) {
    if (planes.empty()) return {};
    assert(planes.size() == 8 && "bitplanes_to_bytes expects 8 planes");
    std::size_t L = planes[0].size();
    for (std::size_t j = 1; j < 8; ++j) {
        assert(planes[j].size() == L && "all planes must have same length");
    }

    std::vector<std::uint8_t> out(L);
    for (std::size_t t = 0; t < L; ++t) {
        std::uint8_t val = static_cast<std::uint8_t>(
            ((planes[0][t] & 1) << 7) |
            ((planes[1][t] & 1) << 6) |
            ((planes[2][t] & 1) << 5) |
            ((planes[3][t] & 1) << 4) |
            ((planes[4][t] & 1) << 3) |
            ((planes[5][t] & 1) << 2) |
            ((planes[6][t] & 1) << 1) |
            ((planes[7][t] & 1) << 0)
        );
        out[t] = val;
    }
    return out;
}

// Average run length for a binary sequence.  Count number of runs and divide
// length by run count.
double avg_run_bits(const std::vector<int>& bits) {
    if (bits.empty()) return 0.0;
    std::size_t runs = 1;
    int prev = bits[0];
    for (std::size_t i = 1; i < bits.size(); ++i) {
        if (bits[i] != prev) {
            runs++;
            prev = bits[i];
        }
    }
    return static_cast<double>(bits.size()) / runs;
}

// 0th‑order entropy of a binary sequence.  Uses natural log base 2.
double H0_bits(const std::vector<int>& bits) {
    if (bits.empty()) return 0.0;
    std::size_t n = bits.size();
    std::size_t c1 = std::count(bits.begin(), bits.end(), 1);
    double p = static_cast<double>(c1) / static_cast<double>(n);
    if (p == 0.0 || p == 1.0) return 0.0;
    return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
}

// Run‑length encode a binary sequence: returns first bit and a list of run
// lengths.  Consecutive identical bits are grouped.
std::pair<int, std::vector<int>> rle_binary(const std::vector<int>& bits) {
    if (bits.empty()) return {0, {}};
    std::vector<int> runs;
    runs.reserve(bits.size());
    int cur = 1;
    for (std::size_t i = 1; i < bits.size(); ++i) {
        if (bits[i] == bits[i - 1]) {
            cur++;
        } else {
            runs.push_back(cur);
            cur = 1;
        }
    }
    runs.push_back(cur);
    return {bits[0], runs};
}

// Inverse of rle_binary: reconstruct bits given first bit and run lengths.
std::vector<int> unrle_binary(int first_bit, const std::vector<int>& runs) {
    std::vector<int> out;
    out.reserve(std::accumulate(runs.begin(), runs.end(), 0));
    int b = first_bit & 1;
    for (int r : runs) {
        out.insert(out.end(), r, b);
        b ^= 1;
    }
    return out;
}

// Pack a binary vector into a byte array.  Bits are filled MSB first in
// each byte.  The number of output bytes is (len(bits)+7)//8.
std::vector<std::uint8_t> pack_bits_to_bytes(const std::vector<int>& bits) {
    std::size_t n = bits.size();
    std::vector<std::uint8_t> out((n + 7) / 8);
    for (std::size_t i = 0; i < n; ++i) {
        if (bits[i] & 1) {
            out[i >> 3] |= static_cast<std::uint8_t>(1u << (7 - (i & 7)));
        }
    }
    return out;
}

// Unpack a bit vector from a byte buffer.  Only the first nbits bits are
// produced; trailing bits are ignored.
std::vector<int> unpack_bits_from_bytes(const std::vector<std::uint8_t>& buf,
                                        std::size_t nbits) {
    std::vector<int> out(nbits);
    for (std::size_t i = 0; i < nbits; ++i) {
        out[i] = ((buf[i >> 3] >> (7 - (i & 7))) & 1u) != 0;
    }
    return out;
}

// Rice encoding of a list of non‑negative integers.  Parameter k selects
// 2^k as the modulus.  Unary quotient followed by fixed k‑bit remainder.
std::vector<std::uint8_t> rice_encode(const std::vector<std::uint64_t>& seq,
                                      std::uint8_t k) {
    std::string bitstr;
    bitstr.reserve(seq.size() * 8);
    std::uint64_t M = static_cast<std::uint64_t>(1) << k;
    for (auto n : seq) {
        std::uint64_t q = n / M;
        std::uint64_t r = n % M;
        // Append q ones followed by a zero.
        bitstr.append(q, '1');
        bitstr.push_back('0');
        // Append remainder bits if k>0.
        if (k > 0) {
            for (int bit = k - 1; bit >= 0; --bit) {
                bitstr.push_back(((r >> bit) & 1u) ? '1' : '0');
            }
        }
    }
    // Pad bitstring to byte boundary with zeros.
    std::size_t pad = (8 - (bitstr.size() % 8)) % 8;
    bitstr.append(pad, '0');
    // Pack bits into bytes.
    std::vector<std::uint8_t> out;
    out.reserve(bitstr.size() / 8);
    for (std::size_t i = 0; i < bitstr.size(); i += 8) {
        std::uint8_t b = 0;
        for (int j = 0; j < 8; ++j) {
            b = static_cast<std::uint8_t>((b << 1) | (bitstr[i + j] == '1' ? 1u : 0u));
        }
        out.push_back(b);
    }
    return out;
}

// Rice decoding: decode nvals integers from payload using parameter k.  The
// bitstring is interpreted in order; if insufficient bits remain, an error
// is thrown.  Works for k>=0.
std::vector<std::uint64_t> rice_decode(const std::vector<std::uint8_t>& data,
                                       std::uint8_t k,
                                       std::size_t nvals) {
    // Convert bytes to a bitstring representation.
    std::string bitstr;
    bitstr.reserve(data.size() * 8);
    for (auto b : data) {
        for (int bit = 7; bit >= 0; --bit) {
            bitstr.push_back(((b >> bit) & 1u) ? '1' : '0');
        }
    }
    std::size_t i = 0;
    std::uint64_t M = static_cast<std::uint64_t>(1) << k;
    std::vector<std::uint64_t> out;
    out.reserve(nvals);
    auto need = [&](std::size_t bits) {
        return i + bits <= bitstr.size();
    };
    for (std::size_t v = 0; v < nvals; ++v) {
        // Read unary quotient q: count ones until first zero.
        std::uint64_t q = 0;
        while (true) {
            if (!need(1)) {
                throw std::runtime_error("Rice stream truncated while reading unary part");
            }
            if (bitstr[i] == '1') {
                q++;
                i++;
            } else {
                i++;
                break;
            }
        }
        // Read remainder r of k bits.
        std::uint64_t r = 0;
        if (k > 0) {
            if (!need(k)) {
                throw std::runtime_error("Rice stream truncated while reading remainder");
            }
            for (std::uint8_t bit = 0; bit < k; ++bit) {
                r = (r << 1) | (bitstr[i + bit] == '1' ? 1u : 0u);
            }
            i += k;
        }
        out.push_back(q * M + r);
    }
    return out;
}

// Gray encode each byte: g = x ^ (x>>1).
std::vector<std::uint8_t> gray_encode_bytes(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (auto b : data) {
        out.push_back(static_cast<std::uint8_t>((b ^ (b >> 1)) & 0xFFu));
    }
    return out;
}

// Gray decode bytes by iterative XOR with right shifts.  See Python for details.
std::vector<std::uint8_t> gray_decode_bytes(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out;
    out.reserve(data.size());
    for (auto g : data) {
        std::uint8_t n = g;
        n ^= static_cast<std::uint8_t>(n >> 1);
        n ^= static_cast<std::uint8_t>(n >> 2);
        n ^= static_cast<std::uint8_t>(n >> 4);
        out.push_back(static_cast<std::uint8_t>(n & 0xFFu));
    }
    return out;
}

// Naive LZ77 encoder.  Window size limited to 4096 bytes as in Python.  This
// implementation uses ULEB128 for length/distances and produces a stream of
// markers (0 for literal, 1 for match) followed by encoded metadata.
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_lz77(const std::vector<std::uint8_t>& block) {
    std::vector<std::uint8_t> window;
    std::vector<std::uint8_t> out;
    std::size_t pos = 0;
    std::size_t n = block.size();
    while (pos < n) {
        std::size_t best_len = 0;
        std::size_t best_dist = 0;
        // Search in a limited window for the best match (length≥3).
        std::size_t max_window = std::min<std::size_t>(window.size(), 255);
        for (std::size_t dist = 1; dist <= max_window; ++dist) {
            std::size_t length = 0;
            while (length < 255 && pos + length < n &&
                   window[window.size() - dist + length] == block[pos + length]) {
                ++length;
            }
            if (length > best_len) {
                best_len = length;
                best_dist = dist;
            }
        }
        if (best_len >= 3) {
            out.push_back(1);  // match marker
            std::vector<std::uint8_t> len_enc = uleb128_encode(best_len);
            out.insert(out.end(), len_enc.begin(), len_enc.end());
            std::vector<std::uint8_t> dist_enc = uleb128_encode(best_dist);
            out.insert(out.end(), dist_enc.begin(), dist_enc.end());
            for (std::size_t i = 0; i < best_len; ++i) {
                window.push_back(block[pos]);
                ++pos;
            }
        } else {
            out.push_back(0);  // literal marker
            out.push_back(block[pos]);
            window.push_back(block[pos]);
            ++pos;
        }
        // Keep window manageable: drop older entries beyond 4096.
        if (window.size() > 4096) {
            window.erase(window.begin(), window.end() - 4096);
        }
    }
    return {out, {}};
}

// Naive LZ77 decoder.  Accepts original length to know when to stop.  Throws
// if decompressed length mismatches.
std::vector<std::uint8_t> decode_lz77(const std::vector<std::uint8_t>& data,
                                      std::size_t orig_len) {
    std::vector<std::uint8_t> window;
    std::vector<std::uint8_t> out;
    std::size_t i = 0;
    while (i < data.size() && out.size() < orig_len) {
        std::uint8_t flag = data[i++];
        if (flag == 0) {
            // Literal
            if (i >= data.size()) throw std::runtime_error("LZ77 truncated literal");
            std::uint8_t b = data[i++];
            out.push_back(b);
            window.push_back(b);
        } else {
            // Match
            auto [length, ni] = uleb128_decode_stream(data, i);
            i = ni;
            auto [dist, nj] = uleb128_decode_stream(data, i);
            i = nj;
            for (std::size_t k = 0; k < length; ++k) {
                if (dist == 0 || dist > window.size()) {
                    throw std::runtime_error("Invalid LZ77 distance");
                }
                std::uint8_t b = window[window.size() - dist];
                out.push_back(b);
                window.push_back(b);
            }
        }
        if (window.size() > 4096) {
            window.erase(window.begin(), window.end() - 4096);
        }
    }
    if (out.size() != orig_len) {
        throw std::runtime_error("LZ77 output length mismatch");
    }
    return out;
}


// ---------- helpers: count pairs & replace non-overlapping ----------
inline std::unordered_map<std::pair<int,int>, int, PairHash>
count_pairs(const std::vector<int>& seq) {
    std::unordered_map<std::pair<int,int>, int, PairHash> freq;
    if (seq.size() < 2) return freq;
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
        auto p = std::make_pair(seq[i], seq[i+1]);
        ++freq[p];
    }
    return freq;
}

// 返回 (新序列, 替换次数)
inline std::pair<std::vector<int>, int>
replace_non_overlapping(const std::vector<int>& seq,
                        const std::pair<int,int>& target,
                        int new_sym) {
    const int a = target.first, b = target.second;
    std::vector<int> out;
    out.reserve(seq.size());
    int replaced = 0;
    for (std::size_t i = 0; i < seq.size();) {
        if (i + 1 < seq.size() && seq[i] == a && seq[i+1] == b) {
            out.push_back(new_sym);
            i += 2;
            ++replaced;
        } else {
            out.push_back(seq[i]);
            ++i;
        }
    }
    return {std::move(out), replaced};
}

// 递归+记忆化展开：与 Re-Pair SLP 定义一致（A -> XY），稳定可逆
static const std::vector<std::uint8_t>& expand_symbol(
    int sym,
    const std::unordered_map<int, std::pair<int,int>>& rules,
    std::unordered_map<int, std::vector<std::uint8_t>>& memo)
{
    if (sym < 256) {
        static std::array<std::vector<std::uint8_t>, 256> term_cache{};
        static std::once_flag once;
        std::call_once(once, []{
            for (int t = 0; t < 256; ++t)
                term_cache[t] = std::vector<std::uint8_t>{ static_cast<std::uint8_t>(t) };
        });
        return term_cache[sym];
    }
    auto it = memo.find(sym);
    if (it != memo.end()) return it->second;

    auto r = rules.find(sym);
    if (r == rules.end()) {
        throw std::runtime_error("RePair: nonterminal without rule");
    }
    const auto& left  = expand_symbol(r->second.first,  rules, memo);
    const auto& right = expand_symbol(r->second.second, rules, memo);

    std::vector<std::uint8_t> buf;
    buf.reserve(left.size() + right.size());
    buf.insert(buf.end(), left.begin(),  left.end());
    buf.insert(buf.end(), right.begin(), right.end());

    auto [ins, _] = memo.emplace(sym, std::move(buf));
    return ins->second;
}

// -----------------------------
// Strict Re-Pair compression (RP format, all ints ULEB128)
// -----------------------------
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
repair_compress(const std::vector<std::uint8_t>& block) {
    // 空输入：直接写 RP, terminals=256, nrules=0, seqlen=0
    if (block.empty()) {
        std::vector<std::uint8_t> out;
        out.push_back('R'); out.push_back('P');                // magic
        auto t = uleb128_encode(256);                          // terminals
        out.insert(out.end(), t.begin(), t.end());
        auto z = uleb128_encode(0);                            // nrules=0
        out.insert(out.end(), z.begin(), z.end());
        out.insert(out.end(), z.begin(), z.end());             // seq_len=0
        std::unordered_map<std::string,std::string> meta{
            {"rules","0"},{"final_len","0"},{"terminals","256"},{"nrules","0"}};
        return {out, meta};
    }

    std::vector<int> seq(block.begin(), block.end());          // 终结符 0..255
    int next_sym = 256;                                        // 非终结符从 256 开始
    std::vector<std::pair<int,int>> rules_vec;                 // 按创建顺序保存 RHS

    while (true) {
        auto freq = count_pairs(seq);
        if (freq.empty()) break;

        // 选频次 >=2 的最频繁 pair；频率相同按字典序稳定选
        std::pair<int,int> best_pair{0,0};
        int best_f = 1;
        bool has_cand = false;
        for (const auto& kv : freq) {
            const auto& p = kv.first; int f = kv.second;
            if (f > best_f || (f == best_f && has_cand && p < best_pair)) {
                best_pair = p; best_f = f; has_cand = true;
            } else if (!has_cand && f > 1) {
                best_pair = p; best_f = f; has_cand = true;
            }
        }
        if (!has_cand) break;

        auto [new_seq, replaced] = replace_non_overlapping(seq, best_pair, next_sym);
        if (replaced < 2) break;  // 与你的 Python 保持一致：低于 2 次不引入规则

        rules_vec.emplace_back(best_pair.first, best_pair.second);
        seq = std::move(new_seq);
        ++next_sym;
    }

    // ---- 序列化（完全按你的 Python 格式）----
    std::vector<std::uint8_t> out;
    out.push_back('R'); out.push_back('P');                    // magic
    auto enc = uleb128_encode(256);                            // terminals
    out.insert(out.end(), enc.begin(), enc.end());

    std::size_t nrules = rules_vec.size();
    auto nr_enc = uleb128_encode(nrules);                      // nrules
    out.insert(out.end(), nr_enc.begin(), nr_enc.end());

    // 规则 LHS 隐式：256+i；按创建顺序写 RHS 两个 ULEB128
    for (std::size_t i = 0; i < nrules; ++i) {
        auto a_enc = uleb128_encode(static_cast<std::size_t>(rules_vec[i].first));
        auto b_enc = uleb128_encode(static_cast<std::size_t>(rules_vec[i].second));
        out.insert(out.end(), a_enc.begin(), a_enc.end());
        out.insert(out.end(), b_enc.begin(), b_enc.end());
    }

    // 最终序列长度 & 符号（全部 ULEB128；符号可以是 >=256 的非终结符）
    auto sl_enc = uleb128_encode(seq.size());
    out.insert(out.end(), sl_enc.begin(), sl_enc.end());
    for (int s : seq) {
        auto s_enc = uleb128_encode(static_cast<std::size_t>(s));
        out.insert(out.end(), s_enc.begin(), s_enc.end());
    }

    std::unordered_map<std::string,std::string> meta{
        {"rules", std::to_string(nrules)},
        {"final_len", std::to_string(seq.size())},
        {"terminals", "256"},
        {"nrules", std::to_string(nrules)}
    };
    return {out, meta};
}

// -----------------------------
// Strict Re-Pair decompression (inverse of above)
// -----------------------------
std::vector<std::uint8_t> repair_decompress(const std::vector<std::uint8_t>& data,
                                            std::size_t orig_len) {
    std::size_t i = 0;
    if (data.size() < 2 || data[0] != 'R' || data[1] != 'P') {
        throw std::runtime_error("RePair: bad magic");
    }
    i = 2;

    // terminals
    auto [terminals, i1] = uleb128_decode_stream(data, i);
    i = i1;
    if (terminals != 256) {
        throw std::runtime_error("RePair: unsupported terminal alphabet");
    }

    // 规则条数
    auto [nrules, i2] = uleb128_decode_stream(data, i);
    i = i2;

    // 读取规则：隐式 LHS = 256 + ridx；RHS 两个都是 ULEB128（可 >=256）
    std::unordered_map<int, std::pair<int,int>> rules;
    rules.reserve(nrules);
    for (std::size_t ridx = 0; ridx < nrules; ++ridx) {
        auto [a, i3] = uleb128_decode_stream(data, i);
        i = i3;
        auto [b, i4] = uleb128_decode_stream(data, i);
        i = i4;
        rules[static_cast<int>(256 + ridx)] =
            { static_cast<int>(a), static_cast<int>(b) };
    }

    // 读取最终序列（每个符号都是 ULEB128，可为 >=256 的非终结符）
    auto [seq_len, i5] = uleb128_decode_stream(data, i);
    i = i5;
    std::vector<int> seq;
    seq.reserve(seq_len);
    for (std::size_t k = 0; k < seq_len; ++k) {
        auto [s, inext] = uleb128_decode_stream(data, i);
        i = inext;
        seq.push_back(static_cast<int>(s));
    }

    // 迭代+记忆化展开，避免重复/深递归
    std::unordered_map<int, std::vector<std::uint8_t>> memo;
    std::vector<std::uint8_t> out;
    out.reserve(orig_len);
    for (int s : seq) {
        const auto& chunk = expand_symbol(s, rules, memo);
        out.insert(out.end(), chunk.begin(), chunk.end());
    }
    if (out.size() != orig_len) {
        throw std::runtime_error("RePair output length mismatch");
    }
    return out;
}

// LFSR predictor: subtract pseudo‑random state from input bytes and encode the
// deltas as ULEB128.  The metadata map is unused here.
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_lfsr_predict(const std::vector<std::uint8_t>& block) {
    std::uint8_t state = 1;
    std::uint8_t taps = 0b10010110;
    std::vector<std::uint8_t> out;
    out.reserve(block.size());
    for (auto b : block) {
        std::uint8_t pred = state;
        std::uint8_t delta = static_cast<std::uint8_t>((b - pred) & 0xFFu);
        std::vector<std::uint8_t> enc = uleb128_encode(delta);
        out.insert(out.end(), enc.begin(), enc.end());
        // Update LFSR state.
        std::uint8_t fb = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if ((taps >> bit) & 1u) {
                fb ^= (state >> bit) & 1u;
            }
        }
        state = static_cast<std::uint8_t>((state << 1) | fb);
    }
    return {out, {}};
}

// Decode LFSR predictor encoded stream back into original bytes.  orig_len
// specifies the number of bytes to decode.
std::vector<std::uint8_t> decode_lfsr_predict(const std::vector<std::uint8_t>& data,
                                              std::size_t orig_len) {
    std::uint8_t state = 1;
    std::uint8_t taps = 0b10010110;
    std::vector<std::uint8_t> out;
    out.reserve(orig_len);
    std::size_t pos = 0;
    for (std::size_t k = 0; k < orig_len; ++k) {
        auto [delta, ni] = uleb128_decode_stream(data, pos);
        pos = ni;
        std::uint8_t b = static_cast<std::uint8_t>((delta + state) & 0xFFu);
        out.push_back(b);
        // Update LFSR state.
        std::uint8_t fb = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if ((taps >> bit) & 1u) {
                fb ^= (state >> bit) & 1u;
            }
        }
        state = static_cast<std::uint8_t>((state << 1) | fb);
    }
    return out;
}

// Encode a block using BBWT→MTF→Rice with optional bitwise transforms.  The
// flags field encodes which transforms were applied.  Meta stores flags,
// rice parameter k, and original lengths for use by the decoder.
std::pair<std::vector<std::uint8_t>, BBWTMeta>
encode_bbwt_mtf_rice(const std::vector<std::uint8_t>& block,
                     bool use_bitplane,
                     bool use_lfsr,
                     bool use_nibble,
                     bool use_bitrev,
                     bool use_gray,
                     std::uint8_t rice_param) {
    std::vector<std::uint8_t> bbwt = bbwt_forward(block);
    std::vector<std::uint8_t> mtf_list = mtf_encode(bbwt);
    std::vector<std::uint8_t> seq_bytes = mtf_list;
    // Apply optional bitwise transforms in order.
    if (use_bitplane) seq_bytes = bitplane_interleave(seq_bytes);
    if (use_lfsr)     seq_bytes = lfsr_whiten(seq_bytes);
    if (use_nibble)   seq_bytes = nibble_swap(seq_bytes);
    if (use_bitrev)   seq_bytes = bit_reverse(seq_bytes);
    if (use_gray)     seq_bytes = gray_encode_bytes(seq_bytes);
    // Rice encode the transformed bytes as a sequence of non‑negative ints.
    std::vector<std::uint64_t> seq64(seq_bytes.begin(), seq_bytes.end());
    std::vector<std::uint8_t> payload = rice_encode(seq64, rice_param);
    std::uint8_t flags = 0;
    if (use_bitplane) flags |= 1;
    if (use_lfsr)     flags |= 2;
    if (use_nibble)   flags |= 4;
    if (use_bitrev)   flags |= 8;
    if (use_gray)     flags |= 16;
    BBWTMeta meta{flags, rice_param, seq_bytes.size(), block.size()};
    return {payload, meta};
}

// Decode BBWT→MTF→Rice payload given meta information.  This function
// reconstructs the original block by inverting the optional transforms in
// reverse order, decoding Rice, reversing bitwise modules, MTF, and BWT.
std::vector<std::uint8_t> decode_bbwt_mtf_rice(const std::vector<std::uint8_t>& payload,
                                               const BBWTMeta& meta) {
    std::vector<std::uint64_t> seq = rice_decode(payload, meta.k, meta.length);
    std::vector<std::uint8_t> seq_bytes;
    seq_bytes.reserve(seq.size());
    for (auto n : seq) {
        seq_bytes.push_back(static_cast<std::uint8_t>(n & 0xFFu));
    }
    // Apply inverse transforms in reverse order of encoding.
    if (meta.flags & 16) seq_bytes = gray_decode_bytes(seq_bytes);
    if (meta.flags & 8)  seq_bytes = bit_reverse(seq_bytes);
    if (meta.flags & 4)  seq_bytes = nibble_swap(seq_bytes);
    if (meta.flags & 2)  seq_bytes = lfsr_whiten(seq_bytes); // whitening is self‑inverse
    if (meta.flags & 1)  seq_bytes = bitplane_deinterleave(seq_bytes, meta.length);
    // MTF decode and BBWT inverse.
    std::vector<std::uint8_t> mtf_list(seq_bytes.begin(), seq_bytes.end());
    std::vector<std::uint8_t> bbwt = mtf_decode(mtf_list);
    return bbwt_inverse(bbwt);
}

// Encode a block using the new V2 pipeline.  This pipeline selects a
// reversible automaton, then Gray encodes bytes, splits into bitplanes,
// optionally applies BBWT+RLE+Rice on each plane, and stores the
// necessary metadata in a header.  The returned meta map is unused but
// kept to mirror the Python signature.
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>>
encode_new_pipeline(const std::vector<std::uint8_t>& block) {
    // If block is empty, output a single flag byte (1) signalling raw.
    if (block.empty()) {
        return {std::vector<std::uint8_t>{1}, {}};
    }
    // Block‑level bypass: if the block has very high entropy (≈8) and
    // extremely short runs (≈1), store it raw.  This heuristic matches
    // Python's H0>=7.95 and avg_run<=1.05.
    auto byte_entropy = [](const std::vector<std::uint8_t>& b) {
        if (b.empty()) return 0.0;
        std::array<std::uint64_t, 256> cnt{};
        for (auto x : b) cnt[x]++;
        double n = static_cast<double>(b.size());
        double h = 0.0;
        for (auto c : cnt) {
            if (c == 0) continue;
            double p = static_cast<double>(c) / n;
            h -= p * std::log2(p);
        }
        return h;
    };
    auto avg_run_bytes = [](const std::vector<std::uint8_t>& b) {
        if (b.empty()) return 0.0;
        std::size_t runs = 1;
        std::uint8_t prev = b[0];
        for (std::size_t i = 1; i < b.size(); ++i) {
            if (b[i] != prev) {
                runs++;
                prev = b[i];
            }
        }
        return static_cast<double>(b.size()) / runs;
    };
    double H0 = byte_entropy(block);
    double avgR = avg_run_bytes(block);
    if (H0 >= 7.95 && avgR <= 1.05) {
        std::vector<std::uint8_t> out;
        out.reserve(1 + block.size());
        out.push_back(1);
        out.insert(out.end(), block.begin(), block.end());
        return {out, {}};
    }
    // Apply reversible automaton to bytes.
    AutomatonResult ar = circuit_map_automaton_forward(block);
    std::vector<std::uint8_t> mapped = ar.data;
    std::uint8_t mode = ar.mode;
    // Split into bitplanes.
    auto [planes, L] = bytes_to_bitplanes(mapped);
    std::vector<std::uint8_t> header;
    std::vector<std::uint8_t> payload;
    header.push_back(0);  // flag=COMP (0 means compressed)
    // Encode L as ULEB128.
    std::vector<std::uint8_t> Lenc = uleb128_encode(L);
    header.insert(header.end(), Lenc.begin(), Lenc.end());
    header.push_back(mode);  // store automaton mode
    // Process each bitplane.
    for (int j = 0; j < 8; ++j) {
        const auto& Uj = planes[j];
        // Simple bypass: if H0_bits>=0.99 and avg_run_bits<=1.02, store raw
        double h0 = H0_bits(Uj);
        double avg = avg_run_bits(Uj);
        if (h0 >= 0.99 && avg <= 1.02) {
            header.push_back(0x00);
            std::vector<std::uint8_t> packed = pack_bits_to_bytes(Uj);
            std::vector<std::uint8_t> enc_len = uleb128_encode(packed.size());
            header.insert(header.end(), enc_len.begin(), enc_len.end());
            payload.insert(payload.end(), packed.begin(), packed.end());
        } else {
            // Encode plane via BBWT -> RLE -> Rice.
            std::vector<std::uint8_t> Lj;
            Lj.reserve(Uj.size());
            // Convert bit vector to bytes for BWT (but each element is 0/1).
            for (int b : Uj) Lj.push_back(static_cast<std::uint8_t>(b));
            std::vector<std::uint8_t> Lj_bbwt = bbwt_forward(Lj);
            std::vector<int> Lj_bits;
            Lj_bits.reserve(Lj_bbwt.size());
            for (auto b : Lj_bbwt) Lj_bits.push_back(static_cast<int>(b));
            auto [b1, runs] = rle_binary(Lj_bits);
            double mean_r = runs.empty() ? 1.0 : (static_cast<double>(std::accumulate(runs.begin(), runs.end(), 0)) / runs.size());
            int k = (mean_r > 0 ? static_cast<int>(std::floor(std::log2(mean_r))) : 0);
            if (k < 0) k = 0;
            std::vector<std::uint8_t> rb = rice_encode(std::vector<std::uint64_t>(runs.begin(), runs.end()), static_cast<std::uint8_t>(k));
            // Encode metadata for this plane.
            header.push_back(0x01);
            header.push_back(static_cast<std::uint8_t>(b1 & 1));
            header.push_back(static_cast<std::uint8_t>(k & 0xFF));
            std::vector<std::uint8_t> rc_enc = uleb128_encode(runs.size());
            std::vector<std::uint8_t> rb_len_enc = uleb128_encode(rb.size());
            header.insert(header.end(), rc_enc.begin(), rc_enc.end());
            header.insert(header.end(), rb_len_enc.begin(), rb_len_enc.end());
            payload.insert(payload.end(), rb.begin(), rb.end());
        }
    }
    // Concatenate header and payload.
    std::vector<std::uint8_t> out = header;
    out.insert(out.end(), payload.begin(), payload.end());
    return {out, {}};
}

// Decode the V2 pipeline.  Inverse of encode_new_pipeline.  Parses header
// fields to reconstruct bitplanes, applies BBWT inverse, merges them and
// reverses the automaton.
std::vector<std::uint8_t> decode_new_pipeline(const std::vector<std::uint8_t>& payload,
                                              std::size_t orig_len,
                                              const std::unordered_map<std::string, std::string>& /*meta*/) {
    if (payload.empty()) return {};
    std::size_t pos = 0;
    std::uint8_t flag = payload[pos++];
    if (flag == 1) {
        // Raw block: return first orig_len bytes.
        if (payload.size() < pos + orig_len) {
            throw std::runtime_error("decode_new_pipeline: truncated raw block");
        }
        return std::vector<std::uint8_t>(payload.begin() + pos, payload.begin() + pos + orig_len);
    }
    // Read L
    auto [L, ni] = uleb128_decode_stream(payload, pos);
    pos = ni;
    std::uint8_t mode = payload[pos++];
    // Read plane descriptors
    struct PlaneDesc {
        // tag 0 raw: store nbytes; tag 1 enc: b1,k,run_count,paylen
        std::uint8_t tag;
        std::uint64_t nbytes_or_run_count;
        std::uint8_t b1;
        std::uint8_t k;
        std::uint64_t run_count;
        std::uint64_t paylen;
    };
    std::vector<PlaneDesc> descs;
    descs.reserve(8);
    for (int p = 0; p < 8; ++p) {
        if (pos >= payload.size()) throw std::runtime_error("decode_new_pipeline: truncated descriptor");
        std::uint8_t tag = payload[pos++];
        if (tag == 0x00) {
            auto [nbytes, nj] = uleb128_decode_stream(payload, pos);
            pos = nj;
            descs.push_back({tag, nbytes, 0, 0, 0, 0});
        } else if (tag == 0x01) {
            if (pos + 2 > payload.size()) throw std::runtime_error("decode_new_pipeline: truncated plane header");
            std::uint8_t b1 = payload[pos++];
            std::uint8_t k_val = payload[pos++];
            auto [run_count, nj] = uleb128_decode_stream(payload, pos);
            pos = nj;
            auto [paylen, nk] = uleb128_decode_stream(payload, pos);
            pos = nk;
            descs.push_back({tag, 0, b1, k_val, run_count, paylen});
        } else {
            throw std::runtime_error("decode_new_pipeline: unknown plane tag");
        }
    }
    // Parse payload for planes
    std::vector<std::vector<int>> planes;
    planes.reserve(8);
    std::size_t data_pos = pos;
    for (const auto& d : descs) {
        if (d.tag == 0x00) {
            std::uint64_t nbytes = d.nbytes_or_run_count;
            if (data_pos + nbytes > payload.size()) {
                throw std::runtime_error("decode_new_pipeline: raw plane truncated");
            }
            std::vector<std::uint8_t> buf(payload.begin() + data_pos,
                                          payload.begin() + data_pos + static_cast<std::size_t>(nbytes));
            data_pos += static_cast<std::size_t>(nbytes);
            planes.push_back(unpack_bits_from_bytes(buf, static_cast<std::size_t>(L)));
        } else {
            std::uint64_t run_count = d.run_count;
            std::uint64_t paylen = d.paylen;
            if (data_pos + paylen > payload.size()) {
                throw std::runtime_error("decode_new_pipeline: encoded plane truncated");
            }
            std::vector<std::uint8_t> rice_buf(payload.begin() + data_pos,
                                               payload.begin() + data_pos + static_cast<std::size_t>(paylen));
            data_pos += static_cast<std::size_t>(paylen);
            std::vector<std::uint64_t> runs64 = rice_decode(rice_buf, d.k, static_cast<std::size_t>(run_count));
            std::vector<int> runs;
            runs.reserve(runs64.size());
            for (auto r : runs64) runs.push_back(static_cast<int>(r));
            std::vector<int> Lj_bits = unrle_binary(d.b1, runs);
            // Convert to bytes and invert BBWT.
            std::vector<std::uint8_t> Uj_bytes;
            Uj_bytes.reserve(Lj_bits.size());
            for (auto b : Lj_bits) Uj_bytes.push_back(static_cast<std::uint8_t>(b));
            std::vector<std::uint8_t> Uj_bb = bbwt_inverse(Uj_bytes);
            if (Uj_bb.size() != L) {
                // Pad or trim if necessary.
                if (Uj_bb.size() > L) Uj_bb.resize(L);
                else Uj_bb.insert(Uj_bb.end(), L - Uj_bb.size(), 0);
            }
            std::vector<int> Uj;
            Uj.reserve(Uj_bb.size());
            for (auto b : Uj_bb) Uj.push_back(static_cast<int>(b));
            planes.push_back(Uj);
        }
    }
    // Reconstruct mapped block from bitplanes
    std::vector<std::uint8_t> mapped = bitplanes_to_bytes(planes);
    // Reverse automaton.
    std::vector<std::uint8_t> block = circuit_map_automaton_inverse(mapped, mode);
    return block;
}

// Top‑level compression: break input data into content‑defined blocks,
// evaluate candidate encoders on each block, and choose the smallest
// representation.  A simple container format is produced beginning with
// magic 'KOLR', followed by block size, total length, number of blocks,
// then for each block: method id, original length, payload length and the
// payload itself.  See Python for candidate definitions and ordering.
std::vector<std::uint8_t> compress_blocks(const std::vector<std::uint8_t>& data,
                                          std::size_t block_size) {
    // Magic header identifying format.
    std::vector<std::uint8_t> out;
    out.insert(out.end(), {'K','O','L','R'});
    // Store block size and total length as little‑endian 32‑bit values.
    auto write_le32 = [&](std::uint32_t val) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((val >> (8 * i)) & 0xFFu));
        }
    };
    write_le32(static_cast<std::uint32_t>(block_size));
    write_le32(static_cast<std::uint32_t>(data.size()));
    // Determine content defined boundaries.
    BoundaryList boundaries = cdc_fast_boundaries(data, 4096, block_size, 2 * block_size);
    // Write number of blocks as little‑endian 16‑bit.
    std::uint32_t nblocks = static_cast<std::uint16_t>(boundaries.size());
    out.push_back(static_cast<std::uint8_t>(nblocks & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((nblocks >> 8) & 0xFFu));
    // Candidate encoders ordered by expected cost.  Each encoder returns
    // payload bytes and any meta; we ignore meta except for those requiring
    // state to decode.  The order must align with decompress()'s decoder list.
    using Encoder = std::function<std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string,std::string>>(const std::vector<std::uint8_t>&)>;
    std::vector<Encoder> candidates = {
        // raw (id 0)
        [](const auto& b) { return std::make_pair(b, std::unordered_map<std::string,std::string>{}); },
        // xor delta coding (id 1)
        [](const auto& b) {
            std::vector<std::uint8_t> out;
            out.reserve(b.size());
            std::uint8_t prev = 0;
            for (auto x : b) {
                std::vector<std::uint8_t> enc = uleb128_encode((x - prev) & 0xFFu);
                out.insert(out.end(), enc.begin(), enc.end());
                prev = x;
            }
            return std::make_pair(out, std::unordered_map<std::string,std::string>{});
        },
        // bbwt base (no transforms)
        [](const auto& b) {
            auto [payload, meta] = encode_bbwt_mtf_rice(b, false, false, false, false, false, 2);
            return std::make_pair(payload, std::unordered_map<std::string,std::string>{});
        },
        // bbwt + bitplane
        [](const auto& b) {
            auto [payload, meta] = encode_bbwt_mtf_rice(b, true, false, false, false, false, 2);
            return std::make_pair(payload, std::unordered_map<std::string,std::string>{});
        },
        // bbwt + nibble
        [](const auto& b) {
            auto [payload, meta] = encode_bbwt_mtf_rice(b, false, false, true, false, false, 2);
            return std::make_pair(payload, std::unordered_map<std::string,std::string>{});
        },
        // bbwt + bit reverse
        [](const auto& b) {
            auto [payload, meta] = encode_bbwt_mtf_rice(b, false, false, false, true, false, 2);
            return std::make_pair(payload, std::unordered_map<std::string,std::string>{});
        },
        // bbwt + gray code
        [](const auto& b) {
            auto [payload, meta] = encode_bbwt_mtf_rice(b, false, false, false, false, true, 2);
            return std::make_pair(payload, std::unordered_map<std::string,std::string>{});
        },
        // lz77
        [](const auto& b) { return encode_lz77(b); },
        // lfsr predictor
        [](const auto& b) { return encode_lfsr_predict(b); },
        // repair
        [](const auto& b) { return repair_compress(b); },
        // new pipeline
        [](const auto& b) { return encode_new_pipeline(b); }
    };
    // For each block select smallest payload.
    for (const auto& [start, end] : boundaries) {
        std::vector<std::uint8_t> block(data.begin() + start, data.begin() + end);
        std::size_t best_size = std::numeric_limits<std::size_t>::max();
        std::size_t best_id = 0;
        std::vector<std::uint8_t> best_payload;
        for (std::size_t mid = 0; mid < candidates.size(); ++mid) {
            try {
                auto [payload, meta] = candidates[mid](block);
                if (payload.size() < best_size) {
                    best_size = payload.size();
                    best_id = mid;
                    best_payload = payload;
                }
            } catch (...) {
                // Ignore exceptions; skip this candidate.
            }
        }
        // Write method id, original block length, payload length and payload.
        out.push_back(static_cast<std::uint8_t>(best_id));
        write_le32(static_cast<std::uint32_t>(block.size()));
        write_le32(static_cast<std::uint32_t>(best_payload.size()));
        out.insert(out.end(), best_payload.begin(), best_payload.end());
    }
    return out;
}

// Decompress the container produced by compress_blocks.  Read magic, block
// size, total length and number of blocks then dispatch to the correct
// decoder.  The decoder list must align exactly with the candidate list in
// compress_blocks.  The meta map is unused (placeholder).
std::vector<std::uint8_t> decompress(const std::vector<std::uint8_t>& data) {
    std::size_t pos = 0;
    if (data.size() < 4 || std::string(data.begin(), data.begin() + 4) != "KOLR") {
        throw std::runtime_error("Invalid magic");
    }
    pos = 4;
    auto read_le32 = [&](std::size_t offset) {
        if (offset + 4 > data.size()) throw std::runtime_error("Truncated header");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(data[offset + i]) << (8 * i);
        }
        return v;
    };
    std::uint32_t block_size = read_le32(pos); pos += 4;
    std::uint32_t total_len = read_le32(pos); pos += 4;
    std::uint16_t nblocks = static_cast<std::uint16_t>(data[pos] | (data[pos+1] << 8));
    pos += 2;
    // Decoder functions align with candidate ordering.
    using Decoder = std::function<std::vector<std::uint8_t>(const std::vector<std::uint8_t>&, std::size_t, const std::unordered_map<std::string,std::string>&)>;
    std::vector<Decoder> decoders = {
        // raw
        [](const auto& p, std::size_t length, const auto&) {
            if (p.size() != length) throw std::runtime_error("raw decoder length mismatch");
            return p;
        },
        // xor delta
        [](const auto& p, std::size_t length, const auto&) {
            std::vector<std::uint8_t> out;
            out.reserve(length);
            std::size_t pos = 0;
            std::uint8_t prev = 0;
            for (std::size_t i = 0; i < length; ++i) {
                auto [delta, ni] = uleb128_decode_stream(p, pos);
                pos = ni;
                std::uint8_t b = static_cast<std::uint8_t>((prev + delta) & 0xFFu);
                out.push_back(b);
                prev = b;
            }
            return out;
        },
        // bbwt base
        [](const auto& p, std::size_t length, const auto&) {
            BBWTMeta meta{0, 2, length, length};
            return decode_bbwt_mtf_rice(p, meta);
        },
        // bbwt + bitplane
        [](const auto& p, std::size_t length, const auto&) {
            BBWTMeta meta{1, 2, length, length};
            return decode_bbwt_mtf_rice(p, meta);
        },
        // bbwt + nibble
        [](const auto& p, std::size_t length, const auto&) {
            BBWTMeta meta{4, 2, length, length};
            return decode_bbwt_mtf_rice(p, meta);
        },
        // bbwt + bit reverse
        [](const auto& p, std::size_t length, const auto&) {
            BBWTMeta meta{8, 2, length, length};
            return decode_bbwt_mtf_rice(p, meta);
        },
        // bbwt + gray
        [](const auto& p, std::size_t length, const auto&) {
            BBWTMeta meta{16, 2, length, length};
            return decode_bbwt_mtf_rice(p, meta);
        },
        // lz77
        [](const auto& p, std::size_t length, const auto&) {
            return decode_lz77(p, length);
        },
        // lfsr predictor
        [](const auto& p, std::size_t length, const auto&) {
            return decode_lfsr_predict(p, length);
        },
        // repair
        [](const auto& p, std::size_t length, const auto&) {
            return repair_decompress(p, length);
        },
        // new pipeline
        [](const auto& p, std::size_t length, const auto&) {
            return decode_new_pipeline(p, length, {});
        }
    };
    std::vector<std::uint8_t> out;
    out.reserve(total_len);
    for (std::size_t b = 0; b < nblocks; ++b) {
        if (pos >= data.size()) throw std::runtime_error("decompress: truncated block header");
        std::uint8_t method_id = data[pos++];
        std::uint32_t orig_len = read_le32(pos); pos += 4;
        std::uint32_t payload_len = read_le32(pos); pos += 4;
        if (pos + payload_len > data.size()) throw std::runtime_error("decompress: truncated payload");
        std::vector<std::uint8_t> payload(data.begin() + pos,
                                         data.begin() + pos + payload_len);
        pos += payload_len;
        if (method_id >= decoders.size()) throw std::runtime_error("decompress: unknown method id");
        std::vector<std::uint8_t> block = decoders[method_id](payload, orig_len, {});
        out.insert(out.end(), block.begin(), block.end());
    }
    if (out.size() != total_len) {
        throw std::runtime_error("decompress: output length mismatch");
    }
    return out;
}

void run_self_test() {
    using clock = std::chrono::high_resolution_clock;

    struct Dataset {
        std::string name;
        std::vector<std::uint8_t> data;
    };

    // ---------------------------
    // 1) 构造更丰富的数据集
    // ---------------------------
    std::vector<Dataset> datasets;

    // text_small：霍比特段落 ×10
    {
        std::string para =
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
            "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
            "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
            "hobbit-hole, and that means comfort.";
        std::string repeated;
        for (int i = 0; i < 10; ++i) repeated += para;
        datasets.push_back({"text", std::vector<std::uint8_t>(repeated.begin(), repeated.end())});
    }

    // text_large：同段落 ×200（长文本）
    {
        std::string para =
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
            "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
            "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
            "hobbit-hole, and that means comfort.";
        std::string repeated;
        for (int i = 0; i < 200; ++i) repeated += para;
        datasets.push_back({"text_big", std::vector<std::uint8_t>(repeated.begin(), repeated.end())});
    }

    // random_10k：伪随机
    {
        std::vector<std::uint8_t> rnd(10240);
        std::mt19937 rng(123456789);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto &b : rnd) b = static_cast<std::uint8_t>(dist(rng));
        datasets.push_back({"random", std::move(rnd)});
    }

    // repetitive：全 'a' 20 KiB
    {
        datasets.push_back({"repetitive", std::vector<std::uint8_t>(20480, (std::uint8_t)'a')});
    }

    // abab：'ab' × 10k
    {
        std::vector<std::uint8_t> v; v.reserve(20000);
        for (int i = 0; i < 10000; ++i) { v.push_back('a'); v.push_back('b'); }
        datasets.push_back({"abab", std::move(v)});
    }

    // abcabc：'abc' × 6000
    {
        std::vector<std::uint8_t> v; v.reserve(18000);
        for (int i = 0; i < 6000; ++i) { v.push_back('a'); v.push_back('b'); v.push_back('c'); }
        datasets.push_back({"abcabc", std::move(v)});
    }

    // zero：全 0，16 KiB（测试游程/预测）
    {
        datasets.push_back({"zero", std::vector<std::uint8_t>(16384, 0)});
    }

    // ramp：0..255 循环增长，8 KiB
    {
        std::vector<std::uint8_t> v(8192);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(i & 0xFF);
        datasets.push_back({"ramp", std::move(v)});
    }

    // utf8_mixed：中英混合 UTF-8，×200
    {
        std::string s = "数据压缩 data compression 可逆性 reversibility —— Kolmogorov-style.";
        std::string rep;
        for (int i = 0; i < 200; ++i) rep += s;
        datasets.push_back({"utf8_mixed", std::vector<std::uint8_t>(rep.begin(), rep.end())});
    }

    // ---------------------------
    // 2) 定义候选模型（与容器中的顺序严格一致）
    // ---------------------------
    const std::vector<std::string> method_names = {
        "Raw", "XOR",
        "BBWT", "BBWT+Bitplane", "BBWT+Nibble", "BBWT+BitRev", "BBWT+Gray",
        "LZ77", "LFSR predictor", "Re-Pair", "V2 New"
    };

    using Encoder = std::function<
        std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string,std::string>>
        (const std::vector<std::uint8_t>&)
    >;

    using Decoder = std::function<
        std::vector<std::uint8_t>(const std::vector<std::uint8_t>&, std::size_t,
                                  const std::unordered_map<std::string,std::string>&)
    >;

    std::vector<Encoder> encoders = {
        // 0 Raw
        [](const auto& b){ return std::make_pair(b, std::unordered_map<std::string,std::string>{}); },
        // 1 XOR
        [](const auto& b){
            std::vector<std::uint8_t> out; out.reserve(b.size());
            std::uint8_t prev = 0;
            for (auto x : b) {
                auto enc = uleb128_encode((x - prev) & 0xFFu);
                out.insert(out.end(), enc.begin(), enc.end());
                prev = x;
            }
            return std::make_pair(out, std::unordered_map<std::string,std::string>{});
        },
        // 2~6 BBWT variants
        [](const auto& b){
            auto [p,_] = encode_bbwt_mtf_rice(b,false,false,false,false,false,2);
            return std::make_pair(std::move(p), std::unordered_map<std::string,std::string>{});
        },
        [](const auto& b){
            auto [p,_] = encode_bbwt_mtf_rice(b,true ,false,false,false,false,2);
            return std::make_pair(std::move(p), std::unordered_map<std::string,std::string>{});
        },
        [](const auto& b){
            auto [p,_] = encode_bbwt_mtf_rice(b,false,false,true ,false,false,2);
            return std::make_pair(std::move(p), std::unordered_map<std::string,std::string>{});
        },
        [](const auto& b){
            auto [p,_] = encode_bbwt_mtf_rice(b,false,false,false,true ,false,2);
            return std::make_pair(std::move(p), std::unordered_map<std::string,std::string>{});
        },
        [](const auto& b){
            auto [p,_] = encode_bbwt_mtf_rice(b,false,false,false,false,true ,2);
            return std::make_pair(std::move(p), std::unordered_map<std::string,std::string>{});
        },
        // 7 LZ77
        [](const auto& b){ return encode_lz77(b); },
        // 8 LFSR predictor
        [](const auto& b){ return encode_lfsr_predict(b); },
        // 9 Re-Pair (严格版 RP / ULEB128)
        [](const auto& b){ return repair_compress(b); },
        // 10 V2 New
        [](const auto& b){ return encode_new_pipeline(b); }
    };

    std::vector<Decoder> decoders = {
        // 0 Raw
        [](const auto& p, std::size_t length, const auto&){
            if (p.size() != length) throw std::runtime_error("raw decoder length mismatch");
            return p;
        },
        // 1 XOR
        [](const auto& p, std::size_t length, const auto&){
            std::vector<std::uint8_t> out; out.reserve(length);
            std::size_t pos = 0; std::uint8_t prev = 0;
            for (std::size_t i = 0; i < length; ++i) {
                auto [delta, ni] = uleb128_decode_stream(p, pos);
                pos = ni;
                std::uint8_t b = static_cast<std::uint8_t>((prev + delta) & 0xFFu);
                out.push_back(b); prev = b;
            }
            return out;
        },
        // 2~6 BBWT variants（flag 0/1/4/8/16，Rice k=2）
        [](const auto& p, std::size_t len, const auto&){ BBWTMeta m{0 ,2,len,len}; return decode_bbwt_mtf_rice(p, m); },
        [](const auto& p, std::size_t len, const auto&){ BBWTMeta m{1 ,2,len,len}; return decode_bbwt_mtf_rice(p, m); },
        [](const auto& p, std::size_t len, const auto&){ BBWTMeta m{4 ,2,len,len}; return decode_bbwt_mtf_rice(p, m); },
        [](const auto& p, std::size_t len, const auto&){ BBWTMeta m{8 ,2,len,len}; return decode_bbwt_mtf_rice(p, m); },
        [](const auto& p, std::size_t len, const auto&){ BBWTMeta m{16,2,len,len}; return decode_bbwt_mtf_rice(p, m); },
        // 7 LZ77
        [](const auto& p, std::size_t len, const auto&){ return decode_lz77(p, len); },
        // 8 LFSR predictor
        [](const auto& p, std::size_t len, const auto&){ return decode_lfsr_predict(p, len); },
        // 9 Re-Pair (严格 RP)
        [](const auto& p, std::size_t len, const auto&){ return repair_decompress(p, len); },
        // 10 V2 New
        [](const auto& p, std::size_t len, const auto&){ return decode_new_pipeline(p, len, {}); }
    };

    // ---------------------------
    // 3) 打印对齐图表风格的表格：每个数据集 × 每个方法
    // ---------------------------
    auto print_header = [](){
        std::cout << std::left  << std::setw(12) << "Dataset"
                  << std::left  << std::setw(16) << "Method"
                  << std::right << std::setw(12) << "OrigBytes"
                  << std::setw(14) << "CompBytes"
                  << std::setw(10) << "Ratio"
                  << std::setw(14) << "Comp(ms)"
                  << std::setw(14) << "Decomp(ms)"
                  << std::setw(12) << "OK/ERR"
                  << "\n";
        std::cout << std::string(114, '-') << "\n";
    };

    print_header();

    // 记录每个数据集最佳方案（按 ratio）
    struct Best { double ratio = 1e100; std::string method; std::size_t size=0; double c=0, d=0; };
    std::unordered_map<std::string, Best> best_of;

    for (const auto &ds : datasets) {
        const auto& orig = ds.data;
        const std::size_t orig_size = orig.size();

        for (std::size_t mid = 0; mid < encoders.size(); ++mid) {
            std::string ok = "OK";
            std::size_t comp_size = 0;
            double comp_ms = 0.0, decomp_ms = 0.0, ratio = 0.0;

            try {
                auto t0 = clock::now();
                auto [payload, meta] = encoders[mid](orig);
                auto t1 = clock::now();

                comp_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
                comp_size = payload.size();
                ratio = (orig_size == 0) ? 1.0 : (double)comp_size / (double)orig_size;

                t0 = clock::now();
                auto rec = decoders[mid](payload, orig_size, meta);
                t1 = clock::now();
                decomp_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

                if (rec != orig) {
                    ok = "FAIL";
                }
            } catch (const std::exception& ex) {
                ok = "ERR";
                ratio = std::numeric_limits<double>::infinity();
            }

            std::cout << std::left  << std::setw(12) << ds.name
                      << std::left  << std::setw(16) << method_names[mid]
                      << std::right << std::setw(12) << orig_size
                      << std::setw(14) << comp_size
                      << std::setw(10) << std::fixed << std::setprecision(3) << ratio
                      << std::setw(14) << std::fixed << std::setprecision(3) << comp_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << decomp_ms
                      << std::setw(12) << ok
                      << "\n";

            if (ok == "OK" && ratio < best_of[ds.name].ratio) {
                best_of[ds.name] = {ratio, method_names[mid], comp_size, comp_ms, decomp_ms};
            }
        }
    }

    std::cout << std::string(114, '-') << "\n";
    std::cout << "Best method per dataset (by ratio):\n";
    for (const auto& kv : best_of) {
        const auto& b = kv.second;
        std::cout << "  " << std::left << std::setw(12) << kv.first
                  << " -> " << std::left << std::setw(16) << b.method
                  << " size=" << b.size
                  << " ratio=" << std::fixed << std::setprecision(3) << b.ratio
                  << " comp(ms)=" << std::fixed << std::setprecision(3) << b.c
                  << " decomp(ms)=" << std::fixed << std::setprecision(3) << b.d
                  << "\n";
    }
    std::cout << std::string(114, '-') << "\n";
    std::cout << "Self-test completed.\n";
}


// -----------------------------------------------------------------------------
// Command line interface
//
// When compiled as a standalone program, this main function allows the user
// to compress or decompress files from disk.  The behaviour mirrors the
// Python CLI: supply an input file; use -d/--decompress to decompress; use
// -o to specify an output file; use -b for custom block size.  The
// experiment option is not implemented here.
#define BUILD_KOLM_MAIN

#ifdef BUILD_KOLM_MAIN
int main(int argc, char** argv) {
    // If no command line arguments are provided (argc == 1), run the
    // built‑in self test.  Because main() returns int, we explicitly
    // return 0 after invoking run_self_test().
    if (argc == 1) {
        run_self_test();
        return 0;
    }
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [options] input_file\n";
        std::cerr << "Options:\n";
        std::cerr << "  -d, --decompress   Decompress input\n";
        std::cerr << "  -o <file>          Output file\n";
        std::cerr << "  -b <size>          Block size (default 8192)\n";
        return 1;
    }
    bool decompress_flag = false;
    std::string input_name;
    std::string output_name;
    std::size_t block_size = 8192;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" || arg == "--decompress") {
            decompress_flag = true;
        } else if (arg == "-o" && i + 1 < argc) {
            output_name = argv[++i];
        } else if (arg == "-b" && i + 1 < argc) {
            block_size = std::stoul(argv[++i]);
        } else if (arg.rfind("-", 0) == 0) {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        } else {
            input_name = arg;
        }
    }
    if (input_name.empty()) {
        std::cerr << "No input file specified\n";
        return 1;
    }
    // Read entire input file
    std::vector<std::uint8_t> data;
    {
        std::FILE* f = std::fopen(input_name.c_str(), "rb");
        if (!f) {
            std::cerr << "Failed to open input file\n";
            return 1;
        }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz < 0) sz = 0;
        data.resize(static_cast<std::size_t>(sz));
        std::fread(data.data(), 1, data.size(), f);
        std::fclose(f);
    }
    if (decompress_flag) {
        std::vector<std::uint8_t> out = decompress(data);
        std::string out_name = !output_name.empty() ? output_name : (input_name + ".out");
        std::FILE* f = std::fopen(out_name.c_str(), "wb");
        if (!f) {
            std::cerr << "Failed to open output file\n";
            return 1;
        }
        std::fwrite(out.data(), 1, out.size(), f);
        std::fclose(f);
        std::cout << "Decompressed " << data.size() << " bytes to " << out.size() << " bytes -> " << out_name << "\n";
    } else {
        std::vector<std::uint8_t> blob = compress_blocks(data, block_size);
        std::string out_name = !output_name.empty() ? output_name : (input_name + ".kolr");
        std::FILE* f = std::fopen(out_name.c_str(), "wb");
        if (!f) {
            std::cerr << "Failed to open output file\n";
            return 1;
        }
        std::fwrite(blob.data(), 1, blob.size(), f);
        std::fclose(f);
        double ratio = data.empty() ? 1.0 : static_cast<double>(blob.size()) / data.size();
        std::cout << "Compressed " << data.size() << " bytes to " << blob.size() << " bytes (ratio " << ratio << ") -> " << out_name << "\n";
    }
    return 0;
}
#endif // BUILD_KOLM_MAIN
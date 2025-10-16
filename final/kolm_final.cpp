// kolm_final.cpp
//
// High‑performance C++ implementation of the "Kolm" compression
// pipeline inspired by Kolmogorov/MDL principles.  The compressor
// operates on blocks of bytes and evaluates several simple models on
// each block, selecting the model whose description (model
// parameters plus encoded payload) is shortest.  Models are kept
// intentionally small and bit‑friendly, relying on primitive
// operations (XOR, run length, small dictionary) rather than heavy
// statistical context mixing.  A content‑defined chunking routine
// provides boundaries that align with data structure.  The resulting
// container format is self‑describing and supports streaming decode.
//
// This implementation is designed for C++20 and aims to balance
// readability with performance.  It demonstrates how the design
// philosophy articulated in the Python prototype can be realized in
// a compiled language: short code paths, explicit bit packing, and
// minimal dynamic allocations.  The algorithm still trades some
// compression ratio for speed when compared to highly tuned
// compressors like zlib or Zstd, but it showcases how small,
// transparent models can achieve respectable compression without
// complex machinery.

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// ULEB128 encoding/decoding of unsigned integers.
// --------------------------------------------------------------------------

static inline std::vector<uint8_t> uleb128_encode(uint32_t value) {
    std::vector<uint8_t> out;
    while (true) {
        uint8_t byte = static_cast<uint8_t>(value & 0x7Fu);
        value >>= 7;
        if (value == 0) {
            out.push_back(byte);
            break;
        } else {
            out.push_back(static_cast<uint8_t>(byte | 0x80u));
        }
    }
    return out;
}

static inline uint32_t uleb128_decode(const std::vector<uint8_t> &data, size_t &pos) {
    uint32_t result = 0;
    uint32_t shift = 0;
    while (pos < data.size()) {
        uint8_t byte = data[pos++];
        result |= static_cast<uint32_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) break;
        shift += 7;
    }
    return result;
}

// --------------------------------------------------------------------------
// Simple Rice/Golomb encoding and decoding for non‑negative integers.  The
// quotient is encoded in unary (q bits of '1' followed by a '0'); the
// remainder is encoded in k fixed bits.  These routines pack bits
// densely into bytes.
// --------------------------------------------------------------------------

static std::vector<uint8_t> rice_encode(const std::vector<uint32_t> &values, unsigned k) {
    std::vector<uint8_t> out;
    out.reserve(values.size() * 2); // reserve some space
    uint8_t current_byte = 0;
    int bit_count = 0;
    for (uint32_t v : values) {
        uint32_t q = v >> k;
        uint32_t r = v & ((1u << k) - 1u);
        // emit q times '1' bits
        for (uint32_t i = 0; i < q; ++i) {
            current_byte = (current_byte << 1) | 1u;
            bit_count++;
            if (bit_count == 8) {
                out.push_back(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }
        // emit a terminating '0'
        current_byte <<= 1;
        bit_count++;
        if (bit_count == 8) {
            out.push_back(current_byte);
            current_byte = 0;
            bit_count = 0;
        }
        // emit k bits of remainder (MSB first)
        for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
            current_byte = (current_byte << 1) | static_cast<uint8_t>((r >> i) & 1u);
            bit_count++;
            if (bit_count == 8) {
                out.push_back(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }
    }
    // flush any remaining bits (pad with zeros)
    if (bit_count > 0) {
        current_byte <<= (8 - bit_count);
        out.push_back(current_byte);
    }
    return out;
}

static std::vector<uint32_t> rice_decode(const std::vector<uint8_t> &data, unsigned k, size_t &bit_pos, size_t count) {
    std::vector<uint32_t> out;
    out.reserve(count);
    size_t total_bits = data.size() * 8;
    while (out.size() < count && bit_pos < total_bits) {
        // read unary quotient
        uint32_t q = 0;
        while (bit_pos < total_bits) {
            uint8_t byte = data[bit_pos / 8];
            int bit_index = 7 - static_cast<int>(bit_pos % 8);
            uint8_t bit = (byte >> bit_index) & 1u;
            bit_pos++;
            if (bit == 1u) {
                ++q;
            } else {
                break; // stop on zero delimiter
            }
        }
        // read k bits for remainder
        uint32_t r = 0;
        for (unsigned i = 0; i < k; ++i) {
            if (bit_pos >= total_bits) break;
            uint8_t byte = data[bit_pos / 8];
            int bit_index = 7 - static_cast<int>(bit_pos % 8);
            uint8_t bit = (byte >> bit_index) & 1u;
            bit_pos++;
            r = (r << 1) | bit;
        }
        out.push_back((q << k) | r);
    }
    return out;
}

// --------------------------------------------------------------------------
// A naive Burrows–Wheeler transform (BWT) using a sentinel symbol 0x00.
// The sentinel must not appear within the input block.  The transform
// returns a pair consisting of the last column of the sorted rotations
// and the primary index (the row which corresponds to the original
// string).  Inverse transform reconstructs the original block by
// repeatedly applying the LF-mapping.  This routine has O(n^2 log n)
// complexity and is intended for moderate block sizes (≤64 KiB).
// --------------------------------------------------------------------------

static std::pair<std::vector<uint8_t>, uint32_t> bwt_transform(const std::vector<uint8_t> &input) {
    // append sentinel (0)
    std::vector<uint8_t> data = input;
    data.push_back(0);
    const size_t n = data.size();
    // generate vector of rotation offsets
    std::vector<uint32_t> idx(n);
    for (uint32_t i = 0; i < n; ++i) idx[i] = i;
    // comparator for rotations
    auto cmp = [&data, n](uint32_t a, uint32_t b) {
        if (a == b) return false;
        for (size_t i = 0; i < n; ++i) {
            uint8_t ca = data[(a + i) % n];
            uint8_t cb = data[(b + i) % n];
            if (ca < cb) return true;
            if (ca > cb) return false;
        }
        return false;
    };
    std::sort(idx.begin(), idx.end(), cmp);
    std::vector<uint8_t> last_column(n);
    uint32_t primary = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t rot_start = idx[i];
        if (rot_start == 0) primary = static_cast<uint32_t>(i);
        // last column value is the byte preceding the rotation start (with wrap)
        size_t j = (rot_start + n - 1) % n;
        last_column[i] = data[j];
    }
    return {last_column, primary};
}

static std::vector<uint8_t> bwt_inverse(const std::vector<uint8_t> &last_column, uint32_t primary) {
    // The inverse BWT constructs the original string by applying the
    // LF-mapping.  We build a table of counts to resolve ties and then
    // iterate to rebuild the string backwards.
    size_t n = last_column.size();
    // Compute frequency and assign ranks within each character
    std::array<uint32_t, 256> count{};
    std::array<uint32_t, 256> totals{};
    std::vector<uint32_t> ranks(n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t c = last_column[i];
        ranks[i] = count[c]++;
    }
    // compute cumulative totals to map each (char, rank) → position in first column
    uint32_t sum = 0;
    for (size_t c = 0; c < 256; ++c) {
        totals[c] = sum;
        sum += count[c];
    }
    // reconstruct original by stepping through LF-mapping
    std::vector<uint8_t> out;
    out.resize(n);
    uint32_t idx = primary;
    for (size_t i = 0; i < n; ++i) {
        uint8_t c = last_column[idx];
        out[n - 1 - i] = c;
        idx = totals[c] + ranks[idx];
    }
    // strip sentinel (last byte should be 0)
    if (!out.empty() && out.back() == 0) {
        out.pop_back();
    }
    return out;
}

// --------------------------------------------------------------------------
// Move‑To‑Front encoding and decoding.  Maintains a dynamic list of
// 256 symbols; each input byte is replaced by its index and moved
// to the front of the list.  Decoding performs the inverse mapping.
// --------------------------------------------------------------------------

static std::vector<uint8_t> mtf_encode(const std::vector<uint8_t> &input) {
    std::vector<uint8_t> table(256);
    for (int i = 0; i < 256; ++i) table[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> out;
    out.reserve(input.size());
    for (uint8_t b : input) {
        // find index
        uint8_t idx = 0;
        for (; idx < 256; ++idx) {
            if (table[idx] == b) break;
        }
        out.push_back(idx);
        // move to front
        uint8_t saved = table[idx];
        for (int j = idx; j > 0; --j) table[j] = table[j - 1];
        table[0] = saved;
    }
    return out;
}

static std::vector<uint8_t> mtf_decode(const std::vector<uint8_t> &input) {
    std::vector<uint8_t> table(256);
    for (int i = 0; i < 256; ++i) table[i] = static_cast<uint8_t>(i);
    std::vector<uint8_t> out;
    out.reserve(input.size());
    for (uint8_t idx : input) {
        uint8_t b = table[idx];
        out.push_back(b);
        // move to front
        for (int j = idx; j > 0; --j) table[j] = table[j - 1];
        table[0] = b;
    }
    return out;
}

// --------------------------------------------------------------------------
// XOR predictor model: encode residuals as ULEB128.
// --------------------------------------------------------------------------
static std::vector<uint8_t> encode_xor(const std::vector<uint8_t> &block) {
    std::vector<uint8_t> payload;
    payload.reserve(block.size());
    uint8_t prev = 0;
    for (uint8_t b : block) {
        uint8_t diff = b ^ prev;
        prev = b;
        auto enc = uleb128_encode(diff);
        payload.insert(payload.end(), enc.begin(), enc.end());
    }
    return payload;
}

static std::vector<uint8_t> decode_xor(const std::vector<uint8_t> &payload, size_t orig_len) {
    std::vector<uint8_t> block;
    block.reserve(orig_len);
    size_t pos = 0;
    uint8_t prev = 0;
    while (block.size() < orig_len && pos < payload.size()) {
        uint8_t b = uleb128_decode(payload, pos);
        uint8_t val = b ^ prev;
        block.push_back(val);
        prev = val;
    }
    return block;
}

// --------------------------------------------------------------------------
// Naive LZ77 encoder/decoder.  Matches of length ≥3 are encoded as
// (flag=1, length, distance) with ULEB128; literals are (flag=0, byte).
// --------------------------------------------------------------------------
static std::vector<uint8_t> encode_lz77(const std::vector<uint8_t> &block) {
    const size_t n = block.size();
    size_t i = 0;
    std::vector<uint8_t> out;
    out.reserve(n);
    const size_t window_size = 255;
    const size_t lookahead_limit = 127;
    while (i < n) {
        size_t best_len = 0;
        size_t best_dist = 0;
        // search for longest match within window
        size_t win_start = (i > window_size ? i - window_size : 0);
        for (size_t dist = 1; dist <= i - win_start; ++dist) {
            size_t j = i - dist;
            size_t length = 0;
            while (length < lookahead_limit && i + length < n && block[j + length] == block[i + length]) {
                ++length;
            }
            if (length >= 3 && length > best_len) {
                best_len = length;
                best_dist = dist;
                if (best_len == lookahead_limit) break;
            }
        }
        if (best_len >= 3) {
            out.push_back(1);
            auto enc_len = uleb128_encode(static_cast<uint32_t>(best_len));
            out.insert(out.end(), enc_len.begin(), enc_len.end());
            auto enc_dist = uleb128_encode(static_cast<uint32_t>(best_dist));
            out.insert(out.end(), enc_dist.begin(), enc_dist.end());
            i += best_len;
        } else {
            out.push_back(0);
            out.push_back(block[i]);
            ++i;
        }
    }
    return out;
}

static std::vector<uint8_t> decode_lz77(const std::vector<uint8_t> &payload, size_t orig_len) {
    std::vector<uint8_t> out;
    out.reserve(orig_len);
    size_t pos = 0;
    const size_t n = payload.size();
    while (pos < n && out.size() < orig_len) {
        uint8_t flag = payload[pos++];
        if (flag == 0) {
            if (pos < n) {
                out.push_back(payload[pos++]);
            }
        } else if (flag == 1) {
            uint32_t length = uleb128_decode(payload, pos);
            uint32_t dist = uleb128_decode(payload, pos);
            size_t copy_start = out.size() - dist;
            for (uint32_t k = 0; k < length; ++k) {
                out.push_back(out[copy_start + k]);
            }
        } else {
            // unknown flag, abort
            break;
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Model 2: BWT→MTF→Rice.  For each block, compute the BWT with
// sentinel, then apply MTF.  Separate the resulting sequence into
// zero runs and (value-1) nonzeros.  Each of these two integer
// sequences is Rice‑coded with a small parameter k chosen to
// minimize the bit length.  The ordering of zeros and nonzeros is
// recorded in a tag array of equal length to the MTF sequence.  The
// encoded payload layout is:
//   [k_zero][k_nonzero][len_zero_runs][len_nonzeros][encoded_zero_runs][encoded_nonzeros][tag_bytes]
// where k_zero and k_nonzero are single bytes storing the Rice parameter;
// lengths are ULEB128; and tag_bytes is a packed array of bits (8 tags per byte) with 0 for zero run and 1 for nonzero.  The tag bits identify which sequence to consume next when reconstructing the MTF values.
// --------------------------------------------------------------------------

static std::vector<uint8_t> encode_model_bwt_mtf_rice(const std::vector<uint8_t> &block) {
    // BWT
    auto bwt_res = bwt_transform(block);
    auto &last = bwt_res.first;
    uint32_t primary = bwt_res.second;
    // MTF
    std::vector<uint8_t> mtf = mtf_encode(last);
    // separate zero runs and nonzero values
    std::vector<uint32_t> zero_runs;
    std::vector<uint32_t> nonzeros;
    std::vector<uint8_t> tags; // tag bits (0→zero run, 1→nonzero)
    tags.reserve((mtf.size() + 7) / 8);
    std::vector<uint8_t> tag_bits;
    tag_bits.reserve(mtf.size());
    size_t i = 0;
    while (i < mtf.size()) {
        if (mtf[i] == 0) {
            // count zero run
            size_t j = i;
            while (j < mtf.size() && mtf[j] == 0) ++j;
            zero_runs.push_back(static_cast<uint32_t>(j - i));
            // record tags: one tag per run occurrence
            for (size_t r = 0; r < (j - i); ++r) {
                tag_bits.push_back(0);
            }
            i = j;
        } else {
            nonzeros.push_back(static_cast<uint32_t>(mtf[i] - 1));
            tag_bits.push_back(1);
            ++i;
        }
    }
    // choose Rice parameter k for zero_runs
    auto choose_k = [](const std::vector<uint32_t> &vals) {
        if (vals.empty()) return 0u;
        unsigned best_k = 0;
        size_t best_bits = static_cast<size_t>(-1);
        // restrict k up to 7 bits for reasonable code lengths
        for (unsigned k = 0; k <= 7; ++k) {
            size_t bits = 0;
            for (uint32_t v : vals) {
                bits += (v >> k) + 1 + k;
            }
            if (bits < best_bits) {
                best_bits = bits;
                best_k = k;
            }
        }
        return best_k;
    };
    unsigned k_zero = choose_k(zero_runs);
    unsigned k_nonzero = choose_k(nonzeros);
    // encode zero_runs and nonzeros
    std::vector<uint8_t> zr_encoded = rice_encode(zero_runs, k_zero);
    std::vector<uint8_t> nz_encoded = rice_encode(nonzeros, k_nonzero);
    // pack tags into bytes
    std::vector<uint8_t> tag_bytes;
    tag_bytes.reserve((tag_bits.size() + 7) / 8);
    uint8_t accum = 0;
    int bitpos = 0;
    for (uint8_t t : tag_bits) {
        accum = (accum << 1) | (t & 1);
        ++bitpos;
        if (bitpos == 8) {
            tag_bytes.push_back(accum);
            accum = 0;
            bitpos = 0;
        }
    }
    if (bitpos > 0) {
        accum <<= (8 - bitpos);
        tag_bytes.push_back(accum);
    }
    // build output: k_zero, k_nonzero, primary index, lengths, encoded runs
    std::vector<uint8_t> out;
    out.reserve(2 + 5 + zr_encoded.size() + nz_encoded.size() + tag_bytes.size());
    out.push_back(static_cast<uint8_t>(k_zero));
    out.push_back(static_cast<uint8_t>(k_nonzero));
    // encode primary index as ULEB128
    auto enc_primary = uleb128_encode(primary);
    out.insert(out.end(), enc_primary.begin(), enc_primary.end());
    // encode lengths of zero_runs and nonzeros as ULEB128
    auto enc_z_len = uleb128_encode(static_cast<uint32_t>(zero_runs.size()));
    out.insert(out.end(), enc_z_len.begin(), enc_z_len.end());
    auto enc_n_len = uleb128_encode(static_cast<uint32_t>(nonzeros.size()));
    out.insert(out.end(), enc_n_len.begin(), enc_n_len.end());
    // append encoded arrays
    out.insert(out.end(), zr_encoded.begin(), zr_encoded.end());
    out.insert(out.end(), nz_encoded.begin(), nz_encoded.end());
    // append tag bytes
    out.insert(out.end(), tag_bytes.begin(), tag_bytes.end());
    return out;
}

static std::vector<uint8_t> decode_model_bwt_mtf_rice(const std::vector<uint8_t> &payload, size_t orig_len) {
    // decode k_zero and k_nonzero
    size_t pos = 0;
    unsigned k_zero = payload[pos++];
    unsigned k_nonzero = payload[pos++];
    // decode primary index
    uint32_t primary = uleb128_decode(payload, pos);
    // decode lengths
    uint32_t len_z = uleb128_decode(payload, pos);
    uint32_t len_n = uleb128_decode(payload, pos);
    // decode zero_runs and nonzeros
    // payload after pos contains concatenated Rice codes for zero_runs and nonzeros followed by tag bytes.
    size_t bit_pos = 0;
    std::vector<uint8_t> zr_nz_bytes;
    // find where tags start: we know number of tags = orig_len (with sentinel) but tags correspond to MTF sequence length = block_size+1.
    // We can't easily know tags offset; instead we decode exactly len_z and len_n values using bit positions on the fly.
    // The Rice decode function uses bit_pos to track progress; we just supply the same payload and maintain bit_pos.
    // After decoding both sequences, bit_pos will be at beginning of tag bits.
    std::vector<uint32_t> zero_runs = rice_decode(payload, k_zero, bit_pos, len_z);
    std::vector<uint32_t> nonzeros = rice_decode(payload, k_nonzero, bit_pos, len_n);
    // compute how many tags are needed: MTF sequence length = orig_len + 1 (include sentinel)
    size_t tag_count = orig_len + 1;
    std::vector<uint8_t> tags;
    tags.reserve(tag_count);
    size_t bytes_len = payload.size();
    while (tags.size() < tag_count && bit_pos < bytes_len * 8) {
        uint8_t byte = payload[bit_pos / 8];
        int bit_index = 7 - static_cast<int>(bit_pos % 8);
        uint8_t bit = (byte >> bit_index) & 1u;
        tags.push_back(bit);
        ++bit_pos;
    }
    // reconstruct MTF sequence from tags and run arrays
    std::vector<uint8_t> mtf_seq;
    mtf_seq.reserve(tag_count);
    size_t zi = 0, ni = 0;
    for (uint8_t tag : tags) {
        if (tag == 0) {
            if (zi < zero_runs.size()) {
                uint32_t run = zero_runs[zi++];
                for (uint32_t r = 0; r < run; ++r) mtf_seq.push_back(0);
            }
        } else {
            if (ni < nonzeros.size()) {
                mtf_seq.push_back(static_cast<uint8_t>(nonzeros[ni++] + 1));
            }
        }
    }
    // decode MTF → last column
    std::vector<uint8_t> last = mtf_decode(mtf_seq);
    // inverse BWT
    return bwt_inverse(last, primary);
}

// --------------------------------------------------------------------------
// Content defined chunking via a simple gear hash.  The mask size is
// derived from the desired average block size.  If no cutpoint is
// found within [min_size, max_size], a forced cut is performed at
// max_size.
// --------------------------------------------------------------------------
static std::vector<std::pair<size_t, size_t>> cdc_boundaries(const std::vector<uint8_t> &data, size_t min_size, size_t avg_size, size_t max_size) {
    std::vector<std::pair<size_t, size_t>> bounds;
    const size_t n = data.size();
    if (n == 0) return bounds;
    // generate a deterministic gear table
    static std::array<uint32_t, 256> gear_table;
    static bool gear_init = false;
    if (!gear_init) {
        std::mt19937 rng(2025);
        for (auto &x : gear_table) x = rng();
        gear_init = true;
    }
    unsigned k = std::max<size_t>(6, std::min<size_t>(20, static_cast<size_t>(std::bit_width(avg_size))));
    uint32_t mask = (1u << k) - 1u;
    size_t i = 0;
    while (i < n) {
        size_t start = i;
        uint32_t h = 0;
        size_t end_min = std::min(n, start + min_size);
        size_t end_max = std::min(n, start + max_size);
        i = end_min;
        bool cut_found = false;
        while (i < end_max) {
            h = (h << 1) + gear_table[data[i]];
            if ((h & mask) == 0) {
                ++i;
                cut_found = true;
                break;
            }
            ++i;
        }
        // if no cutpoint found, cut at max_size
        bounds.emplace_back(start, i);
    }
    return bounds;
}

// --------------------------------------------------------------------------
// Compressor driver: compresses a vector of bytes using block
// modelling.  The output format begins with 'KOLM' magic, block
// parameters, and the number of blocks.  Each block stores: method
// identifier, original length, payload length, and payload.  Method
// identifiers: 0=RAW, 1=XOR, 2=BBWT→MTF→Rice, 3=LZ77.  Raw always
// produces a payload equal to the block.  In high entropy blocks the
// encoder may skip expensive models to save time.
// --------------------------------------------------------------------------
static std::vector<uint8_t> compress_data(const std::vector<uint8_t> &input) {
    // choose block parameters
    const size_t min_size = 4096;
    const size_t avg_size = 8192;
    const size_t max_size = 16384;
    // compute CDC boundaries
    auto blocks = cdc_boundaries(input, min_size, avg_size, max_size);
    // output buffer
    std::vector<uint8_t> out;
    // header: magic
    out.insert(out.end(), {'K','O','L','M'});
    // encode block size (avg size) as u32 little endian
    uint32_t block_sz = static_cast<uint32_t>(avg_size);
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&block_sz), reinterpret_cast<uint8_t*>(&block_sz) + 4);
    // total length as u64 little endian
    uint64_t total_len = static_cast<uint64_t>(input.size());
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&total_len), reinterpret_cast<uint8_t*>(&total_len) + 8);
    // number of blocks as u16 little endian
    uint16_t nblocks = static_cast<uint16_t>(blocks.size());
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&nblocks), reinterpret_cast<uint8_t*>(&nblocks) + 2);
    // process each block
    for (const auto &rng : blocks) {
        size_t start = rng.first;
        size_t end = rng.second;
        if (end > input.size()) end = input.size();
        std::vector<uint8_t> block(input.begin() + start, input.begin() + end);
        size_t orig_len = block.size();
        // high entropy guard: sample Shannon entropy on a subset to decide whether to skip heavy models
        auto sample_entropy = [](const std::vector<uint8_t> &data) {
            const size_t sample_rate = 64;
            size_t step = data.size() / sample_rate;
            if (step == 0) step = 1;
            std::array<uint32_t, 256> hist{};
            size_t count = 0;
            for (size_t i = 0; i < data.size(); i += step) {
                hist[data[i]]++;
                ++count;
            }
            double H = 0.0;
            for (uint32_t h : hist) {
                if (h > 0) {
                    double p = static_cast<double>(h) / count;
                    H -= p * std::log2(p);
                }
            }
            return H;
        };
        double ent = sample_entropy(block);
        // candidate encoders
        struct Candidate { uint8_t method; std::vector<uint8_t> payload; size_t model_overhead; };
        std::vector<Candidate> cands;
        cands.reserve(4);
        // always consider RAW
        cands.push_back({0, block, 1 + 4 + 4}); // method + orig_len + payload_len overhead (u32 + u32)
        // XOR model
        {
            auto pay = encode_xor(block);
            cands.push_back({1, pay, 1 + 4 + 4});
        }
        // BWT→MTF→Rice (skip if high entropy to save time)
        if (ent < 7.5) {
            auto pay = encode_model_bwt_mtf_rice(block);
            cands.push_back({2, pay, 1 + 4 + 4});
        }
        // LZ77 (skip in high entropy if desired)
        if (ent < 7.8) {
            auto pay = encode_lz77(block);
            cands.push_back({3, pay, 1 + 4 + 4});
        }
        // choose minimal total bytes
        size_t best_size = static_cast<size_t>(-1);
        Candidate best = cands[0];
        for (const auto &c : cands) {
            size_t total = c.model_overhead + c.payload.size();
            if (total < best_size) {
                best_size = total;
                best = c;
            }
        }
        // write method id
        out.push_back(best.method);
        // write orig_len and payload_len (little endian u32)
        uint32_t orig = static_cast<uint32_t>(orig_len);
        uint32_t paylen = static_cast<uint32_t>(best.payload.size());
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&orig), reinterpret_cast<uint8_t*>(&orig) + 4);
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&paylen), reinterpret_cast<uint8_t*>(&paylen) + 4);
        // write payload
        out.insert(out.end(), best.payload.begin(), best.payload.end());
    }
    return out;
}

// --------------------------------------------------------------------------
// Decompressor: reads the container format produced by compress_data()
// and reconstructs the original input.  Returns an empty vector on
// error.
// --------------------------------------------------------------------------
static std::vector<uint8_t> decompress_data(const std::vector<uint8_t> &data) {
    size_t pos = 0;
    // check magic
    if (data.size() < 4 || std::string((char*)data.data(), 4) != "KOLM") return {};
    pos += 4;
    // read block size (unused)
    if (pos + 4 > data.size()) return {};
    pos += 4;
    // read total_len
    if (pos + 8 > data.size()) return {};
    uint64_t total_len = *reinterpret_cast<const uint64_t*>(&data[pos]);
    pos += 8;
    // read number of blocks
    if (pos + 2 > data.size()) return {};
    uint16_t nblocks = *reinterpret_cast<const uint16_t*>(&data[pos]);
    pos += 2;
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(total_len));
    for (uint16_t b = 0; b < nblocks; ++b) {
        if (pos + 1 + 4 + 4 > data.size()) return {};
        uint8_t method = data[pos++];
        uint32_t orig_len = *reinterpret_cast<const uint32_t*>(&data[pos]);
        pos += 4;
        uint32_t pay_len = *reinterpret_cast<const uint32_t*>(&data[pos]);
        pos += 4;
        if (pos + pay_len > data.size()) return {};
        std::vector<uint8_t> payload(data.begin() + pos, data.begin() + pos + pay_len);
        pos += pay_len;
        std::vector<uint8_t> block;
        switch (method) {
            case 0:
                block = payload;
                break;
            case 1:
                block = decode_xor(payload, orig_len);
                break;
            case 2:
                block = decode_model_bwt_mtf_rice(payload, orig_len);
                break;
            case 3:
                block = decode_lz77(payload, orig_len);
                break;
            default:
                return {};
        }
        out.insert(out.end(), block.begin(), block.end());
    }
    if (out.size() != total_len) {
        // truncated or corrupt
        return {};
    }
    return out;
}

// --------------------------------------------------------------------------
// Main program: compress or decompress a file based on command‑line.
// Usage:
//   ./kolm_final -c input_file output_file
//   ./kolm_final -d input_file output_file
// If no files are provided, reads from stdin and writes to stdout.
// --------------------------------------------------------------------------
int main(int argc, char **argv) {
    bool compress_mode = true;
    std::string infile, outfile;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-c") compress_mode = true;
        else if (arg == "-d") compress_mode = false;
        else if (infile.empty()) infile = arg;
        else if (outfile.empty()) outfile = arg;
    }
    // read input
    std::vector<uint8_t> data;
    if (!infile.empty()) {
        std::ifstream fin(infile, std::ios::binary);
        if (!fin) {
            std::cerr << "Failed to open input file\n";
            return 1;
        }
        fin.seekg(0, std::ios::end);
        std::streamsize size = fin.tellg();
        fin.seekg(0, std::ios::beg);
        data.resize(size);
        fin.read(reinterpret_cast<char*>(data.data()), size);
    } else {
        // stdin
        std::istreambuf_iterator<char> it(std::cin.rdbuf()), end;
        data.assign(it, end);
    }
    std::vector<uint8_t> result;
    if (compress_mode) {
        result = compress_data(data);
    } else {
        result = decompress_data(data);
    }
    if (!outfile.empty()) {
        std::ofstream fout(outfile, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size()));
    } else {
        std::cout.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size()));
    }
    return 0;
}
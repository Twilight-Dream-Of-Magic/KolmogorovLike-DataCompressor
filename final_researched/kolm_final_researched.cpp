// kolm_final_researched.cpp
//
// C++20 translation of the researched Kolmogorov compressor prototype.  This
// implementation follows the "boolean‑circuit first, probability last" design:
// reversible bitwise transforms (BBWT, MTF, bit plane interleaving, LFSR
// whitening, nibble swap and bit reversal) prepare the data before entropy
// coding.  Additional simple coders such as naive LZ77, LFSR predictor and
// Re‑Pair grammar compression are provided as fallbacks.  The minimum
// description length (MDL) selector chooses the smallest encoding for each
// content‑defined chunk.
//
// Theoretical references:
//  * Duval's linear time Lyndon factorisation【154237816494091†L294-L301】.
//  * Bijective Burrows–Wheeler transform via Lyndon words【555806496076120†L507-L516】.
//  * Golomb/Rice coding for geometric distributions【966564189297361†L152-L170】.
//  * Re‑Pair grammar compression and its memory trade‑off【967225900425034†L139-L166】.
//  * Linear feedback shift registers (LFSR) and bitwise XOR operations【879695488005067†L166-L179】.
//  * FastCDC content defined chunking【629903081498632†L40-L52】.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ULEB128 encoding/decoding
static std::vector<uint8_t> uleb128_encode(uint64_t n)
{
    std::vector<uint8_t> out;
    do {
        uint8_t byte = n & 0x7Fu;
        n >>= 7;
        if (n)
            byte |= 0x80u;
        out.push_back(byte);
    } while (n);
    return out;
}

static uint64_t uleb128_decode(const std::vector<uint8_t>& data, size_t& pos)
{
    uint64_t result = 0;
    int shift = 0;
    while (true)
    {
        if (pos >= data.size())
            throw std::runtime_error("ULEB128 EOF");
        uint8_t b = data[pos++];
        result |= uint64_t(b & 0x7F) << shift;
        if ((b & 0x80) == 0)
            break;
        shift += 7;
    }
    return result;
}

// FastCDC simplified gear hash table
static std::array<uint32_t,256> make_gear_table()
{
    std::mt19937_64 rng(2025);
    std::array<uint32_t,256> tbl;
    for (auto& x : tbl)
        x = static_cast<uint32_t>(rng());
    return tbl;
}
static const auto GEAR = make_gear_table();

static std::vector<std::pair<size_t,size_t>> cdc_fast_boundaries(const std::vector<uint8_t>& data, size_t min_size=4096, size_t avg_size=8192, size_t max_size=16384)
{
    size_t n = data.size();
    if (n == 0)
        return {};
    int k = std::max(6, std::min(20, int(std::bit_width(avg_size) - 1)));
    uint32_t mask = (1u << k) - 1u;
    std::vector<std::pair<size_t,size_t>> bounds;
    size_t i = 0;
    while (i < n)
    {
        size_t start = i;
        uint32_t h = 0;
        size_t end_min = std::min(n, start + min_size);
        size_t end_max = std::min(n, start + max_size);
        i = end_min;
        while (i < end_max)
        {
            h = ((h << 1) & 0xFFFFFFFFu) + GEAR[data[i]];
            if ((h & mask) == 0u)
            {
                ++i;
                break;
            }
            ++i;
        }
        bounds.emplace_back(start, i);
    }
    return bounds;
}

// Duval Lyndon factorisation
static std::vector<std::pair<size_t,size_t>> duval_lyndon(const std::vector<uint8_t>& s)
{
    size_t n = s.size();
    std::vector<std::pair<size_t,size_t>> out;
    size_t i = 0;
    while (i < n)
    {
        size_t j = i + 1;
        size_t k = i;
        while (j < n && s[k] <= s[j])
        {
            if (s[k] < s[j])
                k = i;
            else
                ++k;
            ++j;
        }
        size_t p = j - k;
        while (i <= k)
        {
            out.emplace_back(i, i + p);
            i += p;
        }
    }
    return out;
}

// BBWT forward using Lyndon factorisation and simple SA per factor
static std::vector<uint8_t> bbwt_forward(const std::vector<uint8_t>& s)
{
    size_t n = s.size();
    if (n == 0)
        return {};
    auto facs = duval_lyndon(s);
    auto sa_prefix = [](const std::vector<uint8_t>& t) {
        size_t n = t.size();
        size_t k = 1;
        std::vector<int> rank(t.begin(), t.end());
        std::vector<int> tmp(n);
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        while (true)
        {
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                int ra = rank[a];
                int rb = rank[b];
                int ra2 = (a + k < n) ? rank[a + k] : -1;
                int rb2 = (b + k < n) ? rank[b + k] : -1;
                return std::pair(ra, ra2) < std::pair(rb, rb2);
            });
            tmp[idx[0]] = 0;
            for (size_t j = 1; j < n; ++j)
            {
                int a = idx[j - 1];
                int b = idx[j];
                tmp[b] = tmp[a] + ((std::pair(rank[a], (a + k < n ? rank[a + k] : -1)) < std::pair(rank[b], (b + k < n ? rank[b + k] : -1))) ? 1 : 0);
            }
            rank = tmp;
            if (rank[idx.back()] == int(n - 1))
                break;
            k <<= 1;
        }
        return idx;
    };
    // compute rotation order per factor
    struct Node
    {
        int fi;
        int k;
        const std::vector<uint8_t>* w;
        const std::vector<int>* order;
        bool operator<(const Node& other) const
        {
            int i = (*order)[k];
            int j = (*other.order)[other.k];
            const auto& u = *w;
            const auto& v = *other.w;
            int m = u.size();
            int n = v.size();
            int p = 0;
            while (p < m + n)
            {
                uint8_t cu = u[(i + p) % m];
                uint8_t cv = v[(j + p) % n];
                if (cu != cv)
                    return cu > cv; // reverse for min-heap
                ++p;
            }
            return std::pair(fi, i) > std::pair(other.fi, j);
        }
    };
    std::vector<std::vector<uint8_t>> words;
    std::vector<std::vector<int>> orders;
    for (auto [a,b] : facs)
    {
        std::vector<uint8_t> w(s.begin() + a, s.begin() + b);
        size_t m = w.size();
        std::vector<uint8_t> ww = w;
        ww.insert(ww.end(), w.begin(), w.end());
        auto sa = sa_prefix(ww);
        std::vector<int> rot;
        for (auto p : sa)
            if (p < int(m))
                rot.push_back(p);
        words.push_back(std::move(w));
        orders.push_back(std::move(rot));
    }
    // k-way merge using a heap
    std::priority_queue<Node> heap;
    for (size_t fi = 0; fi < words.size(); ++fi)
    {
        if (!orders[fi].empty())
            heap.push(Node{(int)fi, 0, &words[fi], &orders[fi]});
    }
    std::vector<uint8_t> out;
    out.reserve(n);
    while (!heap.empty())
    {
        Node nd = heap.top(); heap.pop();
        int i = (*nd.order)[nd.k];
        const auto& w = *nd.w;
        int m = (int)w.size();
        out.push_back(w[(i - 1 + m) % m]);
        nd.k++;
        if (nd.k < (int)nd.order->size())
            heap.push(nd);
    }
    return out;
}

static std::vector<uint8_t> bbwt_inverse(const std::vector<uint8_t>& L)
{
    size_t n = L.size();
    if (n == 0)
        return {};
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        return std::pair(L[a], a) < std::pair(L[b], b);
    });
    std::vector<int> pi = order;
    std::vector<bool> seen(n);
    std::vector<std::vector<int>> cycles;
    for (size_t i = 0; i < n; ++i)
    {
        if (!seen[i])
        {
            size_t cur = i;
            std::vector<int> cyc;
            while (!seen[cur])
            {
                seen[cur] = true;
                cyc.push_back(cur);
                cur = pi[cur];
            }
            cycles.push_back(std::move(cyc));
        }
    }
    std::sort(cycles.begin(), cycles.end(), [](const auto& a, const auto& b){ return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end()); });
    std::vector<uint8_t> out;
    out.reserve(n);
    for (auto it = cycles.rbegin(); it != cycles.rend(); ++it)
    {
        const auto& cyc = *it;
        int i0 = *std::min_element(cyc.begin(), cyc.end());
        int d = (int)cyc.size();
        int cur = i0;
        std::vector<uint8_t> seq;
        seq.reserve(d);
        for (int _ = 0; _ < d; ++_)
        {
            cur = pi[cur];
            seq.push_back(L[cur]);
        }
        out.insert(out.end(), seq.begin(), seq.end());
    }
    return out;
}

// Move-to-front
static std::vector<uint8_t> mtf_encode(const std::vector<uint8_t>& data)
{
    std::array<uint8_t,256> table;
    std::iota(table.begin(), table.end(), 0);
    std::vector<uint8_t> out;
    out.reserve(data.size());
    for (uint8_t b : data)
    {
        auto it = std::find(table.begin(), table.end(), b);
        uint8_t idx = it - table.begin();
        out.push_back(idx);
        // move to front
        std::rotate(table.begin(), it, it+1);
        table[0] = b;
    }
    return out;
}

static std::vector<uint8_t> mtf_decode(const std::vector<uint8_t>& seq)
{
    std::array<uint8_t,256> table;
    std::iota(table.begin(), table.end(), 0);
    std::vector<uint8_t> out;
    out.reserve(seq.size());
    for (uint8_t idx : seq)
    {
        uint8_t b = table[idx];
        out.push_back(b);
        std::rotate(table.begin(), table.begin()+idx, table.begin()+idx+1);
        table[0] = b;
    }
    return out;
}

// Bitwise circuits
static std::vector<uint8_t> bitplane_interleave(const std::vector<uint8_t>& data)
{
    size_t n = data.size();
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n);
    // pack bits bit by bit across bytes
    std::vector<int> bits;
    bits.reserve(n * 8);
    for (int bit = 7; bit >= 0; --bit)
    {
        for (size_t i = 0; i < n; ++i)
            bits.push_back((data[i] >> bit) & 1);
    }
    for (size_t i = 0; i < bits.size(); i += 8)
    {
        uint8_t b = 0;
        for (int j = 0; j < 8 && i + j < bits.size(); ++j)
            b |= (bits[i + j] << (7 - j));
        out.push_back(b);
    }
    return out;
}

static std::vector<uint8_t> bitplane_deinterleave(const std::vector<uint8_t>& data, size_t orig_len)
{
    if (orig_len == 0) return {};
    // extract bits
    std::vector<int> bits;
    bits.reserve(data.size() * 8);
    for (uint8_t byte : data)
    {
        for (int j = 7; j >= 0; --j)
            bits.push_back((byte >> j) & 1);
    }
    std::vector<uint8_t> out(orig_len);
    for (size_t bit = 0; bit < 8; ++bit)
    {
        for (size_t i = 0; i < orig_len; ++i)
        {
            out[i] |= bits[bit * orig_len + i] << (7 - bit);
        }
    }
    return out;
}

static uint8_t nibble_swap_byte(uint8_t b)
{
    return (uint8_t)(((b & 0x0F) << 4) | ((b & 0xF0) >> 4));
}
static std::vector<uint8_t> nibble_swap_vec(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = nibble_swap_byte(data[i]);
    return out;
}

// bit reversal lookup table
static std::array<uint8_t,256> make_bit_reverse_table()
{
    std::array<uint8_t,256> tbl;
    for (int i = 0; i < 256; ++i)
    {
        uint8_t b = i;
        uint8_t r = 0;
        for (int j = 0; j < 8; ++j)
        {
            r |= ((b >> j) & 1) << (7 - j);
        }
        tbl[i] = r;
    }
    return tbl;
}
static const auto BITREV = make_bit_reverse_table();
static std::vector<uint8_t> bit_reverse_vec(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = BITREV[data[i]];
    return out;
}

// Gray code transform
// A Gray code g(x) = x ^ (x >> 1) ensures that consecutive integer codes
// differ by only one bit.  We apply the Gray transform on each byte to
// expose local structure before entropy coding.  The inverse transform
// reconstructs the original value by cumulatively XORing the shifted Gray
// code, as derived from the definition【882443461041209†L485-L536】【882443461041209†L564-L580】.

static std::vector<uint8_t> gray_encode_vec(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        uint8_t x = data[i];
        out[i] = (uint8_t)((x ^ (x >> 1)) & 0xFF);
    }
    return out;
}

static std::vector<uint8_t> gray_decode_vec(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        uint8_t g = data[i];
        uint8_t n = g;
        n ^= (n >> 1);
        n ^= (n >> 2);
        n ^= (n >> 4);
        out[i] = n;
    }
    return out;
}

static std::vector<uint8_t> lfsr_whiten(const std::vector<uint8_t>& data, uint8_t taps=0x96, uint8_t seed=1)
{
    uint8_t state = seed;
    std::vector<uint8_t> out;
    out.reserve(data.size());
    for (uint8_t b : data)
    {
        out.push_back(b ^ state);
        uint8_t fb = 0;
        for (int bit = 0; bit < 8; ++bit)
        {
            if ((taps >> bit) & 1)
                fb ^= (state >> bit) & 1;
        }
        state = (uint8_t)(((state << 1) & 0xFF) | fb);
    }
    return out;
}

// Rice coding (no unary/gamma encode for metadata; we embed k and length externally)
static std::vector<uint8_t> rice_encode(const std::vector<uint8_t>& seq, int k)
{
    int M = 1 << k;
    std::string bitstr;
    bitstr.reserve(seq.size());
    for (uint8_t n : seq)
    {
        int q = n / M;
        int r = n % M;
        bitstr.append(q, '1');
        bitstr.push_back('0');
        for (int i = k - 1; i >= 0; --i)
            bitstr.push_back(((r >> i) & 1) + '0');
    }
    int pad = (8 - (int)bitstr.size() % 8) % 8;
    bitstr.append(pad, '0');
    std::vector<uint8_t> out;
    out.reserve(bitstr.size() / 8);
    for (size_t i = 0; i < bitstr.size(); i += 8)
    {
        uint8_t b = 0;
        for (int j = 0; j < 8; ++j)
            b = (b << 1) | (bitstr[i + j] - '0');
        out.push_back(b);
    }
    return out;
}

static std::vector<uint8_t> rice_decode(const std::vector<uint8_t>& data, int k, size_t nvals)
{
    std::string bitstr;
    bitstr.reserve(data.size() * 8);
    for (uint8_t b : data)
    {
        for (int j = 7; j >= 0; --j)
            bitstr.push_back(((b >> j) & 1) + '0');
    }
    size_t i = 0;
    int M = 1 << k;
    std::vector<uint8_t> out;
    out.reserve(nvals);
    while (out.size() < nvals && i < bitstr.size())
    {
        int q = 0;
        while (bitstr[i] == '1') { ++q; ++i; }
        ++i; // skip '0'
        int r = 0;
        for (int j = 0; j < k; ++j)
        {
            r = (r << 1) | (bitstr[i++] - '0');
        }
        out.push_back(static_cast<uint8_t>(q * M + r));
    }
    return out;
}

// Naive LZ77 encoder/decoder
static std::pair<std::vector<uint8_t>, std::monostate> encode_lz77(const std::vector<uint8_t>& block)
{
    std::vector<uint8_t> window;
    std::vector<uint8_t> out;
    size_t pos = 0;
    size_t n = block.size();
    while (pos < n)
    {
        size_t best_len = 0, best_dist = 0;
        for (size_t dist = 1; dist <= std::min(window.size(), (size_t)255); ++dist)
        {
            size_t length = 0;
            while (length < 255 && pos + length < n && window[window.size() - dist + length] == block[pos + length])
                ++length;
            if (length > best_len)
            {
                best_len = length;
                best_dist = dist;
            }
        }
        if (best_len >= 3)
        {
            out.push_back(1);
            auto e1 = uleb128_encode(best_len);
            out.insert(out.end(), e1.begin(), e1.end());
            auto e2 = uleb128_encode(best_dist);
            out.insert(out.end(), e2.begin(), e2.end());
            for (size_t i = 0; i < best_len; ++i)
                window.push_back(block[pos + i]);
            pos += best_len;
        }
        else
        {
            out.push_back(0);
            out.push_back(block[pos]);
            window.push_back(block[pos]);
            ++pos;
        }
        if (window.size() > 4096)
            window.erase(window.begin(), window.end() - 4096);
    }
    return {out, {}};
}

static std::vector<uint8_t> decode_lz77(const std::vector<uint8_t>& data, size_t orig_len)
{
    std::vector<uint8_t> window;
    std::vector<uint8_t> out;
    size_t i = 0;
    while (out.size() < orig_len)
    {
        uint8_t flag = data[i++];
        if (flag == 0)
        {
            uint8_t b = data[i++];
            out.push_back(b);
            window.push_back(b);
        }
        else
        {
            size_t pos_tmp = i;
            uint64_t length = uleb128_decode(data, pos_tmp);
            uint64_t dist = uleb128_decode(data, pos_tmp);
            i = pos_tmp;
            for (size_t k = 0; k < length; ++k)
            {
                uint8_t b = window[window.size() - dist];
                out.push_back(b);
                window.push_back(b);
            }
        }
        if (window.size() > 4096)
            window.erase(window.begin(), window.end() - 4096);
    }
    return out;
}

// Simple Re‑Pair grammar compressor (same as Python version)
static std::pair<std::vector<uint8_t>, std::monostate> encode_repair(const std::vector<uint8_t>& block)
{
    std::vector<int> seq(block.begin(), block.end());
    int next_sym = 256;
    std::unordered_map<int, std::pair<int,int>> rules;
    int max_rules = 256;
    while (true)
    {
        std::unordered_map<long long,int> freq;
        int best_f = 1;
        long long best_pair = -1;
        for (size_t i = 0; i + 1 < seq.size(); ++i)
        {
            long long key = ((long long)seq[i] << 32) | (unsigned)seq[i+1];
            int f = ++freq[key];
            if (f > best_f)
            {
                best_f = f;
                best_pair = key;
            }
        }
        if (best_pair < 0 || next_sym >= 256 + max_rules)
            break;
        int a = (int)(best_pair >> 32);
        int b = (int)(best_pair & 0xFFFFFFFF);
        rules[next_sym] = {a,b};
        std::vector<int> new_seq;
        new_seq.reserve(seq.size());
        for (size_t i = 0; i < seq.size();)
        {
            if (i + 1 < seq.size() && seq[i] == a && seq[i+1] == b)
            {
                new_seq.push_back(next_sym);
                i += 2;
            }
            else
            {
                new_seq.push_back(seq[i]);
                ++i;
            }
        }
        seq.swap(new_seq);
        ++next_sym;
    }
    std::vector<uint8_t> out;
    auto enc = uleb128_encode(rules.size());
    out.insert(out.end(), enc.begin(), enc.end());
    for (auto& kv : rules)
    {
        int nt = kv.first;
        int a = kv.second.first;
        int b = kv.second.second;
        out.push_back((uint8_t)(nt - 256));
        out.push_back((uint8_t)a);
        out.push_back((uint8_t)b);
    }
    auto enc_len = uleb128_encode(seq.size());
    out.insert(out.end(), enc_len.begin(), enc_len.end());
    for (int sym : seq)
        out.push_back((uint8_t)(sym < 256 ? sym : sym - 256));
    return {out, {}};
}

static std::vector<uint8_t> decode_repair(const std::vector<uint8_t>& data, size_t orig_len)
{
    size_t pos = 0;
    size_t nrules = uleb128_decode(data, pos);
    std::unordered_map<int, std::pair<int,int>> rules;
    for (size_t i = 0; i < nrules; ++i)
    {
        int nt = data[pos++] + 256;
        int a = data[pos++];
        int b = data[pos++];
        rules[nt] = {a,b};
    }
    size_t seq_len = uleb128_decode(data, pos);
    std::vector<int> seq;
    seq.reserve(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
    {
        int sym = data[pos++];
        if (sym >= 256)
            sym += 256;
        seq.push_back(sym);
    }
    std::function<void(int,std::vector<uint8_t>&)> expand = [&](int sym, std::vector<uint8_t>& out) {
        if (sym < 256)
            out.push_back((uint8_t)sym);
        else
        {
            auto it = rules.find(sym);
            if (it != rules.end())
            {
                expand(it->second.first, out);
                expand(it->second.second, out);
            }
        }
    };
    std::vector<uint8_t> out;
    out.reserve(orig_len);
    for (int sym : seq)
        expand(sym, out);
    return out;
}

// LFSR predictor encode/decode
static std::pair<std::vector<uint8_t>, std::monostate> encode_lfsr_predict(const std::vector<uint8_t>& block)
{
    uint8_t state = 1;
    std::vector<uint8_t> out;
    out.reserve(block.size());
    for (uint8_t b : block)
    {
        uint8_t pred = state;
        uint8_t delta = (uint8_t)((b - pred) & 0xFF);
        auto enc = uleb128_encode(delta);
        out.insert(out.end(), enc.begin(), enc.end());
        uint8_t fb = 0;
        uint8_t taps = 0x96;
        for (int bit = 0; bit < 8; ++bit)
            if ((taps >> bit) & 1)
                fb ^= (state >> bit) & 1;
        state = (uint8_t)(((state << 1) & 0xFF) | fb);
    }
    return {out, {}};
}

static std::vector<uint8_t> decode_lfsr_predict(const std::vector<uint8_t>& data, size_t orig_len)
{
    uint8_t state = 1;
    std::vector<uint8_t> out;
    out.reserve(orig_len);
    size_t pos = 0;
    for (size_t i = 0; i < orig_len; ++i)
    {
        uint64_t delta = uleb128_decode(data, pos);
        uint8_t b = (uint8_t)((delta + state) & 0xFF);
        out.push_back(b);
        uint8_t fb = 0;
        uint8_t taps = 0x96;
        for (int bit = 0; bit < 8; ++bit)
            if ((taps >> bit) & 1)
                fb ^= (state >> bit) & 1;
        state = (uint8_t)(((state << 1) & 0xFF) | fb);
    }
    return out;
}

// BBWT→MTF→Rice model with optional bitwise modules
enum MethodID {
    RAW = 0,
    XOR_RESIDUAL = 1,
    BBWT_RICE = 2,
    BBWT_BITPLANE = 3,
    BBWT_LFSR = 4,
    BBWT_NIBBLE = 5,
    BBWT_BITREV = 6,
    BBWT_BP_LFSR = 7,
    BBWT_GRAY = 8,
    LZ77 = 9,
    LFSR_PRED = 10,
    REPAIR = 11
};

static std::pair<std::vector<uint8_t>, std::monostate> encode_bbwt_mtf_rice(const std::vector<uint8_t>& block,
                                                  bool use_bitplane,
                                                  bool use_lfsr,
                                                  bool use_nibble,
                                                  bool use_bitrev,
                                                  bool use_gray = false,
                                                  int k = 2)
{
    auto bbwt = bbwt_forward(block);
    auto mtf = mtf_encode(bbwt);
    std::vector<uint8_t> seq_bytes = mtf;
    if (use_bitplane)
        seq_bytes = bitplane_interleave(seq_bytes);
    if (use_lfsr)
        seq_bytes = lfsr_whiten(seq_bytes);
    if (use_nibble)
        seq_bytes = nibble_swap_vec(seq_bytes);
    if (use_bitrev)
        seq_bytes = bit_reverse_vec(seq_bytes);
    if (use_gray)
        seq_bytes = gray_encode_vec(seq_bytes);
    auto payload = rice_encode(seq_bytes, k);
    return {payload, {}};
}

static std::vector<uint8_t> decode_bbwt_mtf_rice(const std::vector<uint8_t>& payload,
                                    size_t orig_len,
                                    bool use_bitplane,
                                    bool use_lfsr,
                                    bool use_nibble,
                                    bool use_bitrev,
                                    bool use_gray = false,
                                    int k = 2)
{
    size_t nvals = orig_len;
    auto seq = rice_decode(payload, k, nvals);
    if (use_gray)
        seq = gray_decode_vec(seq);
    if (use_bitrev)
        seq = bit_reverse_vec(seq);
    if (use_nibble)
        seq = nibble_swap_vec(seq);
    if (use_lfsr)
        seq = lfsr_whiten(seq);
    if (use_bitplane)
        seq = bitplane_deinterleave(seq, orig_len);
    auto mtf_dec = mtf_decode(seq);
    return bbwt_inverse(mtf_dec);
}

// XOR residual encode/decode
static std::pair<std::vector<uint8_t>, std::monostate> encode_xor(const std::vector<uint8_t>& block)
{
    std::vector<uint8_t> out;
    out.reserve(block.size());
    uint8_t prev = 0;
    for (uint8_t b : block)
    {
        uint8_t delta = (uint8_t)((b - prev) & 0xFF);
        auto e = uleb128_encode(delta);
        out.insert(out.end(), e.begin(), e.end());
        prev = b;
    }
    return {out, {}};
}

static std::vector<uint8_t> decode_xor(const std::vector<uint8_t>& data, size_t orig_len)
{
    std::vector<uint8_t> out;
    out.reserve(orig_len);
    uint8_t prev = 0;
    size_t pos = 0;
    for (size_t i = 0; i < orig_len; ++i)
    {
        uint64_t delta = uleb128_decode(data, pos);
        uint8_t b = (uint8_t)((prev + delta) & 0xFF);
        out.push_back(b);
        prev = b;
    }
    return out;
}

// Raw copy encode/decode
static std::pair<std::vector<uint8_t>, std::monostate> encode_raw(const std::vector<uint8_t>& block)
{
    return {block, {}};
}
static std::vector<uint8_t> decode_raw(const std::vector<uint8_t>& data, size_t orig_len)
{
    return data;
}

// MDL block compressor
static std::vector<uint8_t> compress_blocks(const std::vector<uint8_t>& data, size_t block_size)
{
    auto bounds = cdc_fast_boundaries(data, 4096, block_size, block_size * 2);
    std::vector<uint8_t> out;
    // header: magic, block size, total len, number of blocks
    out.insert(out.end(), {'K','O','L','R'});
    out.push_back((uint8_t)(block_size & 0xFF));
    out.push_back((uint8_t)((block_size >> 8) & 0xFF));
    out.push_back((uint8_t)((block_size >> 16) & 0xFF));
    out.push_back((uint8_t)((block_size >> 24) & 0xFF));
    uint32_t total = (uint32_t)data.size();
    out.push_back((uint8_t)(total & 0xFF));
    out.push_back((uint8_t)((total >> 8) & 0xFF));
    out.push_back((uint8_t)((total >> 16) & 0xFF));
    out.push_back((uint8_t)((total >> 24) & 0xFF));
    uint16_t nb = (uint16_t)bounds.size();
    out.push_back((uint8_t)(nb & 0xFF));
    out.push_back((uint8_t)((nb >> 8) & 0xFF));
    for (auto [start,end] : bounds)
    {
        std::vector<uint8_t> block(data.begin() + start, data.begin() + end);
        // evaluate models
        std::vector<std::pair<int,std::vector<uint8_t>>> candidates;
        // try raw
        candidates.emplace_back(RAW, block);
        // XOR
        candidates.emplace_back(XOR_RESIDUAL, encode_xor(block).first);
        // BBWT variants
        candidates.emplace_back(BBWT_RICE, encode_bbwt_mtf_rice(block, false, false, false, false, false).first);
        candidates.emplace_back(BBWT_BITPLANE, encode_bbwt_mtf_rice(block, true, false, false, false, false).first);
        candidates.emplace_back(BBWT_LFSR, encode_bbwt_mtf_rice(block, false, true, false, false, false).first);
        candidates.emplace_back(BBWT_NIBBLE, encode_bbwt_mtf_rice(block, false, false, true, false, false).first);
        candidates.emplace_back(BBWT_BITREV, encode_bbwt_mtf_rice(block, false, false, false, true, false).first);
        candidates.emplace_back(BBWT_BP_LFSR, encode_bbwt_mtf_rice(block, true, true, false, false, false).first);
        // Gray code variant
        candidates.emplace_back(BBWT_GRAY, encode_bbwt_mtf_rice(block, false, false, false, false, true).first);
        // LZ77
        candidates.emplace_back(LZ77, encode_lz77(block).first);
        // LFSR predictor
        candidates.emplace_back(LFSR_PRED, encode_lfsr_predict(block).first);
        // RePair
        candidates.emplace_back(REPAIR, encode_repair(block).first);
        // choose minimal size
        int best_id = RAW;
        size_t best_size = candidates[0].second.size();
        for (auto& [mid,payload] : candidates)
        {
            if (payload.size() < best_size)
            {
                best_size = payload.size();
                best_id = mid;
            }
        }
        // record block header: method_id, orig_len, payload_len
        // note: we will re-encode the chosen block here to avoid storing meta
        std::vector<uint8_t> payload;
        switch (best_id)
        {
            case RAW: payload = block; break;
            case XOR_RESIDUAL: payload = encode_xor(block).first; break;
            case BBWT_RICE: payload = encode_bbwt_mtf_rice(block, false, false, false, false, false).first; break;
            case BBWT_BITPLANE: payload = encode_bbwt_mtf_rice(block, true, false, false, false, false).first; break;
            case BBWT_LFSR: payload = encode_bbwt_mtf_rice(block, false, true, false, false, false).first; break;
            case BBWT_NIBBLE: payload = encode_bbwt_mtf_rice(block, false, false, true, false, false).first; break;
            case BBWT_BITREV: payload = encode_bbwt_mtf_rice(block, false, false, false, true, false).first; break;
            case BBWT_BP_LFSR: payload = encode_bbwt_mtf_rice(block, true, true, false, false, false).first; break;
            case BBWT_GRAY: payload = encode_bbwt_mtf_rice(block, false, false, false, false, true).first; break;
            case LZ77: payload = encode_lz77(block).first; break;
            case LFSR_PRED: payload = encode_lfsr_predict(block).first; break;
            case REPAIR: payload = encode_repair(block).first; break;
        }
        out.push_back((uint8_t)best_id);
        uint32_t orig = (uint32_t)block.size();
        out.push_back((uint8_t)(orig & 0xFF));
        out.push_back((uint8_t)((orig >> 8) & 0xFF));
        out.push_back((uint8_t)((orig >> 16) & 0xFF));
        out.push_back((uint8_t)((orig >> 24) & 0xFF));
        uint32_t plen = (uint32_t)payload.size();
        out.push_back((uint8_t)(plen & 0xFF));
        out.push_back((uint8_t)((plen >> 8) & 0xFF));
        out.push_back((uint8_t)((plen >> 16) & 0xFF));
        out.push_back((uint8_t)((plen >> 24) & 0xFF));
        out.insert(out.end(), payload.begin(), payload.end());
    }
    return out;
}

static std::vector<uint8_t> decompress_blocks(const std::vector<uint8_t>& data)
{
    size_t pos = 0;
    if (data.size() < 12 || data[0] != 'K' || data[1] != 'O' || data[2] != 'L' || data[3] != 'R')
        throw std::runtime_error("Invalid magic");
    pos = 4;
    uint32_t block_size = data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24);
    pos += 4;
    uint32_t total_len = data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24);
    pos += 4;
    uint16_t nblocks = data[pos] | (data[pos+1] << 8);
    pos += 2;
    std::vector<uint8_t> out;
    out.reserve(total_len);
    for (uint16_t bi = 0; bi < nblocks; ++bi)
    {
        uint8_t method_id = data[pos++];
        uint32_t orig_len = data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24);
        pos += 4;
        uint32_t payload_len = data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24);
        pos += 4;
        std::vector<uint8_t> payload(data.begin() + pos, data.begin() + pos + payload_len);
        pos += payload_len;
        std::vector<uint8_t> block;
        switch (method_id)
        {
            case RAW: block = decode_raw(payload, orig_len); break;
            case XOR_RESIDUAL: block = decode_xor(payload, orig_len); break;
            case BBWT_RICE: block = decode_bbwt_mtf_rice(payload, orig_len, false, false, false, false, false); break;
            case BBWT_BITPLANE: block = decode_bbwt_mtf_rice(payload, orig_len, true, false, false, false, false); break;
            case BBWT_LFSR: block = decode_bbwt_mtf_rice(payload, orig_len, false, true, false, false, false); break;
            case BBWT_NIBBLE: block = decode_bbwt_mtf_rice(payload, orig_len, false, false, true, false, false); break;
            case BBWT_BITREV: block = decode_bbwt_mtf_rice(payload, orig_len, false, false, false, true, false); break;
            case BBWT_BP_LFSR: block = decode_bbwt_mtf_rice(payload, orig_len, true, true, false, false, false); break;
            case BBWT_GRAY: block = decode_bbwt_mtf_rice(payload, orig_len, false, false, false, false, true); break;
            case LZ77: block = decode_lz77(payload, orig_len); break;
            case LFSR_PRED: block = decode_lfsr_predict(payload, orig_len); break;
            case REPAIR: block = decode_repair(payload, orig_len); break;
            default: throw std::runtime_error("Unknown method");
        }
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

// Main CLI
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <file> [-d] [-o output] [-b blocksize]" << std::endl;
        return 1;
    }
    std::string input = argv[1];
    bool decompress_flag = false;
    std::string output;
    size_t blocksize = 8192;
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-d")
            decompress_flag = true;
        else if (arg == "-o" && i + 1 < argc)
            output = argv[++i];
        else if (arg == "-b" && i + 1 < argc)
            blocksize = std::stoul(argv[++i]);
    }
    // read input file
    std::ifstream fin(input, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    if (!fin)
    {
        std::cerr << "Failed to read input file" << std::endl;
        return 1;
    }
    if (decompress_flag)
    {
        auto outbuf = decompress_blocks(data);
        std::string outname = !output.empty() ? output : input + ".out";
        std::ofstream fout(outname, std::ios::binary);
        fout.write((const char*)outbuf.data(), outbuf.size());
        std::cerr << "Decompressed " << data.size() << " bytes to " << outbuf.size() << " bytes -> " << outname << std::endl;
    }
    else
    {
        auto blob = compress_blocks(data, blocksize);
        std::string outname = !output.empty() ? output : input + ".kolr";
        std::ofstream fout(outname, std::ios::binary);
        fout.write((const char*)blob.data(), blob.size());
        double ratio = data.empty() ? 1.0 : double(blob.size()) / double(data.size());
        std::cerr << "Compressed " << data.size() << " bytes to " << blob.size() << " bytes (ratio " << ratio << ") -> " << outname << std::endl;
    }
    return 0;
}
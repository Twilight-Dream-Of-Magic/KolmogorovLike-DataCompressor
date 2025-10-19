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
#include <string_view>
#include <sstream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
#include <span>
#include <filesystem>

// NOTE: EF bitstream I/O function declarations are placed *after* BitWriter/BitReader definitions.
// or timing and formatted output in run_self_test.  These
// headers provide std::chrono for wall‑clock measurements and std::setw,
// std::setprecision, etc. for nicely formatted tables.
#include <bit>		// for std::bit_width (if used from linked units)
#include <chrono>
#include <iomanip>
#include <queue>
#include <sstream>	// for std::ostringstream in toc_brief()

using ByteSpan = std::span<const std::uint8_t>;

// --no-lz77
static bool G_NO_LZ77 = false;

// --only <method_id or name>
static bool		   G_ONLY_ENABLED = false;
static std::size_t G_ONLY_METHOD = static_cast<std::size_t>( -1 );

// --progress : show per-block progress on stderr
static bool G_PROGRESS = false;

void set_no_lz77( bool v )
{
	G_NO_LZ77 = v;
}

void set_only_method( std::size_t id )
{
	G_ONLY_ENABLED = true;
	G_ONLY_METHOD = id;
}

void clear_only()
{
	G_ONLY_ENABLED = false;
	G_ONLY_METHOD = static_cast<std::size_t>( -1 );
}

// =========================================
// Header field: mode pack/unpack (compatibility bit)
// =========================================
// MODE_FIXED = 0  # fixed-size chunking
// MODE_CDC   = 1  # FastCDC
constexpr std::uint32_t MODE_FIXED = 0;
constexpr std::uint32_t MODE_CDC = 1;

// Forward declarations for functions defined later in the file.  The
// compressor uses a wide variety of helper routines and data structures
// defined throughout this file; listing prototypes up front makes the
// ordering of definitions less critical.

struct BitWriter;

// ------------------------------------------------------------
// BitWriter: MSB-first bitstream writer used by Rice encoder.
// We emit bits left-to-right and pack them into bytes where the
// first bit we write goes to bit position 7 (MSB).
// ------------------------------------------------------------
struct BitWriter
{
	std::vector<std::uint8_t> buf;
	std::uint8_t			  cur { 0 };
	int						  bitpos { 0 };	 // number of bits already written into 'cur' (0..7)

	// Write a single bit (0/1) MSB-first.
	void write_bit( int b )
	{
		cur |= ( ( b & 1 ) << ( 7 - bitpos ) );
		++bitpos;
		if ( bitpos == 8 )
		{
			buf.push_back( cur );
			cur = 0;
			bitpos = 0;
		}
	}

	// Write 'k' bits of 'value' MSB-first (like Python's format(...,'0kb')).
	void write_kbits( std::uint32_t value, int k )
	{
		for ( int i = k - 1; i >= 0; --i )
			write_bit( ( value >> i ) & 1 );
	}

	// Write q in unary: '1'*q + '0'.
	void write_unary( std::uint32_t q )
	{
		for ( std::uint32_t i = 0; i < q; ++i )
			write_bit( 1 );
		write_bit( 0 );
	}

	// Pad with zeros to the next byte boundary.
	void pad_to_byte()
	{
		if ( bitpos != 0 )
		{
			buf.push_back( cur );
			cur = 0;
			bitpos = 0;
		}
	}

	// Helper: align BitWriter to next byte boundary (pad with zeros if needed).
	/* Return total number of valid bits currently buffered (without padding). */
	// MSB-first within each byte; counts complete bytes plus partial bits in 'cur'.
	std::size_t bit_length() const noexcept
	{
		return static_cast<std::size_t>( buf.size() ) * 8u + static_cast<std::size_t>( bitpos );
	}
};

struct BitReader;

// ------------------------------------------------------------
// BitReader: MSB-first bitstream reader used by Rice decoder.
// read_bit() returns the next bit (0/1). We can align to the
// next byte and query the current byte/bit cursor via tell().
// ------------------------------------------------------------
struct BitReader
{
	const std::vector<std::uint8_t>& buf;
	std::size_t						 byte { 0 };
	int								 bit { 0 };	 // 0..7; 0 means next read takes MSB of buf[byte]

	explicit BitReader( const std::vector<std::uint8_t>& b, std::size_t byte_pos = 0, int bit_pos = 0 ) : buf( b ), byte( byte_pos ), bit( bit_pos ) {}

	int read_bit()
	{
		if ( byte >= buf.size() )
			throw std::runtime_error( "BitReader: out of data" );
		const std::uint8_t B = buf[ byte ];
		const int		   v = ( B >> ( 7 - bit ) ) & 1;
		++bit;
		if ( bit == 8 )
		{
			bit = 0;
			++byte;
		}
		return v;
	}

	void align_next_byte()
	{
		if ( bit != 0 )
		{
			bit = 0;
			++byte;
		}
	}

	std::pair<std::size_t, int> tell() const
	{
		return { byte, bit };
	}
};

const char* method_name_from_id( std::size_t id )
{
	switch ( id )
	{
	case 0:
		return "Raw";
	case 1:
		return "XOR";
	case 2:
		return "BBWT";
	case 3:
		return "BBWT+Bitplane";
	case 4:
		return "BBWT+Nibble";
	case 5:
		return "BBWT+BitRev";
	case 6:
		return "BBWT+Gray";
	case 7:
		return "LZ77";
	case 8:
		return "LFSR predictor";
	case 9:
		return "Re-Pair";
	case 10:
		return "V2 New";
	default:
		return "Unknown";
	}
}

std::optional<std::size_t> method_id_from_name( std::string_view s )
{
	// 与 candidates 顺序一致
	if ( s == "raw" )
		return 0;
	if ( s == "xor" || s == "delta" )
		return 1;
	if ( s == "bbwt" )
		return 2;
	if ( s == "bbwt+bitplane" || s == "bbwt-plane" )
		return 3;
	if ( s == "bbwt+nibble" || s == "bbwt-nibble" )
		return 4;
	if ( s == "bbwt+bitrev" || s == "bbwt-rev" )
		return 5;
	if ( s == "bbwt+gray" || s == "bbwt-gray" )
		return 6;
	if ( s == "lz77" )
		return 7;
	if ( s == "lfsr" )
		return 8;
	if ( s == "repair" || s == "re-pair" )
		return 9;
	if ( s == "v2" || s == "new" || s == "new-pipeline" )
		return 10;
	return std::nullopt;
}

// -----------------------------------------------------------------------------
// PairHash helper
//
// The Re‑Pair grammar compression algorithm (Re‑Pair) counts the frequency of adjacent symbol pairs using an
// unordered_map with keys of type std::pair<int,int>.  The standard library
// does not provide a noexcept default‑constructible hash for std::pair in all
// implementations (and even when it does, the resulting unordered_map may
// assert that the hash functor is not noexcept).  We define our own hash
// functor here that combines the two 32‑bit integers into a 64‑bit value and
// hashes that using std::hash<std::uint64_t>.  This ensures the hash type is
// properly constructible and noexcept.
struct PairHash
{
	std::size_t operator()( const std::pair<int, int>& p ) const noexcept
	{
		std::uint64_t combined = ( static_cast<std::uint64_t>( static_cast<std::uint32_t>( p.first ) ) << 32 ) ^ static_cast<std::uint32_t>( p.second );
		return std::hash<std::uint64_t> {}( combined );
	}
};


// ULEB128 coding
std::vector<std::uint8_t>			  uleb128_encode( std::uint64_t n );
std::pair<std::uint64_t, std::size_t> uleb128_decode_stream( const std::vector<std::uint8_t>& data, std::size_t pos );

// ---------------------------------------------------------------------------

/*RLE + canonical Huffman helpers used by TOC writers/readers*/

// Run‑length encode a sequence of method IDs. Returns (symbols, run_lengths).
// Marked static inline to match the implementation below and avoid linkage conflicts.
std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> rle_ids( const std::vector<std::uint8_t>& ids );

// Compute Huffman code lengths from symbol frequencies.  The input is a map
// from symbol to its frequency count; the output is a map of symbol to code length.
std::unordered_map<std::uint32_t, int> huff_lengths( const std::unordered_map<std::uint32_t, std::uint32_t>& freq );

// Build canonical Huffman codes from a length table. Returns (enc_tbl, dec_tbl, maxlen).
// Marked static inline to align with its definition and avoid multiple definition errors.
std::tuple<std::unordered_map<std::uint32_t, std::pair<std::uint32_t, int>>, std::unordered_map<int, std::unordered_map<std::uint32_t, std::uint32_t>>, int> huff_canonical( const std::unordered_map<std::uint32_t, int>& lengths );

// Additional helper function declarations for the new TOC container.
// These helpers implement run‑length encoding, canonical Huffman coding,
// Rice coding, ZigZag transforms and Elias–Fano position coding. They
// mirror the Python reference implementations exactly and are defined later
// in this file.


// === Round 3: Method‑ID Huffman (build) + RLE→Huff(id)+Rice(run_len‑1) encoder ==================

// Write and read Huffman‑coded symbols.
void					   huff_encode_symbols( BitWriter& bw, const std::unordered_map<std::uint32_t, std::pair<std::uint32_t, int>>& enc_tbl, const std::vector<std::uint32_t>& syms );
std::vector<std::uint32_t> huff_decode_symbols( BitReader& br, const std::unordered_map<int, std::unordered_map<std::uint32_t, std::uint32_t>>& dec_tbl, int maxlen, std::size_t nvals );

// Rice coding helpers: write a sequence of non‑negative integers and read a fixed count.
void									 rice_write_values( BitWriter& bw, const std::vector<std::uint32_t>& seq, int k );
std::vector<std::uint32_t> rice_read_n( BitReader& br, int k, std::size_t nvals );

// ZigZag encode/decode 32‑bit integers.
std::uint32_t zigzag_encode_32( std::int32_t x );
std::int32_t  zigzag_decode_32( std::uint32_t n );

// Elias–Fano helper: Choose parameter l for Elias–Fano given universe U and number of values n.
std::uint32_t ef_choose_l( std::uint32_t U, std::size_t n )
{
	if ( n == 0 || U <= 1u )
		return 0u;
	std::uint64_t avg = static_cast<std::uint64_t>( U ) / static_cast<std::uint64_t>( n );
	if ( avg <= 1u )
		return 0u;
	double lf = std::log2( static_cast<double>( avg ) );
	if ( lf < 0.0 )
		lf = 0.0;
	return static_cast<std::uint32_t>( std::floor( lf ) );
}

// === Round 4: Elias–Fano parameters + write/read for monotone positions =========================

// ==== Forward declarations for EFParams API (to unify EF usage) ====
struct EFParams;
EFParams ef_choose_params( std::uint32_t N, std::uint32_t M );

/**
 * @brief Elias–Fano parameters.
 * N: universe upper bound (max value in sequence, inclusive upper bound for ends), i.e., total_payload_bytes
 * M: number of values (sequence length), i.e., nblocks
 * L: number of low bits to store explicitly; chosen as floor(log2(N / M)) (standard near‑optimal choice)
 */
struct EFParams
{
	std::uint32_t N = 0;
	std::uint32_t M = 0;
	std::uint32_t L = 0;
};

/**
 * @brief Choose EF parameter L following the standard heuristic: L = floor(log2(N / M)), clamped to [0, 31].
 * If N < M (degenerate), use L = 0.
 */
EFParams ef_choose_params( std::uint32_t N, std::uint32_t M )
{
	EFParams ep;
	ep.N = N;
	ep.M = M;
	if ( M == 0 || N == 0 || N <= M )
	{
		ep.L = 0;
		return ep;
	}
	// Avoid floating point: find floor(log2(N/M)) by integer shifts.
	std::uint32_t q = N / M;
	std::uint32_t L = 0;
	while ( ( q >> ( L + 1 ) ) > 0 )
		++L;
	ep.L = ( L > 31u ) ? 31u : L;
	return ep;
}

/**
 * @brief Write Elias–Fano encoding of a strictly increasing sequence 'ends' using parameters 'ep'.
 * Layout:
 *  - High bits bitmap B of length (M + U), where U = ceil(N / 2^L).
 *	For each value x_i: hi_i = x_i >> L; set bit at position (hi_i + i) to 1.
 *  - Low bits array of L bits per value, concatenated MSB‑first.
 *
 * NOTE: This writer assumes 'ends' is strictly increasing and each 0 <= ends[i] <= N.
 */
void ef_write_positions( const std::vector<std::uint32_t>& ends, const EFParams& ep, BitWriter& bw );

/**
 * @brief Read Elias–Fano encoding back into values using parameters 'ep'.
 * Assumes the bitstream is positioned at the start of the high bitmap, followed immediately by lows.
 * Returns a vector of length M with strictly increasing values in [0, N].
 */
std::vector<std::uint32_t> ef_read_positions( BitReader& br, const EFParams& ep );

// -----------------------------
// Public types
// -----------------------------
using Boundary = std::pair<std::size_t, std::size_t>;
using BoundaryList = std::vector<Boundary>;
using ByteVector = std::vector<std::uint8_t>;
using MetaMap = std::unordered_map<std::string, std::string>;

// ============================================================
// Deterministic GEAR table (fixed seed, cross-platform stable)
// ============================================================
//
// make_gear(seed):
//   Produce a 256-entry 32-bit table via xorshift32 with the given seed.
//   Each entry is OR'ed with 1 to avoid zero values.
//
std::array<std::uint32_t, 256> make_gear( std::uint32_t seed = 0x243F6A88u ) noexcept;

// Global constant gear table constructed with the default seed.
// Matches Python: _GEAR = _make_gear(0x243F6A88)
extern const std::array<std::uint32_t, 256> GEAR;

// =========================================
// Helpers mirroring the Python functions
// =========================================
//
// clamp_mask_bits(avg_size):
//   If avg_size <= 0 -> return 6.
//   Else k = bit_length(avg_size) - 1, then clamp k into [6, 20].
//
std::uint32_t clamp_mask_bits( std::size_t avg_size ) noexcept;

// roll_gear(h, byte_val):
//   Return ((h << 1) & 0xFFFFFFFF) + GEAR[byte_val].
//
std::uint32_t roll_gear( std::uint32_t h, std::uint8_t byte_val ) noexcept;

// =========================================
// Fast Content defined chunking (FastCDC) (strict, non-recursive)
// =========================================
//
// cdc_fast_boundaries_strict(data, min_size, avg_size, max_size, merge_orphan_tail):
//   - Validate: 0 < min_size ≤ avg_size ≤ max_size, and avg_size ≥ 64.
//   - Choose mask bits k via clamp_mask_bits(avg_size); mask = (1<<k)-1.
//   - For each block: start at i, advance to end_min = min(n, start+min_size),
//	 then scan until end_max = min(n, start+max_size) looking for (h&mask)==0
//	 while rolling h by roll_gear. If not found, cut at end_max.
//   - Optionally merge a trailing "orphan tail" (< min_size) into the previous block.
//   - Return closed-open half-intervals [start, end).
//
BoundaryList cdc_fast_boundaries_strict( const std::vector<std::uint8_t>& data, std::size_t min_size = 4096, std::size_t avg_size = 8192, std::size_t max_size = 16384, bool merge_orphan_tail = true );

// =========================================
// Fixed-size chunking
// =========================================
//
// fixed_boundaries(data, block_size):
//   - Validate: block_size > 0.
//   - Return contiguous fixed-width [i, min(n, i+block_size)) segments.
//
BoundaryList fixed_boundaries( const std::vector<std::uint8_t>& data, std::size_t block_size = 8192 );

// Bijective Burrows–Wheeler transform and Lyndon factorisation
using RangeList = std::vector<std::pair<std::size_t, std::size_t>>;
RangeList				  duval_lyndon( const std::vector<std::uint8_t>& s );
std::vector<std::uint8_t> bbwt_forward( const std::vector<std::uint8_t>& s );
std::vector<std::uint8_t> bbwt_inverse( const std::vector<std::uint8_t>& L );

// Move‑to‑front coding
std::vector<std::uint8_t> mtf_encode( const std::vector<std::uint8_t>& data );
std::vector<std::uint8_t> mtf_decode( const std::vector<std::uint8_t>& seq );

// Bitwise reversible circuit modules
std::vector<std::uint8_t> bitplane_interleave( const std::vector<std::uint8_t>& data );
std::vector<std::uint8_t> bitplane_deinterleave( const std::vector<std::uint8_t>& data, std::size_t orig_len );
std::vector<std::uint8_t> lfsr_whiten( const std::vector<std::uint8_t>& data, std::uint8_t taps = 0b10010110, std::uint8_t seed = 1 );
std::vector<std::uint8_t> nibble_swap( const std::vector<std::uint8_t>& data );
std::vector<std::uint8_t> bit_reverse( const std::vector<std::uint8_t>& data );

// Boolean gate primitives and automata
std::uint8_t gate_and( std::uint8_t a, std::uint8_t b )
{
	return a & b;
}
std::uint8_t gate_or( std::uint8_t a, std::uint8_t b )
{
	return a | b;
}
std::uint8_t gate_not( std::uint8_t a, std::uint8_t width = 8 );
std::uint8_t gate_xor( std::uint8_t a, std::uint8_t b, std::uint8_t width = 8 );
// prefix_or_word and left_band helpers have been removed as they were unused.
std::uint8_t byte_eq( std::uint8_t a, std::uint8_t b );
std::uint8_t mux_byte( std::uint8_t m, std::uint8_t a, std::uint8_t b, std::uint8_t width = 8 );
std::uint8_t gray_pred( std::uint8_t v );

// Four automaton modes and selection
std::vector<std::uint8_t> mode1_forward( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t> mode1_inverse( const std::vector<std::uint8_t>& residual );
std::vector<std::uint8_t> mode2_forward( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t> mode2_inverse( const std::vector<std::uint8_t>& residual );
std::vector<std::uint8_t> mode3_forward( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t> mode3_inverse( const std::vector<std::uint8_t>& residual );
std::vector<std::uint8_t> mode4_forward( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t> mode4_inverse( const std::vector<std::uint8_t>& residual );

// Circuit automaton selector
struct AutomatonResult
{
	std::vector<std::uint8_t> data;
	std::uint8_t			  mode;
};
AutomatonResult			  circuit_map_automaton_forward( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t> circuit_map_automaton_inverse( const std::vector<std::uint8_t>& mapped, std::uint8_t mode );

// Helper functions for bit‑plane conversion and run‑length coding
std::tuple<std::vector<std::vector<int>>, std::size_t> bytes_to_bitplanes( const std::vector<std::uint8_t>& data );
std::vector<std::uint8_t>							   bitplanes_to_bytes( const std::vector<std::vector<int>>& planes );
double												   avg_run_bits( const std::vector<int>& bits );
double												   H0_bits( const std::vector<int>& bits );
std::pair<int, std::vector<int>>					   rle_binary( const std::vector<int>& bits );
std::vector<int>									   unrle_binary( int first_bit, const std::vector<int>& runs );
std::vector<std::uint8_t>							   pack_bits_to_bytes( const std::vector<int>& bits );
std::vector<int>									   unpack_bits_from_bytes( const std::vector<std::uint8_t>& buf, std::size_t nbits );

// Rice/Golomb coding for non‑negative integers
std::vector<std::uint8_t>  rice_encode( const std::vector<std::uint64_t>& seq, std::uint8_t k );
std::vector<std::uint64_t> rice_decode( const std::vector<std::uint8_t>& data, std::uint8_t k, std::size_t nvals );

// Gray code helpers
std::vector<std::uint8_t> gray_encode_bytes( const std::vector<std::uint8_t>& data );
std::vector<std::uint8_t> gray_decode_bytes( const std::vector<std::uint8_t>& data );

// Lempel–Ziv 1977 (LZ77) and Re‑Pair coders – declarations

std::unordered_map<std::pair<int, int>, int, PairHash> count_pairs( const std::vector<int>& seq );

std::pair<std::vector<int>, int>												   replace_non_overlapping( const std::vector<int>& seq, const std::pair<int, int>& target, int new_sym );
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> repair_compress( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t>														   repair_decompress( const std::vector<std::uint8_t>& data, std::size_t orig_len );

// Linear Feedback Shift Register (LFSR) predictor
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> encode_lfsr_predict( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t>														   decode_lfsr_predict( const std::vector<std::uint8_t>& data, std::size_t orig_len );

// Bijective Burrows–Wheeler Transform (BBWT) → Move‑to‑Front (MTF) → Rice coding model
// with optional bitwise modules
struct BBWTMeta
{
	std::uint32_t flags;	 // 0,1,4,8,16
	std::uint32_t k;		 // Rice parameter (fixed to 2 here)
	std::size_t	  length;	 // expected output length of transformed byte-seq
	std::size_t	  orig_len;	 // original block length (same as length here)
};
std::pair<std::vector<std::uint8_t>, BBWTMeta> encode_bbwt_mtf_rice( const std::vector<std::uint8_t>& block, bool use_bitplane = false, bool use_lfsr = false, bool use_nibble = false, bool use_bitrev = false, bool use_gray = false, std::uint8_t rice_param = 2 );
std::vector<std::uint8_t>					   decode_bbwt_mtf_rice( const std::vector<std::uint8_t>& payload, const BBWTMeta& meta );

// New pipeline (V2) using automaton + per‑plane BBWT (bijective Burrows–Wheeler Transform)
// + RLE (run‑length encoding) + Rice coding
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> encode_new_pipeline( const std::vector<std::uint8_t>& block );
std::vector<std::uint8_t>														   decode_new_pipeline( const std::vector<std::uint8_t>& payload, std::size_t orig_len, const std::unordered_map<std::string, std::string>& meta );

// Top level minimum description length (MDL) compressor and decompressor
// Using the highest bit of the block_size field as the mode bit (1 bit),
// and the remaining 31 bits to store size:
// - MODE_FIXED: size = block_size
// - MODE_CDC  : size = avg_size
std::uint32_t pack_mode_and_size( std::uint32_t mode, std::uint32_t size31 );
void		  unpack_mode_and_size( std::uint32_t word, std::uint32_t& mode, std::uint32_t& size31 );


// =======================
// Compression (CDC mode)
// =======================
// Compress (CDC variant; header field carries the mode bit)
ByteVector compress_blocks_cdc( const ByteVector& data, std::size_t min_size = 4096, std::size_t avg_size = 8192, std::size_t max_size = 16384 );

// ==========================
// Compression (fixed-size)
// ==========================
// Compress (fixed-size variant; header field carries the mode bit)
ByteVector compress_blocks_fixed( const ByteVector& data, std::size_t block_size = 8192 );

// =======================
// Decompression (compat)
// =======================
// Decompress (compatible with old containers; can read the mode bit)
ByteVector decompress( const ByteVector& container );

// -----------------------------------------
// Candidate encoders / decoders (ordering)
// Define candidate models for MDL selection. V2 removes invalid pipelines
// and introduces encode_new_pipeline. Each candidate returns (payload, meta);
// for each block the smallest payload is selected.
// Decoder list must align exactly with candidate encoders.
// Order:
//   0: raw
//   1: xor
//   2: bbwt
//   3: bbwt+bitplane
//   4: bbwt+nibble
//   5: bbwt+bitrev
//   6: bbwt+gray
//   7: lz77
//   8: lfsr predictor
//   9: re-pair
//   10: V2 new pipeline
// -----------------------------------------

// Minimal public aliases
using ByteVector = std::vector<std::uint8_t>;
using MetaMap = std::unordered_map<std::string, std::string>;

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
std::vector<std::uint8_t> uleb128_encode( std::uint64_t n )
{
	std::vector<std::uint8_t> out;
	// Repeatedly emit 7 bits per iteration; set high bit if more remain.
	while ( true )
	{
		std::uint8_t byte = n & 0x7F;
		n >>= 7;
		if ( n != 0 )
		{
			// More data remains: set continuation bit.
			out.push_back( static_cast<std::uint8_t>( byte | 0x80 ) );
		}
		else
		{
			out.push_back( byte );
			break;
		}
	}
	return out;
}

// ULEB128 decoding: decode a value from data starting at position pos.
std::pair<std::uint64_t, std::size_t> uleb128_decode_stream( const std::vector<std::uint8_t>& data, std::size_t pos )
{
	std::uint64_t result = 0;
	std::uint32_t shift = 0;
	std::size_t	  i = pos;
	while ( true )
	{
		if ( i >= data.size() )
		{
			throw std::runtime_error( "Truncated ULEB128" );
		}
		std::uint8_t b = data[ i++ ];
		result |= static_cast<std::uint64_t>( b & 0x7F ) << shift;
		if ( ( b & 0x80 ) == 0 )
		{
			break;
		}
		shift += 7;
	}
	return { result, i };
}

// ============================================================
// Deterministic GEAR table (xorshift32, OR 1 to avoid zeros)
// ============================================================
std::array<std::uint32_t, 256> make_gear( std::uint32_t seed ) noexcept
{
	std::array<std::uint32_t, 256> tbl {};
	std::uint32_t				   x = seed;
	for ( std::size_t i = 0; i < tbl.size(); ++i )
	{
		// xorshift32, identical bit-twiddling to the Python version
		x ^= ( x << 13 );
		x ^= ( x >> 17 );
		x ^= ( x << 5 );
		tbl[ i ] = ( x | 1u );	// ensure non-zero
	}
	return tbl;
}

// Global GEAR table equivalent to Python's _GEAR = _make_gear(0x243F6A88)
const std::array<std::uint32_t, 256> GEAR = make_gear( 0x243F6A88u );

// =========================================
// Helpers (1:1 with Python semantics)
// =========================================
std::uint32_t clamp_mask_bits( std::size_t avg_size ) noexcept
{
	if ( avg_size == 0 )
		return 6u;
	// bit_length(x) - 1 == floor(log2(x))
	unsigned k = ( std::bit_width( avg_size ) == 0 ? 0u : static_cast<unsigned>( std::bit_width( avg_size ) - 1 ) );
	if ( k < 6u )
		k = 6u;
	if ( k > 20u )
		k = 20u;
	return k;
}

std::uint32_t roll_gear( std::uint32_t h, std::uint8_t byte_val ) noexcept
{
	// ((h << 1) & 0xFFFFFFFF) + GEAR[byte_val]
	// Using uint32_t arithmetic naturally wraps mod 2^32.
	return static_cast<std::uint32_t>( ( h << 1 ) ) + GEAR[ byte_val ];
}

// =========================================
// FastCDC (strict, non-recursive)
// =========================================
BoundaryList cdc_fast_boundaries_strict( const std::vector<std::uint8_t>& data, std::size_t min_size, std::size_t avg_size, std::size_t max_size, bool merge_orphan_tail )
{
	const std::size_t n = data.size();
	if ( n == 0 )
		return {};

	// Validate constraints: 0 < min_size ≤ avg_size ≤ max_size and avg_size ≥ 64
	if ( !( min_size > 0 && min_size <= avg_size && avg_size <= max_size ) )
	{
		throw std::invalid_argument( "Require 0 < min_size \342\211\245 avg_size \342\211\245 max_size" );
	}
	if ( avg_size < 64 )
	{
		throw std::invalid_argument( "avg_size too small; use \342\211\245 64" );
	}

	const std::uint32_t k = clamp_mask_bits( avg_size );
	const std::uint32_t mask = ( 1u << k ) - 1u;

	BoundaryList boundaries;
	boundaries.reserve( std::max<std::size_t>( 1, n / std::max<std::size_t>( 1, avg_size ) ) );

	std::size_t i = 0;
	while ( i < n )
	{
		const std::size_t start = i;
		const std::size_t end_min = std::min( n, start + min_size );
		const std::size_t end_max = std::min( n, start + max_size );

		i = end_min;
		std::uint32_t h = 0;
		bool		  found = false;

		while ( i < end_max )
		{
			h = roll_gear( h, data[ i ] );
			if ( ( h & mask ) == 0u )
			{
				++i;
				found = true;
				break;
			}
			++i;
		}
		if ( !found )
		{
			i = end_max;  // forced split
		}
		boundaries.emplace_back( start, i );
	}

	// Merge orphan tail if enabled and the last block is smaller than min_size
	if ( merge_orphan_tail && boundaries.size() >= 2 )
	{
		const auto [ last_s, last_e ] = boundaries.back();
		if ( ( last_e - last_s ) < min_size )
		{
			auto& prev = boundaries[ boundaries.size() - 2 ];
			prev.second = last_e;
			boundaries.pop_back();
		}
	}

	// Sanity: first block starts at 0 and final block ends at n
	assert( !boundaries.empty() );
	assert( boundaries.front().first == 0 );
	assert( boundaries.back().second == n );

	return boundaries;
}

// =========================================
// Fixed-size chunking
// =========================================
BoundaryList fixed_boundaries( const std::vector<std::uint8_t>& data, std::size_t block_size )
{
	const std::size_t n = data.size();
	if ( n == 0 )
		return {};
	if ( block_size == 0 )
	{
		throw std::invalid_argument( "block_size must be positive" );
	}

	BoundaryList out;
	out.reserve( ( n + block_size - 1 ) / block_size );
	for ( std::size_t i = 0; i < n; i += block_size )
	{
		out.emplace_back( i, std::min( n, i + block_size ) );
	}
	return out;
}

// Duval's Lyndon factorisation: decompose the input into a sequence of
// non‑increasing Lyndon words.  Each pair represents the half‑open interval
// [start, end) of the factor within the original string.  The algorithm is
// linear in the length of the input.
RangeList duval_lyndon( const std::vector<std::uint8_t>& s )
{
	std::size_t n = s.size();
	RangeList	out;
	std::size_t i = 0;
	while ( i < n )
	{
		std::size_t j = i + 1;
		std::size_t k = i;
		// Determine the end of the smallest Lyndon word starting at i.
		while ( j < n && s[ k ] <= s[ j ] )
		{
			if ( s[ k ] < s[ j ] )
			{
				k = i;
			}
			else
			{
				++k;
			}
			++j;
		}
		std::size_t p = j - k;
		// Output repeats of this factor as long as they fit.
		while ( i <= k )
		{
			out.emplace_back( i, i + p );
			i += p;
		}
	}
	return out;
}

// Construct suffix array using prefix doubling.  Because the input size for
// each factor (up to 2*m for a factor of length m) is small compared to a
// typical block, the O(n log n) complexity is acceptable.  Returns a vector
// of starting positions sorted lexicographically.
static std::vector<std::size_t> sa_prefix_doubling( const std::vector<std::uint8_t>& t )
{
	std::size_t n = t.size();
	if ( n == 0 )
	{
		return {};
	}
	std::vector<std::size_t> idx( n );
	std::iota( idx.begin(), idx.end(), 0 );
	std::vector<int> rank( n );
	for ( std::size_t i = 0; i < n; ++i )
	{
		rank[ i ] = static_cast<int>( t[ i ] );
	}
	std::vector<int> tmp( n );
	std::size_t		 k = 1;
	while ( true )
	{
		// Sort by (rank[i], rank[i+k]) using lambda comparator.
		std::sort( idx.begin(), idx.end(), [ & ]( std::size_t a, std::size_t b ) {
			int ra = rank[ a ];
			int rb = rank[ b ];
			int ra_k = ( a + k < n ) ? rank[ a + k ] : -1;
			int rb_k = ( b + k < n ) ? rank[ b + k ] : -1;
			if ( ra != rb )
				return ra < rb;
			return ra_k < rb_k;
		} );
		// Recompute temporary ranks.
		tmp[ idx[ 0 ] ] = 0;
		for ( std::size_t i = 1; i < n; ++i )
		{
			std::size_t a = idx[ i - 1 ];
			std::size_t b = idx[ i ];
			int			ra = rank[ a ];
			int			rb = rank[ b ];
			int			ra_k = ( a + k < n ) ? rank[ a + k ] : -1;
			int			rb_k = ( b + k < n ) ? rank[ b + k ] : -1;
			tmp[ b ] = tmp[ a ] + ( ( ra != rb || ra_k != rb_k ) ? 1 : 0 );
		}
		rank = tmp;
		if ( rank[ idx.back() ] == static_cast<int>( n ) - 1 )
		{
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
std::vector<std::uint8_t> bbwt_forward( const std::vector<std::uint8_t>& s )
{
	if ( s.empty() )
	{
		return {};
	}
	RangeList facs = duval_lyndon( s );
	// Compute rotation order for each factor using doubling suffix array.
	struct FactorData
	{
		std::vector<std::uint8_t> word;
		std::vector<std::size_t>  order;
	};
	std::vector<FactorData> factors;
	factors.reserve( facs.size() );
	for ( auto [ a, b ] : facs )
	{
		std::vector<std::uint8_t> w( s.begin() + a, s.begin() + b );
		std::size_t				  m = w.size();
		// Double the factor to model cyclic rotations.
		std::vector<std::uint8_t> ww = w;
		ww.insert( ww.end(), w.begin(), w.end() );
		std::vector<std::size_t> sa = sa_prefix_doubling( ww );
		// Select starting positions less than m to obtain rotation order.
		std::vector<std::size_t> rot_order;
		rot_order.reserve( m );
		for ( std::size_t p : sa )
		{
			if ( p < m )
				rot_order.push_back( p );
		}
		factors.push_back( { std::move( w ), std::move( rot_order ) } );
	}
	// Node used in the heap merge compares two rotations lexicographically.
	struct Node
	{
		std::size_t						 fi;  // index of factor
		std::size_t						 k;	  // index within rotation order
		const std::vector<std::uint8_t>* w;
		const std::vector<std::size_t>*	 order;
	};
	// Custom comparator for std::priority_queue (min‑heap).  The Python
	// implementation compares up to m+n bytes of two rotations; here we
	// implement a deterministic ordering consistent with the original code.
	auto cmp = []( const Node& a, const Node& b ) {
		const auto& wA = *a.w;
		const auto& wB = *b.w;
		std::size_t i = ( *a.order )[ a.k ];
		std::size_t j = ( *b.order )[ b.k ];
		std::size_t m = wA.size();
		std::size_t n = wB.size();
		// Compare rotations u[i:] + u[:i] and v[j:] + v[:j].  We only need
		// to inspect up to m+n bytes to distinguish them.
		std::size_t p = 0;
		while ( p < m + n )
		{
			std::uint8_t cu = wA[ ( i + p ) % m ];
			std::uint8_t cv = wB[ ( j + p ) % n ];
			if ( cu != cv )
				return cu > cv;	 // reverse for min‑heap
			++p;
		}
		// Ties broken by factor index and rotation index.
		return std::tie( a.fi, i ) > std::tie( b.fi, j );
	};
	// Build initial heap: one node per factor at k=0.
	std::vector<Node> heap;
	heap.reserve( factors.size() );
	for ( std::size_t fi = 0; fi < factors.size(); ++fi )
	{
		if ( !factors[ fi ].order.empty() )
		{
			Node nd { fi, 0, &factors[ fi ].word, &factors[ fi ].order };
			heap.push_back( nd );
		}
	}
	// Convert vector into a heap.
	std::make_heap( heap.begin(), heap.end(), cmp );
	std::vector<std::uint8_t> out;
	out.reserve( s.size() );
	while ( !heap.empty() )
	{
		std::pop_heap( heap.begin(), heap.end(), cmp );
		Node nd = heap.back();
		heap.pop_back();
		std::size_t i = ( *nd.order )[ nd.k ];
		const auto& w = *nd.w;
		std::size_t m = w.size();
		// Append the character preceding the rotation start (cyclically).
		out.push_back( w[ ( i + m - 1 ) % m ] );
		// Advance within this factor; push back into heap if more rotations remain.
		nd.k += 1;
		if ( nd.k < nd.order->size() )
		{
			std::push_heap( heap.begin(), heap.end(), cmp );
			heap.push_back( nd );
			std::push_heap( heap.begin(), heap.end(), cmp );
		}
	}
	return out;
}

// Invert the bijective BWT.  The inverse recovers factors by computing
// permutation cycles over the sorted indices of L.  The method follows the
// algorithm described in the Python code and reconstructs the original
// concatenation of Lyndon words in reverse order.
std::vector<std::uint8_t> bbwt_inverse( const std::vector<std::uint8_t>& L )
{
	std::size_t n = L.size();
	if ( n == 0 )
		return {};
	// Sort indices by (L[idx], idx) to build the permutation.
	std::vector<std::size_t> order( n );
	std::iota( order.begin(), order.end(), 0 );
	std::stable_sort( order.begin(), order.end(), [ & ]( std::size_t a, std::size_t b ) { return std::make_pair( L[ a ], a ) < std::make_pair( L[ b ], b ); } );
	// pi[i] points to the next index in the cycle.
	std::vector<std::size_t>			  pi = order;
	std::vector<bool>					  seen( n, false );
	std::vector<std::vector<std::size_t>> cycles;
	for ( std::size_t i = 0; i < n; ++i )
	{
		if ( !seen[ i ] )
		{
			std::vector<std::size_t> cyc;
			std::size_t				 cur = i;
			while ( !seen[ cur ] )
			{
				seen[ cur ] = true;
				cyc.push_back( cur );
				cur = pi[ cur ];
			}
			cycles.push_back( std::move( cyc ) );
		}
	}
	// Sort cycles by the minimal index to preserve order.
	std::sort( cycles.begin(), cycles.end(), []( const auto& a, const auto& b ) { return *std::min_element( a.begin(), a.end() ) < *std::min_element( b.begin(), b.end() ); } );
	// Reconstruct factors; each cycle yields one Lyndon word.
	std::vector<std::vector<std::uint8_t>> factors;
	factors.reserve( cycles.size() );
	for ( const auto& cyc : cycles )
	{
		std::size_t				  i0 = *std::min_element( cyc.begin(), cyc.end() );
		std::size_t				  d = cyc.size();
		std::size_t				  cur = i0;
		std::vector<std::uint8_t> seq;
		seq.reserve( d );
		for ( std::size_t k = 0; k < d; ++k )
		{
			cur = pi[ cur ];
			seq.push_back( L[ cur ] );
		}
		factors.push_back( seq );
	}
	// Concatenate factors in reverse order.
	std::vector<std::uint8_t> out;
	for ( auto it = factors.rbegin(); it != factors.rend(); ++it )
	{
		out.insert( out.end(), it->begin(), it->end() );
	}
	return out;
}

// Move‑to‑front encoding: maintain a list of 256 symbols; for each input byte
// output its index and move it to the front.  Note that the output values
// may exceed 255 if the original data contains bytes ≥256, but here we
// restrict to 8‑bit bytes so indices lie in [0,255].
std::vector<std::uint8_t> mtf_encode( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> table( 256 );
	std::iota( table.begin(), table.end(), 0 );
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( std::uint8_t b : data )
	{
		// Find the index of b in the current table.
		auto		 it = std::find( table.begin(), table.end(), b );
		std::uint8_t idx = static_cast<std::uint8_t>( std::distance( table.begin(), it ) );
		out.push_back( idx );
		// Move b to the front.
		table.erase( it );
		table.insert( table.begin(), b );
	}
	return out;
}

// Move‑to‑front decoding: invert the MTF transform by maintaining the same
// table.  Each index refers to the current table; the corresponding symbol
// is output and moved to the front.
std::vector<std::uint8_t> mtf_decode( const std::vector<std::uint8_t>& seq )
{
	std::vector<std::uint8_t> table( 256 );
	std::iota( table.begin(), table.end(), 0 );
	std::vector<std::uint8_t> out;
	out.reserve( seq.size() );
	for ( std::uint8_t idx : seq )
	{
		std::uint8_t b = table[ idx ];
		out.push_back( b );
		table.erase( table.begin() + idx );
		table.insert( table.begin(), b );
	}
	return out;
}

// Bitplane interleaving: for each block of 8 bytes, emit 8 bytes whose
// positions group together the corresponding bitplane.  The transform is
// self‑inverting (applying it twice restores the original).
std::vector<std::uint8_t> bitplane_interleave( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	auto it = data.begin();
	while ( it != data.end() )
	{
		// Read up to 8 bytes (pad with zeros if fewer remain).
		std::array<std::uint8_t, 8> block {};
		std::size_t					count = 0;
		for ( ; count < 8 && it != data.end(); ++count, ++it )
		{
			block[ count ] = *it;
		}
		// Pad remaining entries with zeros.
		for ( std::size_t bit = 0; bit < 8; ++bit )
		{
			std::uint8_t v = 0;
			for ( std::size_t i = 0; i < 8; ++i )
			{
				std::uint8_t b = block[ i ];
				v |= ( ( b >> ( 7 - bit ) ) & 1u ) << ( 7 - i );
			}
			out.push_back( v );
		}
	}
	return out;
}

// Bitplane de‑interleaving: invert bitplane_interleave by reconstructing
// original bytes.  orig_len specifies the length of the original data before
// padding.  Blocks are processed in groups of 8 interleaved bytes.
std::vector<std::uint8_t> bitplane_deinterleave( const std::vector<std::uint8_t>& data, std::size_t orig_len )
{
	std::vector<std::uint8_t> out;
	out.reserve( orig_len );
	auto it = data.begin();
	while ( it != data.end() )
	{
		std::array<std::uint8_t, 8> block {};
		for ( std::size_t i = 0; i < 8 && it != data.end(); ++i, ++it )
		{
			block[ i ] = *it;
		}
		// Initialize eight output bytes to zero.
		std::array<std::uint8_t, 8> bytes {};
		for ( std::size_t bit = 0; bit < 8; ++bit )
		{
			std::uint8_t byte = block[ bit ];
			for ( std::size_t i = 0; i < 8; ++i )
			{
				bytes[ i ] |= ( ( byte >> ( 7 - i ) ) & 1u ) << ( 7 - bit );
			}
		}
		for ( std::size_t i = 0; i < 8; ++i )
		{
			out.push_back( bytes[ i ] );
		}
	}
	// Trim to original length (the last block may have contained padding).
	if ( out.size() > orig_len )
		out.resize( orig_len );
	return out;
}

// LFSR whitening: produce a pseudo‑random stream via an 8‑bit linear feedback
// shift register (LFSR) with the given taps and seed.  Each input byte is
// XORed with the current state; the same function applied twice yields the
// original sequence.  The taps and seed defaults match the Python code.
std::vector<std::uint8_t> lfsr_whiten( const std::vector<std::uint8_t>& data, std::uint8_t taps, std::uint8_t seed )
{
	std::uint8_t			  state = seed & 0xFFu;
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( std::uint8_t b : data )
	{
		out.push_back( static_cast<std::uint8_t>( b ^ state ) );
		// Compute feedback bit as XOR of tapped bits of state.
		std::uint8_t fb = 0;
		for ( int bit = 0; bit < 8; ++bit )
		{
			if ( ( taps >> bit ) & 1u )
			{
				fb ^= ( state >> bit ) & 1u;
			}
		}
		state = static_cast<std::uint8_t>( ( state << 1 ) | fb );
	}
	return out;
}

// Swap high and low 4‑bit nibbles of each byte.  Applying nibble_swap twice
// returns the original data.
std::vector<std::uint8_t> nibble_swap( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( std::uint8_t b : data )
	{
		std::uint8_t hi = ( b & 0xF0u ) >> 4;
		std::uint8_t lo = ( b & 0x0Fu ) << 4;
		out.push_back( static_cast<std::uint8_t>( hi | lo ) );
	}
	return out;
}

// Bit reversal lookup table for 8‑bit values.  Precompute once to speed up
// bit_reverse().
static const std::array<std::uint8_t, 256> BIT_REVERSE_TABLE = [] {
	std::array<std::uint8_t, 256> table {};
	for ( int i = 0; i < 256; ++i )
	{
		std::bitset<8> bs( i );
		// Reverse the bits and convert back to integer.
		std::bitset<8> reversed;
		for ( int k = 0; k < 8; ++k )
		{
			reversed[ 7 - k ] = bs[ k ];
		}
		table[ static_cast<std::size_t>( i ) ] = static_cast<std::uint8_t>( reversed.to_ulong() );
	}
	return table;
}();

// Reverse the bit order of each byte using the lookup table above.
std::vector<std::uint8_t> bit_reverse( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( std::uint8_t b : data )
	{
		out.push_back( BIT_REVERSE_TABLE[ b ] );
	}
	return out;
}

// Gate NOT limited to a specific bit‑width.  We mask the complement so that
// only the lowest width bits are retained (two's complement semantics).
std::uint8_t gate_not( std::uint8_t a, std::uint8_t width )
{
	std::uint8_t lane_mask = static_cast<std::uint8_t>( ( 1u << width ) - 1u );
	return static_cast<std::uint8_t>( ( ~a ) & lane_mask );
}

// Gate XOR expressed in terms of OR, AND and NOT.  We mask intermediate
// results to prevent overflow outside the active lane.  See Python comment
// for derivation: XOR = (a OR b) AND NOT(a AND b).
std::uint8_t gate_xor( std::uint8_t a, std::uint8_t b, std::uint8_t width )
{
	std::uint8_t or_result = static_cast<std::uint8_t>( a | b );
	std::uint8_t and_result = static_cast<std::uint8_t>( a & b );
	std::uint8_t not_and = gate_not( and_result, width );
	return static_cast<std::uint8_t>( or_result & not_and );
}

// prefix_or_word and left_band implementations have been removed because they were unused.

// Branch‑free byte equality: returns 0xFF if a == b else 0x00.  Based on
// gate logic: compute x = a XOR b, reduce to a single bit any_one, invert
// this bit to get equality and replicate across the lane.
std::uint8_t byte_eq( std::uint8_t a, std::uint8_t b )
{
	std::uint8_t width = 8;
	std::uint8_t lane_mask = static_cast<std::uint8_t>( ( 1u << width ) - 1u );
	// x = a XOR b within the lane.
	std::uint8_t x = gate_xor( static_cast<std::uint8_t>( a & lane_mask ), static_cast<std::uint8_t>( b & lane_mask ), width );
	// OR‑fold to detect any set bit.
	std::uint8_t x1 = static_cast<std::uint8_t>( x | ( ( x >> 4 ) & 0x0Fu ) );
	x1 = static_cast<std::uint8_t>( x1 | ( ( x1 >> 2 ) & 0x3Fu ) );
	x1 = static_cast<std::uint8_t>( x1 | ( ( x1 >> 1 ) & 0x7Fu ) );
	std::uint8_t any_one = static_cast<std::uint8_t>( x1 & 1u );
	// eq_bit = NOT(any_one) in a 1‑bit lane.
	std::uint8_t eq_bit = static_cast<std::uint8_t>( gate_not( any_one, 1 ) & 1u );
	// Replicate eq_bit across all bits of a byte.
	std::uint8_t mask = 0;
	for ( int i = 0; i < 8; ++i )
	{
		mask |= static_cast<std::uint8_t>( ( eq_bit << i ) & 0xFFu );
	}
	return mask;
}

// 2:1 multiplexer on a byte lane.  Canonical masks are 0xFF (select a) or
// 0x00 (select b).  Non‑canonical masks are supported but may blend bits.
std::uint8_t mux_byte( std::uint8_t m, std::uint8_t a, std::uint8_t b, std::uint8_t width )
{
	return static_cast<std::uint8_t>( ( ( m & a ) | ( gate_not( m, width ) & b ) ) & ( ( 1u << width ) - 1u ) );
}

// Gray predictor: compute g = v XOR (v >> 1).  Use gate_xor to preserve lane.
std::uint8_t gray_pred( std::uint8_t v )
{
	return gate_xor( v, static_cast<std::uint8_t>( v >> 1 ), 8 );
}

// Mode 1 forward: order‑1 residual (y[i] = x[i] XOR x[i‑1]).  The first
// symbol seeds the recursion.
std::vector<std::uint8_t> mode1_forward( const std::vector<std::uint8_t>& block )
{
	if ( block.empty() )
		return {};
	std::size_t				  length = block.size();
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = block[ 0 ];
	for ( std::size_t i = 1; i < length; ++i )
	{
		out[ i ] = gate_xor( block[ i ], block[ i - 1 ], 8 );
	}
	return out;
}

// Mode 1 inverse: reconstruct x from residuals and history.
std::vector<std::uint8_t> mode1_inverse( const std::vector<std::uint8_t>& residual )
{
	if ( residual.empty() )
		return {};
	std::size_t				  length = residual.size();
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = residual[ 0 ];
	for ( std::size_t i = 1; i < length; ++i )
	{
		out[ i ] = gate_xor( residual[ i ], out[ i - 1 ], 8 );
	}
	return out;
}

// Mode 2 forward: Gray predictor residual (y[i] = x[i] XOR Gray(x[i‑1])).
std::vector<std::uint8_t> mode2_forward( const std::vector<std::uint8_t>& block )
{
	if ( block.empty() )
		return {};
	std::size_t				  length = block.size();
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = block[ 0 ];
	for ( std::size_t i = 1; i < length; ++i )
	{
		std::uint8_t predictor = gray_pred( block[ i - 1 ] );
		out[ i ] = gate_xor( block[ i ], predictor, 8 );
	}
	return out;
}

// Mode 2 inverse: recover original bytes using Gray predictor from decoded history.
std::vector<std::uint8_t> mode2_inverse( const std::vector<std::uint8_t>& residual )
{
	if ( residual.empty() )
		return {};
	std::size_t				  length = residual.size();
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = residual[ 0 ];
	for ( std::size_t i = 1; i < length; ++i )
	{
		std::uint8_t predictor = gray_pred( out[ i - 1 ] );
		out[ i ] = gate_xor( residual[ i ], predictor, 8 );
	}
	return out;
}

// Mode 3 forward: order‑2 residual: y[0] = x[0], y[1] = x[1] XOR x[0],
// for i≥2: y[i] = x[i] XOR x[i‑2].
std::vector<std::uint8_t> mode3_forward( const std::vector<std::uint8_t>& block )
{
	std::size_t length = block.size();
	if ( length == 0 )
		return {};
	if ( length == 1 )
		return { block[ 0 ] };
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = block[ 0 ];
	out[ 1 ] = gate_xor( block[ 1 ], block[ 0 ], 8 );
	for ( std::size_t i = 2; i < length; ++i )
	{
		out[ i ] = gate_xor( block[ i ], block[ i - 2 ], 8 );
	}
	return out;
}

// Mode 3 inverse: invert order‑2 residual by reconstructing from history.
std::vector<std::uint8_t> mode3_inverse( const std::vector<std::uint8_t>& residual )
{
	std::size_t length = residual.size();
	if ( length == 0 )
		return {};
	if ( length == 1 )
		return { residual[ 0 ] };
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = residual[ 0 ];
	out[ 1 ] = gate_xor( residual[ 1 ], out[ 0 ], 8 );
	for ( std::size_t i = 2; i < length; ++i )
	{
		out[ i ] = gate_xor( residual[ i ], out[ i - 2 ], 8 );
	}
	return out;
}

// Mode 4 forward: run‑segment selector.  For i>=2 choose predictor = x[i‑1]
// (if x[i‑1]==x[i‑2]) else Gray(x[i‑1]).  Residual is XOR with predictor.
std::vector<std::uint8_t> mode4_forward( const std::vector<std::uint8_t>& block )
{
	std::size_t length = block.size();
	if ( length == 0 )
		return {};
	if ( length == 1 )
		return { block[ 0 ] };
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = block[ 0 ];
	out[ 1 ] = gate_xor( block[ 1 ], block[ 0 ], 8 );
	for ( std::size_t i = 2; i < length; ++i )
	{
		std::uint8_t selector_mask = byte_eq( block[ i - 1 ], block[ i - 2 ] );
		std::uint8_t pred_run = block[ i - 1 ];
		std::uint8_t pred_alt = gray_pred( block[ i - 1 ] );
		std::uint8_t predictor = mux_byte( selector_mask, pred_run, pred_alt );
		out[ i ] = gate_xor( block[ i ], predictor, 8 );
	}
	return out;
}

// Mode 4 inverse: recompute predictor from already decoded bytes.
std::vector<std::uint8_t> mode4_inverse( const std::vector<std::uint8_t>& residual )
{
	std::size_t length = residual.size();
	if ( length == 0 )
		return {};
	if ( length == 1 )
		return { residual[ 0 ] };
	std::vector<std::uint8_t> out( length );
	out[ 0 ] = residual[ 0 ];
	out[ 1 ] = gate_xor( residual[ 1 ], out[ 0 ], 8 );
	for ( std::size_t i = 2; i < length; ++i )
	{
		std::uint8_t selector_mask = byte_eq( out[ i - 1 ], out[ i - 2 ] );
		std::uint8_t pred_run = out[ i - 1 ];
		std::uint8_t pred_alt = gray_pred( out[ i - 1 ] );
		std::uint8_t predictor = mux_byte( selector_mask, pred_run, pred_alt );
		out[ i ] = gate_xor( residual[ i ], predictor, 8 );
	}
	return out;
}

// Approximate first‑order (Markov) bit entropy function removed in V2.
// The V2 pipeline does not rely on entropy heuristics when selecting
// transforms.  The implementation of first_order_bit_entropy is retained
// below but disabled via #if 0 to document the previous approach.
#if 0
double first_order_bit_entropy( const std::vector<std::uint8_t>& block )
{
	std::size_t nbits = block.size() * 8;
	if ( nbits < 2 )
		return 1.0;
	// counts[x][y] counts transitions from bit x to bit y.
	std::array<std::array<std::uint64_t, 2>, 2> counts {};
	// Initialize prev bit to the MSB of first byte.
	std::uint8_t prev = ( block[ 0 ] >> 7 ) & 1u;
	std::size_t bit_idx = 1;
	bool skip_first = true;
	for ( std::uint8_t b : block )
	{
		for ( int k = 0; k < 8; ++k )
		{
			if ( skip_first )
			{
				// Skip the very first bit; it has no predecessor.
				skip_first = false;
				continue;
			}
			std::uint8_t bit = ( b >> ( 7 - k ) ) & 1u;
			counts[ prev ][ bit ] += 1;
			prev = bit;
			++bit_idx;
		}
	}
	double total_trans = static_cast<double>( nbits - 1 );
	double H = 0.0;
	for ( int x = 0; x <= 1; ++x )
	{
		double px = static_cast<double>( counts[ x ][ 0 ] + counts[ x ][ 1 ] );
		if ( px == 0 )
			continue;
		for ( int y = 0; y <= 1; ++y )
		{
			double pxy = static_cast<double>( counts[ x ][ y ] );
			if ( pxy == 0 )
				continue;
			double p_y_given_x = pxy / px;
			H -= ( pxy / total_trans ) * std::log2( p_y_given_x );
		}
	}
	return H;
}
#endif

// Compute 0th‑order entropy of a byte array.  Used by the automaton selector.
double zero_order_entropy( const std::vector<std::uint8_t>& data )
{
	if ( data.empty() )
		return 0.0;
	std::array<std::uint64_t, 256> counts {};
	for ( auto b : data )
		counts[ b ]++;
	double h = 0.0;
	double n = static_cast<double>( data.size() );
	for ( auto c : counts )
	{
		if ( c == 0 )
			continue;
		double p = static_cast<double>( c ) / n;
		h -= p * std::log2( p );
	}
	return h;
}

// Select the best reversible automaton by minimum 0th‑order entropy.  Modes
// considered: identity (0), mode1, mode2, mode3, mode4.  Ties favour
// smaller mode index to ensure deterministic behaviour.  Returns both
// transformed data and the chosen mode identifier.
AutomatonResult circuit_map_automaton_forward( const std::vector<std::uint8_t>& block )
{
	double					  best_entropy = zero_order_entropy( block );
	std::uint8_t			  best_mode = 0;
	std::vector<std::uint8_t> best_bytes = block;
	// Candidates for modes 1..4.
	std::vector<std::pair<std::uint8_t, std::vector<std::uint8_t>>> candidates;
	candidates.emplace_back( 1, mode1_forward( block ) );
	candidates.emplace_back( 2, mode2_forward( block ) );
	candidates.emplace_back( 3, mode3_forward( block ) );
	candidates.emplace_back( 4, mode4_forward( block ) );
	for ( const auto& [ mode_id, mapped ] : candidates )
	{
		double ent = zero_order_entropy( mapped );
		if ( ent < best_entropy - 1e-9 || ( std::abs( ent - best_entropy ) <= 1e-9 && mode_id < best_mode ) )
		{
			best_entropy = ent;
			best_mode = mode_id;
			best_bytes = mapped;
		}
	}
	return { best_bytes, best_mode };
}

// Inverse of the selected automaton.  Given the mode id, dispatch to the
// appropriate inverse function.  Unknown modes fall back to identity.
std::vector<std::uint8_t> circuit_map_automaton_inverse( const std::vector<std::uint8_t>& mapped, std::uint8_t mode )
{
	switch ( mode )
	{
	case 0:
		return mapped;
	case 1:
		return mode1_inverse( mapped );
	case 2:
		return mode2_inverse( mapped );
	case 3:
		return mode3_inverse( mapped );
	case 4:
		return mode4_inverse( mapped );
	default:
		return mapped;
	}
}

// Split bytes -> 8 MSB-first bit-planes.
// planes[j][t] is bit j (0..7, MSB-first) of data[t] as 0/1.
std::tuple<std::vector<std::vector<int>>, std::size_t> bytes_to_bitplanes( const std::vector<std::uint8_t>& data )
{
	std::size_t					  L = data.size();
	std::vector<std::vector<int>> planes( 8, std::vector<int>( L ) );
	for ( std::size_t t = 0; t < L; ++t )
	{
		std::uint8_t b = data[ t ];
		// MSB-first: j=0 => bit7, j=7 => bit0
		planes[ 0 ][ t ] = ( b >> 7 ) & 1;
		planes[ 1 ][ t ] = ( b >> 6 ) & 1;
		planes[ 2 ][ t ] = ( b >> 5 ) & 1;
		planes[ 3 ][ t ] = ( b >> 4 ) & 1;
		planes[ 4 ][ t ] = ( b >> 3 ) & 1;
		planes[ 5 ][ t ] = ( b >> 2 ) & 1;
		planes[ 6 ][ t ] = ( b >> 1 ) & 1;
		planes[ 7 ][ t ] = ( b >> 0 ) & 1;
	}
	return { planes, L };
}

// Reconstruct bytes from 8 MSB-first bit-planes.
// Expects exactly 8 planes and equal lengths.
std::vector<std::uint8_t> bitplanes_to_bytes( const std::vector<std::vector<int>>& planes )
{
	if ( planes.empty() )
		return {};
	assert( planes.size() == 8 && "bitplanes_to_bytes expects 8 planes" );
	std::size_t L = planes[ 0 ].size();
	for ( std::size_t j = 1; j < 8; ++j )
	{
		assert( planes[ j ].size() == L && "all planes must have same length" );
	}

	std::vector<std::uint8_t> out( L );
	for ( std::size_t t = 0; t < L; ++t )
	{
		std::uint8_t val = static_cast<std::uint8_t>( ( ( planes[ 0 ][ t ] & 1 ) << 7 ) | ( ( planes[ 1 ][ t ] & 1 ) << 6 ) | ( ( planes[ 2 ][ t ] & 1 ) << 5 ) | ( ( planes[ 3 ][ t ] & 1 ) << 4 ) | ( ( planes[ 4 ][ t ] & 1 ) << 3 ) | ( ( planes[ 5 ][ t ] & 1 ) << 2 ) | ( ( planes[ 6 ][ t ] & 1 ) << 1 ) | ( ( planes[ 7 ][ t ] & 1 ) << 0 ) );
		out[ t ] = val;
	}
	return out;
}

// Average run length for a binary sequence.  Count number of runs and divide
// length by run count.
double avg_run_bits( const std::vector<int>& bits )
{
#if 0
	// Legacy implementation retained for reference.  Computes the average run
	// length of a binary sequence by counting transitions.
	if ( bits.empty() )
		return 0.0;
	std::size_t runs = 1;
	int prev = bits[ 0 ];
	for ( std::size_t i = 1; i < bits.size(); ++i )
	{
		if ( bits[ i ] != prev )
		{
			runs++;
			prev = bits[ i ];
		}
	}
	return static_cast<double>( bits.size() ) / runs;
#endif
	(void)bits; // suppress unused parameter warning in active builds
	return 0.0;
}

// 0th‑order entropy of a binary sequence.  Uses natural log base 2.
double H0_bits( const std::vector<int>& bits )
{
#if 0
	// Legacy implementation retained for reference.  Computes the zero‑order
	// binary entropy (base‑2) using the proportion of ones.
	if ( bits.empty() )
		return 0.0;
	std::size_t n = bits.size();
	std::size_t c1 = std::count( bits.begin(), bits.end(), 1 );
	double p = static_cast<double>( c1 ) / static_cast<double>( n );
	if ( p == 0.0 || p == 1.0 )
		return 0.0;
	return -p * std::log2( p ) - ( 1.0 - p ) * std::log2( 1.0 - p );
#endif
	(void)bits; // suppress unused parameter warning in active builds
	return 0.0;
}

// Run‑length encode a binary sequence: returns first bit and a list of run
// lengths.  Consecutive identical bits are grouped.
std::pair<int, std::vector<int>> rle_binary( const std::vector<int>& bits )
{
	if ( bits.empty() )
		return { 0, {} };
	std::vector<int> runs;
	runs.reserve( bits.size() );
	int cur = 1;
	for ( std::size_t i = 1; i < bits.size(); ++i )
	{
		if ( bits[ i ] == bits[ i - 1 ] )
		{
			cur++;
		}
		else
		{
			runs.push_back( cur );
			cur = 1;
		}
	}
	runs.push_back( cur );
	return { bits[ 0 ], runs };
}

// Inverse of rle_binary: reconstruct bits given first bit and run lengths.
std::vector<int> unrle_binary( int first_bit, const std::vector<int>& runs )
{
	std::vector<int> out;
	out.reserve( std::accumulate( runs.begin(), runs.end(), 0 ) );
	int b = first_bit & 1;
	for ( int r : runs )
	{
		out.insert( out.end(), r, b );
		b ^= 1;
	}
	return out;
}

// Pack a binary vector into a byte array.  Bits are filled MSB first in
// each byte.  The number of output bytes is (len(bits)+7)//8.
std::vector<std::uint8_t> pack_bits_to_bytes( const std::vector<int>& bits )
{
	std::size_t				  n = bits.size();
	std::vector<std::uint8_t> out( ( n + 7 ) / 8 );
	for ( std::size_t i = 0; i < n; ++i )
	{
		if ( bits[ i ] & 1 )
		{
			out[ i >> 3 ] |= static_cast<std::uint8_t>( 1u << ( 7 - ( i & 7 ) ) );
		}
	}
	return out;
}

// Unpack a bit vector from a byte buffer.  Only the first nbits bits are
// produced; trailing bits are ignored.
std::vector<int> unpack_bits_from_bytes( const std::vector<std::uint8_t>& buf, std::size_t nbits )
{
	std::vector<int> out( nbits );
	for ( std::size_t i = 0; i < nbits; ++i )
	{
		out[ i ] = ( ( buf[ i >> 3 ] >> ( 7 - ( i & 7 ) ) ) & 1u ) != 0;
	}
	return out;
}

// Rice encoding of a list of non‑negative integers.  Parameter k selects
// 2^k as the modulus.  Unary quotient followed by fixed k‑bit remainder.
std::vector<std::uint8_t> rice_encode( const std::vector<std::uint64_t>& seq, std::uint8_t k )
{
	std::string bitstr;
	bitstr.reserve( seq.size() * 8 );
	std::uint64_t M = static_cast<std::uint64_t>( 1 ) << k;
	for ( auto n : seq )
	{
		std::uint64_t q = n / M;
		std::uint64_t r = n % M;
		// Append q ones followed by a zero.
		bitstr.append( q, '1' );
		bitstr.push_back( '0' );
		// Append remainder bits if k>0.
		if ( k > 0 )
		{
			for ( int bit = k - 1; bit >= 0; --bit )
			{
				bitstr.push_back( ( ( r >> bit ) & 1u ) ? '1' : '0' );
			}
		}
	}
	// Pad bitstring to byte boundary with zeros.
	std::size_t pad = ( 8 - ( bitstr.size() % 8 ) ) % 8;
	bitstr.append( pad, '0' );
	// Pack bits into bytes.
	std::vector<std::uint8_t> out;
	out.reserve( bitstr.size() / 8 );
	for ( std::size_t i = 0; i < bitstr.size(); i += 8 )
	{
		std::uint8_t b = 0;
		for ( int j = 0; j < 8; ++j )
		{
			b = static_cast<std::uint8_t>( ( b << 1 ) | ( bitstr[ i + j ] == '1' ? 1u : 0u ) );
		}
		out.push_back( b );
	}
	return out;
}

// Rice decoding: decode nvals integers from payload using parameter k.  The
// bitstring is interpreted in order; if insufficient bits remain, an error
// is thrown.  Works for k>=0.
std::vector<std::uint64_t> rice_decode( const std::vector<std::uint8_t>& data, std::uint8_t k, std::size_t nvals )
{
	// Convert bytes to a bitstring representation.
	std::string bitstr;
	bitstr.reserve( data.size() * 8 );
	for ( auto b : data )
	{
		for ( int bit = 7; bit >= 0; --bit )
		{
			bitstr.push_back( ( ( b >> bit ) & 1u ) ? '1' : '0' );
		}
	}
	std::size_t				   i = 0;
	std::uint64_t			   M = static_cast<std::uint64_t>( 1 ) << k;
	std::vector<std::uint64_t> out;
	out.reserve( nvals );
	auto need = [ & ]( std::size_t bits ) {
		return i + bits <= bitstr.size();
	};
	for ( std::size_t v = 0; v < nvals; ++v )
	{
		// Read unary quotient q: count ones until first zero.
		std::uint64_t q = 0;
		while ( true )
		{
			if ( !need( 1 ) )
			{
				throw std::runtime_error( "Rice stream truncated while reading unary part" );
			}
			if ( bitstr[ i ] == '1' )
			{
				q++;
				i++;
			}
			else
			{
				i++;
				break;
			}
		}
		// Read remainder r of k bits.
		std::uint64_t r = 0;
		if ( k > 0 )
		{
			if ( !need( k ) )
			{
				throw std::runtime_error( "Rice stream truncated while reading remainder" );
			}
			for ( std::uint8_t bit = 0; bit < k; ++bit )
			{
				r = ( r << 1 ) | ( bitstr[ i + bit ] == '1' ? 1u : 0u );
			}
			i += k;
		}
		out.push_back( q * M + r );
	}
	return out;
}

// Gray encode each byte: g = x ^ (x>>1).
std::vector<std::uint8_t> gray_encode_bytes( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( auto b : data )
	{
		out.push_back( static_cast<std::uint8_t>( ( b ^ ( b >> 1 ) ) & 0xFFu ) );
	}
	return out;
}

// Gray decode bytes by iterative XOR with right shifts.  See Python for details.
std::vector<std::uint8_t> gray_decode_bytes( const std::vector<std::uint8_t>& data )
{
	std::vector<std::uint8_t> out;
	out.reserve( data.size() );
	for ( auto g : data )
	{
		std::uint8_t n = g;
		n ^= static_cast<std::uint8_t>( n >> 1 );
		n ^= static_cast<std::uint8_t>( n >> 2 );
		n ^= static_cast<std::uint8_t>( n >> 4 );
		out.push_back( static_cast<std::uint8_t>( n & 0xFFu ) );
	}
	return out;
}

// 允许重叠；window = 已产生的历史（最多 4096），block = 全部输入
std::size_t lz77_match_len_overlap( std::span<const std::uint8_t> window, std::span<const std::uint8_t> block, std::size_t pos, std::size_t dist, std::size_t max_total )
{
	const std::size_t win_size = window.size();
	std::size_t		  matched = 0;

	// ---- Phase 1: 仍需从“窗口尾部”取参考字节的这段（matched < dist）----
	// 这段可以一次性看成 window 的最后 dist 个字节与 block[pos..] 的对比
	if ( dist > 0 )
	{
		if ( dist > win_size )
			return 0;  // 调用约定下不该发生，保险
		const std::size_t can_cmp = std::min( dist, max_total - pos );
		auto			  tail = window.subspan( win_size - dist, dist );
		while ( matched < can_cmp )
		{
			if ( tail[ matched ] != block[ pos + matched ] )
				break;
			++matched;
		}
		// 若在 Phase 1 就不相等，则直接返回
		if ( matched < dist )
			return matched;
	}

	// ---- Phase 2: 完全进入“自我重叠复制”区间（matched >= dist）----
	// 参考来源：block[pos + matched - dist]
	while ( pos + matched < max_total )
	{
		const std::size_t src = pos + matched - dist;  // 因 matched>=dist，src 不会下溢
		if ( block[ src ] != block[ pos + matched ] )
			break;
		++matched;
	}
	return matched;
}

// ---- encoder ----
// Stream format (unchanged):
//  - Literal: [0][byte]
//  - Match  : [1][ULEB length][ULEB dist]
// Window keeps last 4096 bytes (erase from front when exceeded).
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> encode_lz77( const std::vector<std::uint8_t>& block )
{
	constexpr std::size_t WINDOW_MAX = 4096;
	constexpr std::size_t MIN_MATCH = 3;

	std::vector<std::uint8_t> window;
	window.reserve( WINDOW_MAX );

	std::vector<std::uint8_t> out;
	out.reserve( block.size() );

	const std::size_t			  n = block.size();
	std::span<const std::uint8_t> blk( block.data(), block.size() );

	std::size_t pos = 0;
	while ( pos < n )
	{
		std::size_t best_len = 0;
		std::size_t best_dist = 0;

		const std::size_t max_window = std::min<std::size_t>( window.size(), WINDOW_MAX );
		if ( max_window >= 1 )
		{
			// 取“窗口尾”的视图（不拷贝）
			const auto win_tail = std::span<const std::uint8_t>( window.data() + ( window.size() - max_window ), max_window );

			// 近到远地找
			for ( std::size_t dist = 1; dist <= max_window; ++dist )
			{
				const std::size_t m = lz77_match_len_overlap( win_tail, blk, pos, dist, n );
				if ( m > best_len )
				{
					best_len = m;
					best_dist = dist;
					// 若需要，可加早停阈值
					// if (best_len >= 258) break;
				}
			}
		}

		if ( best_len >= MIN_MATCH )
		{
			// 写 match
			out.push_back( 1 );
			const auto len_enc = uleb128_encode( static_cast<std::uint32_t>( best_len ) );
			const auto dist_enc = uleb128_encode( static_cast<std::uint32_t>( best_dist ) );
			out.insert( out.end(), len_enc.begin(), len_enc.end() );
			out.insert( out.end(), dist_enc.begin(), dist_enc.end() );

			// 前进并把匹配区域送入窗口
			for ( std::size_t k = 0; k < best_len; ++k )
				window.push_back( block[ pos + k ] );
			pos += best_len;
		}
		else
		{
			// 写 literal
			out.push_back( 0 );
			out.push_back( block[ pos ] );
			window.push_back( block[ pos ] );
			++pos;
		}

		// 修剪到 4096（仍旧一次性 erase；若想再提速，可换 deque/环形缓冲）
		if ( window.size() > WINDOW_MAX )
			window.erase( window.begin(), window.end() - WINDOW_MAX );
	}
	return { out, {} };
}

// ---- decoder ----
std::vector<std::uint8_t> decode_lz77( const std::vector<std::uint8_t>& data, std::size_t orig_len )
{
	constexpr std::size_t WINDOW_MAX = 4096;

	std::vector<std::uint8_t> window;
	window.reserve( WINDOW_MAX );

	std::vector<std::uint8_t> out;
	out.reserve( orig_len );

	std::span<const std::uint8_t> in( data.data(), data.size() );

	std::size_t i = 0;
	while ( i < in.size() && out.size() < orig_len )
	{
		const std::uint8_t flag = in[ i++ ];

		if ( flag == 0 )
		{
			if ( i >= in.size() )
				throw std::runtime_error( "LZ77 truncated literal" );
			const std::uint8_t b = in[ i++ ];
			out.push_back( b );
			window.push_back( b );
		}
		else if ( flag == 1 )
		{
			auto [ length, ni ] = uleb128_decode_stream( data, i );
			i = ni;
			auto [ dist, nj ] = uleb128_decode_stream( data, i );
			i = nj;

			if ( dist == 0 )
				throw std::runtime_error( "LZ77 invalid distance 0" );
			for ( std::size_t k = 0; k < static_cast<std::size_t>( length ); ++k )
			{
				if ( dist > window.size() )
					throw std::runtime_error( "LZ77 distance beyond window" );
				const std::uint8_t b = window[ window.size() - static_cast<std::size_t>( dist ) ];
				out.push_back( b );
				window.push_back( b );
				if ( out.size() == orig_len )
					break;
			}
		}
		else
		{
			throw std::runtime_error( "LZ77 unknown flag" );
		}

		if ( window.size() > WINDOW_MAX )
			window.erase( window.begin(), window.end() - WINDOW_MAX );
	}

	if ( out.size() != orig_len )
		throw std::runtime_error( "LZ77 output length mismatch" );
	return out;
}

// ---------- helpers: count pairs & replace non-overlapping ----------
inline std::unordered_map<std::pair<int, int>, int, PairHash> count_pairs( const std::vector<int>& seq )
{
	std::unordered_map<std::pair<int, int>, int, PairHash> freq;
	if ( seq.size() < 2 )
		return freq;
	for ( std::size_t i = 0; i + 1 < seq.size(); ++i )
	{
		auto p = std::make_pair( seq[ i ], seq[ i + 1 ] );
		++freq[ p ];
	}
	return freq;
}

// 返回 (新序列, 替换次数)
inline std::pair<std::vector<int>, int> replace_non_overlapping( const std::vector<int>& seq, const std::pair<int, int>& target, int new_sym )
{
	const int		 a = target.first, b = target.second;
	std::vector<int> out;
	out.reserve( seq.size() );
	int replaced = 0;
	for ( std::size_t i = 0; i < seq.size(); )
	{
		if ( i + 1 < seq.size() && seq[ i ] == a && seq[ i + 1 ] == b )
		{
			out.push_back( new_sym );
			i += 2;
			++replaced;
		}
		else
		{
			out.push_back( seq[ i ] );
			++i;
		}
	}
	return { std::move( out ), replaced };
}

// 递归+记忆化展开：与 Re-Pair SLP 定义一致（A -> XY），稳定可逆
static const std::vector<std::uint8_t>& expand_symbol( int sym, const std::unordered_map<int, std::pair<int, int>>& rules, std::unordered_map<int, std::vector<std::uint8_t>>& memo )
{
	if ( sym < 256 )
	{
		static std::array<std::vector<std::uint8_t>, 256> term_cache {};
		static std::once_flag							  once;
		std::call_once( once, [] {
			for ( int t = 0; t < 256; ++t )
				term_cache[ t ] = std::vector<std::uint8_t> { static_cast<std::uint8_t>( t ) };
		} );
		return term_cache[ sym ];
	}
	auto it = memo.find( sym );
	if ( it != memo.end() )
		return it->second;

	auto r = rules.find( sym );
	if ( r == rules.end() )
	{
		throw std::runtime_error( "RePair: nonterminal without rule" );
	}
	const auto& left = expand_symbol( r->second.first, rules, memo );
	const auto& right = expand_symbol( r->second.second, rules, memo );

	std::vector<std::uint8_t> buf;
	buf.reserve( left.size() + right.size() );
	buf.insert( buf.end(), left.begin(), left.end() );
	buf.insert( buf.end(), right.begin(), right.end() );

	auto [ ins, _ ] = memo.emplace( sym, std::move( buf ) );
	return ins->second;
}

// -----------------------------
// Strict Re-Pair compression (RP format, all ints ULEB128)
// -----------------------------
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> repair_compress( const std::vector<std::uint8_t>& block )
{
	// 空输入：直接写 RP, terminals=256, nrules=0, seqlen=0
	if ( block.empty() )
	{
		std::vector<std::uint8_t> out;
		out.push_back( 'R' );
		out.push_back( 'P' );			 // magic
		auto t = uleb128_encode( 256 );	 // terminals
		out.insert( out.end(), t.begin(), t.end() );
		auto z = uleb128_encode( 0 );  // nrules=0
		out.insert( out.end(), z.begin(), z.end() );
		out.insert( out.end(), z.begin(), z.end() );  // seq_len=0
		std::unordered_map<std::string, std::string> meta { { "rules", "0" }, { "final_len", "0" }, { "terminals", "256" }, { "nrules", "0" } };
		return { out, meta };
	}

	std::vector<int>				 seq( block.begin(), block.end() );	 // 终结符 0..255
	int								 next_sym = 256;					 // 非终结符从 256 开始
	std::vector<std::pair<int, int>> rules_vec;							 // 按创建顺序保存 RHS

	while ( true )
	{
		auto freq = count_pairs( seq );
		if ( freq.empty() )
			break;

		// 选频次 >=2 的最频繁 pair；频率相同按字典序稳定选
		std::pair<int, int> best_pair { 0, 0 };
		int					best_f = 1;
		bool				has_cand = false;
		for ( const auto& kv : freq )
		{
			const auto& p = kv.first;
			int			f = kv.second;
			if ( f > best_f || ( f == best_f && has_cand && p < best_pair ) )
			{
				best_pair = p;
				best_f = f;
				has_cand = true;
			}
			else if ( !has_cand && f > 1 )
			{
				best_pair = p;
				best_f = f;
				has_cand = true;
			}
		}
		if ( !has_cand )
			break;

		auto [ new_seq, replaced ] = replace_non_overlapping( seq, best_pair, next_sym );
		if ( replaced < 2 )
			break;	// 与你的 Python 保持一致：低于 2 次不引入规则

		rules_vec.emplace_back( best_pair.first, best_pair.second );
		seq = std::move( new_seq );
		++next_sym;
	}

	// ---- 序列化（完全按你的 Python 格式）----
	std::vector<std::uint8_t> out;
	out.push_back( 'R' );
	out.push_back( 'P' );			   // magic
	auto enc = uleb128_encode( 256 );  // terminals
	out.insert( out.end(), enc.begin(), enc.end() );

	std::size_t nrules = rules_vec.size();
	auto		nr_enc = uleb128_encode( nrules );	// nrules
	out.insert( out.end(), nr_enc.begin(), nr_enc.end() );

	// 规则 LHS 隐式：256+i；按创建顺序写 RHS 两个 ULEB128
	for ( std::size_t i = 0; i < nrules; ++i )
	{
		auto a_enc = uleb128_encode( static_cast<std::size_t>( rules_vec[ i ].first ) );
		auto b_enc = uleb128_encode( static_cast<std::size_t>( rules_vec[ i ].second ) );
		out.insert( out.end(), a_enc.begin(), a_enc.end() );
		out.insert( out.end(), b_enc.begin(), b_enc.end() );
	}

	// 最终序列长度 & 符号（全部 ULEB128；符号可以是 >=256 的非终结符）
	auto sl_enc = uleb128_encode( seq.size() );
	out.insert( out.end(), sl_enc.begin(), sl_enc.end() );
	for ( int s : seq )
	{
		auto s_enc = uleb128_encode( static_cast<std::size_t>( s ) );
		out.insert( out.end(), s_enc.begin(), s_enc.end() );
	}

	std::unordered_map<std::string, std::string> meta { { "rules", std::to_string( nrules ) }, { "final_len", std::to_string( seq.size() ) }, { "terminals", "256" }, { "nrules", std::to_string( nrules ) } };
	return { out, meta };
}

// -----------------------------
// Strict Re-Pair decompression (inverse of above)
// -----------------------------
std::vector<std::uint8_t> repair_decompress( const std::vector<std::uint8_t>& data, std::size_t orig_len )
{
	std::size_t i = 0;
	if ( data.size() < 2 || data[ 0 ] != 'R' || data[ 1 ] != 'P' )
	{
		throw std::runtime_error( "RePair: bad magic" );
	}
	i = 2;

	// terminals
	auto [ terminals, i1 ] = uleb128_decode_stream( data, i );
	i = i1;
	if ( terminals != 256 )
	{
		throw std::runtime_error( "RePair: unsupported terminal alphabet" );
	}

	// 规则条数
	auto [ nrules, i2 ] = uleb128_decode_stream( data, i );
	i = i2;

	// 读取规则：隐式 LHS = 256 + ridx；RHS 两个都是 ULEB128（可 >=256）
	std::unordered_map<int, std::pair<int, int>> rules;
	rules.reserve( nrules );
	for ( std::size_t ridx = 0; ridx < nrules; ++ridx )
	{
		auto [ a, i3 ] = uleb128_decode_stream( data, i );
		i = i3;
		auto [ b, i4 ] = uleb128_decode_stream( data, i );
		i = i4;
		rules[ static_cast<int>(256 + ridx) ] = std::pair<int,int>( static_cast<int>(a), static_cast<int>(b) );
	}

	// 读取最终序列（每个符号都是 ULEB128，可为 >=256 的非终结符）
	auto [ seq_len, i5 ] = uleb128_decode_stream( data, i );
	i = i5;
	std::vector<int> seq;
	seq.reserve( seq_len );
	for ( std::size_t k = 0; k < seq_len; ++k )
	{
		auto [ s, inext ] = uleb128_decode_stream( data, i );
		i = inext;
		seq.push_back( static_cast<int>( s ) );
	}

	// 迭代+记忆化展开，避免重复/深递归
	std::unordered_map<int, std::vector<std::uint8_t>> memo;
	std::vector<std::uint8_t>						   out;
	out.reserve( orig_len );
	for ( int s : seq )
	{
		const auto& chunk = expand_symbol( s, rules, memo );
		out.insert( out.end(), chunk.begin(), chunk.end() );
	}
	if ( out.size() != orig_len )
	{
		throw std::runtime_error( "RePair output length mismatch" );
	}
	return out;
}

// LFSR predictor: subtract pseudo‑random state from input bytes and encode the
// deltas as ULEB128.  The metadata map is unused here.
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> encode_lfsr_predict( const std::vector<std::uint8_t>& block )
{
	std::uint8_t			  state = 1;
	std::uint8_t			  taps = 0b10010110;
	std::vector<std::uint8_t> out;
	out.reserve( block.size() );
	for ( auto b : block )
	{
		std::uint8_t			  pred = state;
		std::uint8_t			  delta = static_cast<std::uint8_t>( ( b - pred ) & 0xFFu );
		std::vector<std::uint8_t> enc = uleb128_encode( delta );
		out.insert( out.end(), enc.begin(), enc.end() );
		// Update LFSR state.
		std::uint8_t fb = 0;
		for ( int bit = 0; bit < 8; ++bit )
		{
			if ( ( taps >> bit ) & 1u )
			{
				fb ^= ( state >> bit ) & 1u;
			}
		}
		state = static_cast<std::uint8_t>( ( state << 1 ) | fb );
	}
	return { out, {} };
}

// Decode LFSR predictor encoded stream back into original bytes.  orig_len
// specifies the number of bytes to decode.
std::vector<std::uint8_t> decode_lfsr_predict( const std::vector<std::uint8_t>& data, std::size_t orig_len )
{
	std::uint8_t			  state = 1;
	std::uint8_t			  taps = 0b10010110;
	std::vector<std::uint8_t> out;
	out.reserve( orig_len );
	std::size_t pos = 0;
	for ( std::size_t k = 0; k < orig_len; ++k )
	{
		auto [ delta, ni ] = uleb128_decode_stream( data, pos );
		pos = ni;
		std::uint8_t b = static_cast<std::uint8_t>( ( delta + state ) & 0xFFu );
		out.push_back( b );
		// Update LFSR state.
		std::uint8_t fb = 0;
		for ( int bit = 0; bit < 8; ++bit )
		{
			if ( ( taps >> bit ) & 1u )
			{
				fb ^= ( state >> bit ) & 1u;
			}
		}
		state = static_cast<std::uint8_t>( ( state << 1 ) | fb );
	}
	return out;
}

// Encode a block using BBWT→MTF→Rice with optional bitwise transforms.  The
// flags field encodes which transforms were applied.  Meta stores flags,
// rice parameter k, and original lengths for use by the decoder.
std::pair<std::vector<std::uint8_t>, BBWTMeta> encode_bbwt_mtf_rice( const std::vector<std::uint8_t>& block, bool use_bitplane, bool use_lfsr, bool use_nibble, bool use_bitrev, bool use_gray, std::uint8_t rice_param )
{
	std::vector<std::uint8_t> bbwt = bbwt_forward( block );
	std::vector<std::uint8_t> mtf_list = mtf_encode( bbwt );
	std::vector<std::uint8_t> seq_bytes = mtf_list;
	// Apply optional bitwise transforms in order.
	if ( use_bitplane )
		seq_bytes = bitplane_interleave( seq_bytes );
	if ( use_lfsr )
		seq_bytes = lfsr_whiten( seq_bytes );
	if ( use_nibble )
		seq_bytes = nibble_swap( seq_bytes );
	if ( use_bitrev )
		seq_bytes = bit_reverse( seq_bytes );
	if ( use_gray )
		seq_bytes = gray_encode_bytes( seq_bytes );
	// Rice encode the transformed bytes as a sequence of non‑negative ints.
	std::vector<std::uint64_t> seq64( seq_bytes.begin(), seq_bytes.end() );
	std::vector<std::uint8_t>  payload = rice_encode( seq64, rice_param );
	std::uint8_t			   flags = 0;
	if ( use_bitplane )
		flags |= 1;
	if ( use_lfsr )
		flags |= 2;
	if ( use_nibble )
		flags |= 4;
	if ( use_bitrev )
		flags |= 8;
	if ( use_gray )
		flags |= 16;
	BBWTMeta meta { flags, rice_param, seq_bytes.size(), block.size() };
	return { payload, meta };
}

// Decode BBWT→MTF→Rice payload given meta information.  This function
// reconstructs the original block by inverting the optional transforms in
// reverse order, decoding Rice, reversing bitwise modules, MTF, and BWT.
std::vector<std::uint8_t> decode_bbwt_mtf_rice( const std::vector<std::uint8_t>& payload, const BBWTMeta& meta )
{
	std::vector<std::uint64_t> seq = rice_decode( payload, meta.k, meta.length );
	std::vector<std::uint8_t>  seq_bytes;
	seq_bytes.reserve( seq.size() );
	for ( auto n : seq )
	{
		seq_bytes.push_back( static_cast<std::uint8_t>( n & 0xFFu ) );
	}
	// Apply inverse transforms in reverse order of encoding.
	if ( meta.flags & 16 )
		seq_bytes = gray_decode_bytes( seq_bytes );
	if ( meta.flags & 8 )
		seq_bytes = bit_reverse( seq_bytes );
	if ( meta.flags & 4 )
		seq_bytes = nibble_swap( seq_bytes );
	if ( meta.flags & 2 )
		seq_bytes = lfsr_whiten( seq_bytes );  // whitening is self‑inverse
	if ( meta.flags & 1 )
		seq_bytes = bitplane_deinterleave( seq_bytes, meta.length );
	// MTF decode and BBWT inverse.
	std::vector<std::uint8_t> mtf_list( seq_bytes.begin(), seq_bytes.end() );
	std::vector<std::uint8_t> bbwt = mtf_decode( mtf_list );
	return bbwt_inverse( bbwt );
}

void ef_write_positions( const std::vector<std::uint32_t>& ends, const EFParams& ep, BitWriter& bw )
{
	const std::uint32_t M = ep.M;
	const std::uint32_t L = ep.L;
	const std::uint32_t N = ep.N;
	if ( ends.size() != static_cast<std::size_t>( M ) )
		throw std::runtime_error( "ef_write_positions: size mismatch" );
	const std::uint32_t U = ( L == 0 ) ? N : ( ( N + ( 1u << L ) - 1u ) >> L );	 // ceil(N / 2^L)
	const std::uint32_t B_len = M + U;
	// 1) Low bits first (MSB-first per value)
	if ( L > 0 )
	{
		for ( std::uint32_t x : ends )
		{
			std::uint32_t lo = x & ( ( 1u << L ) - 1u );
			bw.write_kbits( lo, static_cast<int>( L ) );
		}
	}
	// 2) High bitmap next
	std::size_t idx = 0;
	for ( std::size_t i = 0; i < ends.size(); ++i )
	{
		std::uint32_t x = ends[ i ];
		if ( i > 0 && x <= ends[ i - 1 ] )
			throw std::runtime_error( "ef_write_positions: sequence not strictly increasing" );
		if ( x > N )
			throw std::runtime_error( "ef_write_positions: value exceeds N" );
		std::uint32_t hi = ( L == 0 ) ? x : ( x >> L );
		std::uint32_t pos = hi + static_cast<std::uint32_t>( i );
		while ( idx < pos )
		{
			bw.write_bit( 0 );
			++idx;
		}
		bw.write_bit( 1 );
		++idx;
	}
	while ( idx < B_len )
	{
		bw.write_bit( 0 );
		++idx;
	}
}

std::vector<std::uint32_t> ef_read_positions( BitReader& br, const EFParams& ep )
{
	const std::uint32_t		   M = ep.M;
	const std::uint32_t		   L = ep.L;
	const std::uint32_t		   N = ep.N;
	std::vector<std::uint32_t> lows( M, 0 );
	if ( L > 0 )
	{
		for ( std::uint32_t i = 0; i < M; ++i )
		{
			std::uint32_t v = 0;
			for ( std::uint32_t t = 0; t < L; ++t )
			{
				int b = br.read_bit();
				v = ( v << 1 ) | static_cast<std::uint32_t>( b & 1 );
			}
			lows[ i ] = v;
		}
	}
	const std::uint32_t		   U = ( L == 0 ) ? N : ( ( N + ( 1u << L ) - 1u ) >> L );
	const std::uint32_t		   B_len = M + U;
	std::vector<std::uint32_t> ends;
	ends.reserve( M );
	std::uint32_t seen = 0;
	for ( std::uint32_t pos = 0; pos < B_len; ++pos )
	{
		int b = br.read_bit();
		if ( b == 1 )
		{
			std::uint32_t hi = pos - seen;
			std::uint32_t lo = ( L > 0 ) ? lows[ seen ] : 0u;
			std::uint32_t x = ( hi << L ) | lo;
			if ( x > N )
				throw std::runtime_error( "ef_read_positions: value exceeds N" );
			if ( seen > 0 && x <= ends.back() )
				throw std::runtime_error( "ef_read_positions: non-increasing" );
			ends.push_back( x );
			++seen;
		}
	}
	if ( ends.size() != M )
		throw std::runtime_error( "ef_read_positions: bitmap ones != M" );
	return ends;
}

// ------------------------------------------------------------
// Rice encode: seq of non-negative integers with parameter 2^k
//   Each n is coded as q in unary ('1'*q + '0'), then remainder
//   r in k MSB-first bits. For k==0 we DO NOT emit remainder.
// The resulting bitstream is padded with zeros to a byte boundary.
// ------------------------------------------------------------
std::vector<std::uint8_t> rice_encode_bytes( const std::vector<std::uint32_t>& seq, int k )
{
	if ( k < 0 )
		k = 0;
	BitWriter			bw;
	const std::uint32_t M = ( 1u << k );
	for ( std::uint32_t n : seq )
	{
		// n = q*M + r  (0 <= r < M)
		const std::uint32_t q = ( k ? ( n / M ) : n );	// q even for k==0 is fine
		const std::uint32_t r = ( k ? ( n % M ) : 0 );
		bw.write_unary( q );
		if ( k > 0 )
			bw.write_kbits( r, k );	 // remainder only when k>0
	}
	bw.pad_to_byte();  // byte-align
	return std::move( bw.buf );
}

// (Optional) Classic Rice decode when the caller knows 'nvals'.
// Kept here for completeness; not used by the slim pipeline decode.
std::vector<std::uint32_t> rice_decode_nvals( const std::vector<std::uint8_t>& data, int k, std::size_t nvals )
{
	if ( k < 0 )
		k = 0;
	BitReader				   br( data );
	const std::uint32_t		   M = ( 1u << k );
	std::vector<std::uint32_t> out;
	out.reserve( nvals );

	for ( std::size_t t = 0; t < nvals; ++t )
	{
		// unary q: read '1' until '0'
		std::uint32_t q = 0;
		for ( ;; )
		{
			const int bit = br.read_bit();
			if ( bit == 1 )
				++q;
			else
				break;	// saw the '0' terminator
		}
		// remainder r
		std::uint32_t r = 0;
		if ( k > 0 )
		{
			for ( int i = 0; i < k; ++i )
				r = ( r << 1 ) | br.read_bit();
		}
		out.push_back( q * M + r );
	}
	return out;
}

// Decode Rice values UNTIL the sum of runs equals 'target_len'.
// This matches the Python '_rice_decode_until_len' behavior.
// We read from the current bit position of BitReader, and DO NOT
// consume past the last needed bit (caller can align/advance).
std::vector<std::uint32_t> rice_decode_until_len( BitReader& br, int k, std::size_t target_len )
{
	if ( k < 0 )
		k = 0;
	const std::uint32_t M = ( 1u << k );

	std::vector<std::uint32_t> runs;
	runs.reserve( 64 );

	std::size_t total = 0;
	while ( total < target_len )
	{
		// unary q
		std::uint32_t q = 0;
		for ( ;; )
		{
			const int bit = br.read_bit();	// throws on truncation
			if ( bit == 1 )
				++q;
			else
				break;
		}
		// remainder r
		std::uint32_t r = 0;
		if ( k > 0 )
		{
			for ( int i = 0; i < k; ++i )
				r = ( r << 1 ) | br.read_bit();
		}
		const std::uint32_t val = q * M + r;
		if ( val == 0 )
			throw std::runtime_error( "Invalid Rice value (non-positive)" );
		runs.push_back( val );
		total += val;
		if ( total > target_len )
			throw std::runtime_error( "RLE overrun: sum(runs) > target_len" );
	}
	return runs;
}

// Try k in [0..15] and return (best_k, encoded_bytes) minimizing payload size.
// The output of each candidate is already byte-aligned (via BitWriter).
std::pair<int, std::vector<std::uint8_t>> choose_best_rice( const std::vector<std::uint32_t>& runs )
{
	int						  best_k = 0;
	std::vector<std::uint8_t> best;
	bool					  first = true;
	for ( int k = 0; k < 16; ++k )
	{
		auto buf = rice_encode_bytes( runs, k );
		if ( first || buf.size() < best.size() )
		{
			best_k = k;
			best = std::move( buf );
			first = false;
		}
	}
	return { best_k, std::move( best ) };
}

// === Round 6 helpers ===

// =====================================================================
// Helper functions for the new Table‑of‑Contents (TOC) container.
// These functions implement run‑length encoding, canonical Huffman coding,
// Rice coding, ZigZag transforms and Elias–Fano position coding.
// They mirror the Python reference implementation exactly.  See
// kolm_final_researched_v2.py for details.

// Run‑length encode a sequence of method IDs. Returns (symbols, run_lengths).
std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> rle_ids( const std::vector<std::uint8_t>& ids )
{
	if ( ids.empty() )
		return { {}, {} };
	std::vector<std::uint32_t> syms;
	std::vector<std::uint32_t> runs;
	syms.push_back( static_cast<std::uint32_t>( ids[ 0 ] ) );
	runs.push_back( 1u );
	for ( std::size_t i = 1; i < ids.size(); ++i )
	{
		std::uint32_t x = static_cast<std::uint32_t>( ids[ i ] );
		if ( x == syms.back() )
		{
			++runs.back();
		}
		else
		{
			syms.push_back( x );
			runs.push_back( 1u );
		}
	}
	return { syms, runs };
}

// Compute Huffman code lengths from symbol frequencies.
// Returns a mapping: symbol -> length (in bits).  Single‑symbol input
// yields length 1.  Zero frequencies are ignored.
std::unordered_map<std::uint32_t, int> huff_lengths( const std::unordered_map<std::uint32_t, std::uint32_t>& freq )
{
	// Collect nodes with positive frequency.
	struct Node
	{
		std::uint64_t weight;
		std::uint32_t symbol;
		Node*		  left;
		Node*		  right;
	};

	// Min‑heap ordering by weight then by symbol to ensure deterministic tree.
	struct Cmp
	{
		bool operator()( const Node* a, const Node* b ) const noexcept
		{
			if ( a->weight != b->weight )
				return a->weight > b->weight;
			return a->symbol > b->symbol;
		}
	};

	std::priority_queue<Node*, std::vector<Node*>, Cmp> pq;
	std::vector<std::unique_ptr<Node>>					storage;

	for ( const auto& kv : freq )
	{
		std::uint32_t sym = kv.first;
		std::uint32_t w = kv.second;
		if ( w == 0 )
			continue;
		auto n = std::make_unique<Node>();
		n->weight = w;
		n->symbol = sym;
		n->left = nullptr;
		n->right = nullptr;
		pq.push( n.get() );
		storage.push_back( std::move( n ) );
	}

	std::unordered_map<std::uint32_t, int> lengths;

	if ( pq.empty() )
	{
		return lengths;
	}
	if ( pq.size() == 1 )
	{
		// One symbol => assign length 1.
		Node* single = pq.top();
		lengths[ single->symbol ] = 1;
		return lengths;
	}

	// Merge nodes until one tree remains.
	while ( pq.size() > 1 )
	{
		Node* a = pq.top();
		pq.pop();
		Node* b = pq.top();
		pq.pop();
		auto parent = std::make_unique<Node>();
		parent->weight = a->weight + b->weight;
		// Deterministically order children: smallest symbol goes left.
		if ( a->symbol < b->symbol )
		{
			parent->left = a;
			parent->right = b;
		}
		else
		{
			parent->left = b;
			parent->right = a;
		}
		// Use symbol field of parent as min of children for tie‑breaking.
		parent->symbol = std::min( a->symbol, b->symbol );
		Node* p = parent.get();
		pq.push( p );
		storage.push_back( std::move( parent ) );
	}
	Node* root = pq.top();
	// Recursively assign lengths.
	std::function<void( Node*, int )> assign = [ & ]( Node* n, int depth ) {
		if ( !n->left && !n->right )
		{
			lengths[ n->symbol ] = depth;
			return;
		}
		if ( n->left )
			assign( n->left, depth + 1 );
		if ( n->right )
			assign( n->right, depth + 1 );
	};
	assign( root, 0 );
	// Ensure no length zero remains.
	for ( auto& kv : lengths )
	{
		if ( kv.second <= 0 )
			kv.second = 1;
	}
	return lengths;
}

// Build canonical Huffman codes from a length table.  Returns
// (enc_tbl, dec_tbl, maxlen) where:
//   enc_tbl[sym] = (code, length)
//   dec_tbl[L][code] = sym
//   maxlen = maximum code length.
std::tuple<std::unordered_map<std::uint32_t, std::pair<std::uint32_t, int>>, std::unordered_map<int, std::unordered_map<std::uint32_t, std::uint32_t>>, int> huff_canonical( const std::unordered_map<std::uint32_t, int>& lengths )
{
	std::unordered_map<std::uint32_t, std::pair<std::uint32_t, int>>		  enc_tbl;
	std::unordered_map<int, std::unordered_map<std::uint32_t, std::uint32_t>> dec_tbl;
	int																		  maxlen = 0;
	for ( const auto& kv : lengths )
	{
		if ( kv.second > maxlen )
			maxlen = kv.second;
	}
	// Sort by (length, symbol).
	std::vector<std::pair<int, std::uint32_t>> pairs;
	pairs.reserve( lengths.size() );
	for ( const auto& kv : lengths )
		pairs.emplace_back( kv.second, kv.first );
	std::sort( pairs.begin(), pairs.end(), []( const auto& a, const auto& b ) {
		if ( a.first != b.first )
			return a.first < b.first;
		return a.second < b.second;
	} );
	std::uint32_t code = 0;
	int			  prev_len = 0;
	for ( const auto& kv : pairs )
	{
		int			  L = kv.first;
		std::uint32_t sym = kv.second;
		if ( L > prev_len )
		{
			code <<= ( L - prev_len );
			prev_len = L;
		}
		enc_tbl[ sym ] = { code, L };
		dec_tbl[ L ][ code ] = sym;
		code += 1u;
	}
	return { enc_tbl, dec_tbl, maxlen };
}

// Encode Huffman‑coded symbols to a BitWriter.
void huff_encode_symbols( BitWriter& bw, const std::unordered_map<std::uint32_t, std::pair<std::uint32_t, int>>& enc_tbl, const std::vector<std::uint32_t>& syms )
{
	for ( std::uint32_t s : syms )
	{
		auto it = enc_tbl.find( s );
		if ( it == enc_tbl.end() )
			throw std::runtime_error( "huff_encode_symbols: symbol not in table" );
		const auto&	  code_len = it->second;
		std::uint32_t code = code_len.first;
		int			  L = code_len.second;
		bw.write_kbits( code, L );
	}
}

// Decode Huffman‑coded symbols from a BitReader.  nvals specifies how many symbols.
std::vector<std::uint32_t> huff_decode_symbols( BitReader& br, const std::unordered_map<int, std::unordered_map<std::uint32_t, std::uint32_t>>& dec_tbl, int maxlen, std::size_t nvals )
{
	std::vector<std::uint32_t> out;
	out.reserve( nvals );
	for ( std::size_t i = 0; i < nvals; ++i )
	{
		std::uint32_t c = 0;
		for ( int L = 1; L <= maxlen; ++L )
		{
			c = ( c << 1 ) | static_cast<std::uint32_t>( br.read_bit() );
			auto it1 = dec_tbl.find( L );
			if ( it1 != dec_tbl.end() )
			{
				auto it2 = it1->second.find( c );
				if ( it2 != it1->second.end() )
				{
					out.push_back( it2->second );
					break;
				}
			}
			if ( L == maxlen )
				throw std::runtime_error( "Huffman decode failed" );
		}
	}
	return out;
}

// Write a sequence of non‑negative integers using Rice coding (bit‑precise).
// Parameter k encodes divisor M=2^k; for k==0 no remainder bits are emitted.
void rice_write_values( BitWriter& bw, const std::vector<std::uint32_t>& seq, int k )
{
	if ( k < 0 )
		k = 0;
	for ( std::uint32_t n : seq )
	{
		std::uint32_t q = ( k > 0 ) ? ( n >> k ) : n;  // q = n // 2^k
		for ( std::uint32_t i = 0; i < q; ++i )
			bw.write_bit( 1 );
		bw.write_bit( 0 );
		if ( k > 0 )
		{
			std::uint32_t r = n & ( ( 1u << k ) - 1u );
			bw.write_kbits( r, k );
		}
	}
}

// Read a fixed count of Rice‑coded integers from a BitReader.
std::vector<std::uint32_t> rice_read_n( BitReader& br, int k, std::size_t nvals )
{
	if ( k < 0 )
		k = 0;
	std::vector<std::uint32_t> out;
	out.reserve( nvals );
	for ( std::size_t t = 0; t < nvals; ++t )
	{
		std::uint32_t q = 0;
		// unary: count 1s until 0
		while ( true )
		{
			int bit = br.read_bit();
			if ( bit == 1 )
				++q;
			else
				break;
		}
		std::uint32_t r = 0;
		if ( k > 0 )
		{
			for ( int i = 0; i < k; ++i )
			{
				r = ( r << 1 ) | static_cast<std::uint32_t>( br.read_bit() );
			}
		}
		out.push_back( ( q << k ) | r );
	}
	return out;
}

// ZigZag encode/decode 32‑bit signed integers.
std::uint32_t zigzag_encode_32( std::int32_t x )
{
	// shift left by 1 and invert sign bit
	return ( static_cast<std::uint32_t>( x ) << 1 ) ^ static_cast<std::uint32_t>( x >> 31 );
}
std::int32_t zigzag_decode_32( std::uint32_t n )
{
	// recover sign in low bit
	return static_cast<std::int32_t>( ( n >> 1 ) ^ ( static_cast<std::int32_t>( -( n & 1u ) ) ) );
}

// Write positions P (strictly increasing) using Elias–Fano coding.
// Read positions encoded via Elias–Fano.
// Encode V2 using the slim header (no flag/L/run_count/paylen).
// Layout:
//   header:
//	 hdr0	: 1B, top 3 bits store 'mode' => hdr0 = (mode & 7) << 5
//	 raw_mask: 1B, bit j==1 means plane j is RAW (packed bits)
//	 b1_mask : 1B, bit j stores the first run bit (b1) for ENCODED planes
//	 k_list  : 1B per ENCODED plane (in j order), Rice parameter k
//   payload (byte-aligned per plane, no cross-plane bit sharing):
//	 for j = 0..7:
//	   if RAW	=> ceil(L/8) bytes of packed bits (MSB-first)
//	   if ENCODED=> Rice bitstream for runs (padded to next byte)
//
// Notes:
// - L (plane length in symbols) is exactly the block's byte length, and the
//   decoder receives it from the container (orig_len). We do not store L here.
// - Each plane chooses RAW vs (BBWT -> RLE -> Rice(best k)) by comparing sizes.
//   If ENCODED is chosen, we store 1B k and the b1 bit in b1_mask.
std::pair<std::vector<std::uint8_t>, std::unordered_map<std::string, std::string>> encode_new_pipeline( const std::vector<std::uint8_t>& block )
{
	std::unordered_map<std::string, std::string> meta;	// unused, kept for symmetry
	if ( block.empty() )
		return { std::vector<std::uint8_t> {}, meta };

	// 1) reversible automaton on byte stream
	auto							 ar = circuit_map_automaton_forward( block );  // {data, mode}
	const std::vector<std::uint8_t>& mapped = ar.data;
	const std::uint8_t				 mode = ar.mode & 0x07;

	// 2) split into 8 MSB-first bit-planes
	auto [ planes, L ] = bytes_to_bitplanes( mapped );	// planes[j] is vector<int> of {0,1}
	const std::size_t nbytes_plane = ( L + 7u ) / 8u;

	// 3) decide RAW vs ENCODED per plane, collect masks/k/payload chunks
	std::uint8_t			  raw_mask = 0;
	std::uint8_t			  b1_mask = 0;
	std::vector<std::uint8_t> k_list;
	k_list.reserve( 8 );
	std::vector<std::vector<std::uint8_t>> chunks;
	chunks.reserve( 8 );

	for ( int j = 0; j < 8; ++j )
	{
		const auto& Uj = planes[ j ];

		// RAW candidate: pack bits MSB-first into bytes
		const auto raw_bytes = pack_bits_to_bytes( Uj );

		// ENCODED candidate: BBWT -> RLE -> Rice (choose best k)
		// Convert 0/1 ints to bytes for BBWT
		std::vector<std::uint8_t> Uj8;
		Uj8.reserve( Uj.size() );
		for ( int b : Uj )
			Uj8.push_back( static_cast<std::uint8_t>( b & 1 ) );
		const auto Lj_bytes = bbwt_forward( Uj8 );

		std::vector<int> Lj_bits;
		Lj_bits.reserve( Lj_bytes.size() );
		for ( auto v : Lj_bytes )
			Lj_bits.push_back( static_cast<int>( v & 1 ) );

		auto		rle = rle_binary( Lj_bits );  // pair<int first_bit, vector<int> runs>
		const int	b1 = rle.first & 1;
		const auto& runs_i = rle.second;

		if ( runs_i.empty() )
		{
			// Degenerate: encode as RAW
			raw_mask |= ( 1u << j );
			chunks.push_back( raw_bytes );
			continue;
		}

		// Cast runs to 32-bit and choose best Rice k in [0..15].
		std::vector<std::uint32_t> runs;
		runs.reserve( runs_i.size() );
		for ( int v : runs_i )
			runs.push_back( static_cast<std::uint32_t>( v ) );
		auto [ k_opt, rice_bytes ] = choose_best_rice( runs );

		// Compare sizes: ENCODED pays +1B for 'k'
		if ( raw_bytes.size() <= rice_bytes.size() + 1u )
		{
			raw_mask |= ( 1u << j );
			chunks.push_back( raw_bytes );
		}
		else
		{
			if ( b1 )
				b1_mask |= ( 1u << j );
			k_list.push_back( static_cast<std::uint8_t>( k_opt & 0xFF ) );
			chunks.push_back( std::move( rice_bytes ) );
		}
	}

	// 4) assemble header
	std::vector<std::uint8_t> header;
	header.reserve( 3 + k_list.size() );
	const std::uint8_t hdr0 = ( mode & 0x07 ) << 5;
	header.push_back( hdr0 );
	header.push_back( raw_mask );
	header.push_back( b1_mask );
	// append k_list in plane order (only for encoded planes)
	{
		std::size_t idx = 0;
		for ( int j = 0; j < 8; ++j )
		{
			if ( ( ( raw_mask >> j ) & 1 ) == 0 )
			{
				header.push_back( k_list[ idx++ ] );
			}
		}
	}

	// 5) payload = concat of per-plane bytes (each already byte-aligned)
	std::vector<std::uint8_t> payload;
	std::size_t				  payload_sz = 0;
	for ( auto& ch : chunks )
		payload_sz += ch.size();
	payload.reserve( payload_sz );
	for ( auto& ch : chunks )
		payload.insert( payload.end(), ch.begin(), ch.end() );

	// return header || payload
	std::vector<std::uint8_t> out;
	out.reserve( header.size() + payload.size() );
	out.insert( out.end(), header.begin(), header.end() );
	out.insert( out.end(), payload.begin(), payload.end() );
	return { std::move( out ), std::move( meta ) };
}

// Decode V2 slim header. The container tells us 'orig_len' (L).
// We parse hdr0/raw_mask/b1_mask/k_list, then for each plane:
//  - RAW	 : read ceil(L/8) bytes and unpack bits
//  - ENCODED : read a Rice bitstream until sum(runs)==L, then byte-align
// After reconstructing 8 planes, we pack them back to bytes and invert
// the automaton selected by 'mode'.
std::vector<std::uint8_t> decode_new_pipeline( const std::vector<std::uint8_t>& payload, std::size_t orig_len, const std::unordered_map<std::string, std::string>& /*meta*/ )
{
	const std::size_t L = orig_len;
	if ( L == 0 )
		return {};
	if ( payload.size() < 3 )
		throw std::runtime_error( "V2 slim header truncated" );

	std::size_t		   pos = 0;
	const std::uint8_t hdr0 = payload[ pos++ ];
	const std::uint8_t raw_mask = payload[ pos++ ];
	const std::uint8_t b1_mask = payload[ pos++ ];
	const std::uint8_t mode = ( hdr0 >> 5 ) & 0x07;

	// read k_list for encoded planes, in j order
	const int enc_count = 8 - std::popcount( raw_mask );
	if ( pos + static_cast<std::size_t>( enc_count ) > payload.size() )
		throw std::runtime_error( "V2 slim header k_list truncated" );
	std::vector<std::uint8_t> k_list;
	k_list.reserve( enc_count );
	for ( int i = 0; i < enc_count; ++i )
		k_list.push_back( payload[ pos++ ] );

	// payload bytes start here
	const std::vector<std::uint8_t> data( payload.begin() + pos, payload.end() );
	std::size_t						data_pos = 0;  // byte index inside 'data'

	std::vector<std::vector<int>> planes;
	planes.reserve( 8 );
	auto k_it = k_list.begin();

	for ( int j = 0; j < 8; ++j )
	{
		if ( ( ( raw_mask >> j ) & 1 ) == 1 )
		{
			// RAW plane
			const std::size_t need = ( L + 7u ) / 8u;
			if ( data_pos + need > data.size() )
				throw std::runtime_error( "V2 payload truncated in RAW plane" );
			std::vector<std::uint8_t> buf( data.begin() + data_pos, data.begin() + data_pos + need );
			data_pos += need;
			planes.push_back( unpack_bits_from_bytes( buf, L ) );
		}
		else
		{
			// ENCODED plane: Rice until sum==L, then align to next byte
			if ( k_it == k_list.end() )
				throw std::runtime_error( "k_list exhausted" );
			const int k = static_cast<int>( *k_it++ & 0xFF );
			const int b1 = ( b1_mask >> j ) & 1;

			BitReader  br( data, data_pos, 0 );
			const auto runs32 = rice_decode_until_len( br, k, L );
			br.align_next_byte();
			auto [ next_byte, _bit ] = br.tell();
			data_pos = next_byte;

			// RLE inverse -> BBWT inverse -> plane bits
			std::vector<int> runs_i;
			runs_i.reserve( runs32.size() );
			for ( auto v : runs32 )
				runs_i.push_back( static_cast<int>( v ) );
			const auto Lj_bits = unrle_binary( b1, runs_i );

			std::vector<std::uint8_t> Lj_bytes;
			Lj_bytes.reserve( Lj_bits.size() );
			for ( auto b : Lj_bits )
				Lj_bytes.push_back( static_cast<std::uint8_t>( b & 1 ) );

			auto Uj_bytes = bbwt_inverse( Lj_bytes );  // length should be L
			if ( Uj_bytes.size() != L )
			{
				if ( Uj_bytes.size() > L )
					Uj_bytes.resize( L );
				else
					Uj_bytes.insert( Uj_bytes.end(), L - Uj_bytes.size(), 0 );
			}
			std::vector<int> Uj;
			Uj.reserve( Uj_bytes.size() );
			for ( auto b : Uj_bytes )
				Uj.push_back( static_cast<int>( b & 1 ) );
			planes.push_back( std::move( Uj ) );
		}
	}

	// merge planes -> bytes, invert automaton
	const auto mapped = bitplanes_to_bytes( planes );
	return circuit_map_automaton_inverse( mapped, mode );
}

// Top‑level compression: break input data into content‑defined blocks,
// evaluate candidate encoders on each block, and choose the smallest
// representation.  A simple container format is produced beginning with
// magic 'KOLR', followed by block size, total length, number of blocks,
// then for each block: method id, original length, payload length and the
// payload itself.  See Python for candidate definitions and ordering.
// =========================================
// Header field: mode pack/unpack (compatibility bit)
// =========================================
std::uint32_t pack_mode_and_size( std::uint32_t mode, std::uint32_t size31 )
{
	if ( mode != MODE_FIXED && mode != MODE_CDC )
	{
		throw std::invalid_argument( "invalid mode" );
	}
	if ( size31 > 0x7FFFFFFFu )
	{
		throw std::invalid_argument( "size out of range (must fit in 31 bits)" );
	}
	return ( ( mode & 1u ) << 31 ) | ( size31 & 0x7FFFFFFFu );
}

void unpack_mode_and_size( std::uint32_t word, std::uint32_t& mode, std::uint32_t& size31 )
{
	mode = ( word >> 31 ) & 1u;
	size31 = word & 0x7FFFFFFFu;
}

// -----------------------------------------
// Small LE32 helpers (Python struct.pack/unpack '<I')
// -----------------------------------------
void write_le32( ByteVector& out, std::uint32_t v )
{
	out.push_back( static_cast<std::uint8_t>( v & 0xFFu ) );
	out.push_back( static_cast<std::uint8_t>( ( v >> 8 ) & 0xFFu ) );
	out.push_back( static_cast<std::uint8_t>( ( v >> 16 ) & 0xFFu ) );
	out.push_back( static_cast<std::uint8_t>( ( v >> 24 ) & 0xFFu ) );
}

std::uint32_t read_le32( const ByteVector& in, std::size_t off )
{
	if ( off + 4 > in.size() )
		throw std::runtime_error( "Truncated header" );
	std::uint32_t v = 0;
	v |= static_cast<std::uint32_t>( in[ off + 0 ] ) << 0;
	v |= static_cast<std::uint32_t>( in[ off + 1 ] ) << 8;
	v |= static_cast<std::uint32_t>( in[ off + 2 ] ) << 16;
	v |= static_cast<std::uint32_t>( in[ off + 3 ] ) << 24;
	return v;
}

// Encoder/Decoder type aliases
using Encoder = std::function<std::pair<ByteVector, MetaMap>( ByteSpan )>;
using Decoder = std::function<ByteVector( const ByteVector&, std::size_t, const MetaMap& )>;

// Return the fixed, ordered list of encoder candidates.
// NOTE: Order MUST match the decoder table below (0..10).
static const std::vector<Encoder>& _encoder_candidates()
{
	static const std::vector<Encoder> v = {
		// 0 raw
		[]( ByteSpan s ) { return std::make_pair( ByteVector( s.begin(), s.end() ), MetaMap {} ); },
		// 1 xor
		[]( ByteSpan s ) {
			ByteVector				  out;
			out.reserve( s.size() );
			std::uint8_t			  prev = 0;
			for ( std::uint8_t x : s )
			{
				auto enc = uleb128_encode( static_cast<std::uint32_t>( ( x - prev ) & 0xFFu ) );
				out.insert( out.end(), enc.begin(), enc.end() );
				prev = x;
			}
			return std::make_pair( std::move( out ), MetaMap {} );
		},
		// 2..6 bbwt family
		// bbwt, bbwt+bitplane, bbwt+nibble, bbwt+bitrev, bbwt+gray
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			auto [ p, m ] = encode_bbwt_mtf_rice( v, false, false, false, false, false, 2 );
			( void )m;
			return std::make_pair( std::move( p ), MetaMap {} );
		},
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			auto [ p, m ] = encode_bbwt_mtf_rice( v, true, false, false, false, false, 2 );
			( void )m;
			return std::make_pair( std::move( p ), MetaMap {} );
		},
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			auto [ p, m ] = encode_bbwt_mtf_rice( v, false, false, true, false, false, 2 );
			( void )m;
			return std::make_pair( std::move( p ), MetaMap {} );
		},
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			auto [ p, m ] = encode_bbwt_mtf_rice( v, false, false, false, true, false, 2 );
			( void )m;
			return std::make_pair( std::move( p ), MetaMap {} );
		},
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			auto [ p, m ] = encode_bbwt_mtf_rice( v, false, false, false, false, true, 2 );
			( void )m;
			return std::make_pair( std::move( p ), MetaMap {} );
		},
		// 7 lz77
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			return encode_lz77( v );
		},
		// 8 lfsr predictor
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			return encode_lfsr_predict( v );
		},
		// 9 Re-Pair
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			return repair_compress( v );
		},
		// 10 V2 new pipeline
		[]( ByteSpan s ) {
			ByteVector				  v( s.begin(), s.end() );
			return encode_new_pipeline( v );
		},
	};
	return v;
}

// Decoder table aligned with the encoder list above (same indices).
static const std::vector<Decoder>& _decoder_table()
{
	static const std::vector<Decoder> v = {
		// 0 raw
		[]( const ByteVector& p, std::size_t length, const MetaMap& ) {
			if ( p.size() != length )
				throw std::runtime_error( "raw decoder length mismatch" );
			return p;
		},
		// 1 xor
		[]( const ByteVector& p, std::size_t length, const MetaMap& ) {
			ByteVector out;
			out.reserve( length );
			std::size_t	 ip = 0;
			std::uint8_t prev = 0;
			for ( std::size_t i = 0; i < length; ++i )
			{
				auto [ delta, ni ] = uleb128_decode_stream( p, ip );
				ip = ni;
				std::uint8_t b = static_cast<std::uint8_t>( ( prev + static_cast<std::uint8_t>( delta ) ) & 0xFFu );
				out.push_back( b );
				prev = b;
			}
			return out;
		},
		// 2..6 bbwt family
		// bbwt, bbwt+bitplane, bbwt+nibble, bbwt+bitrev, bbwt+gray
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) {
			BBWTMeta m { 0u, 2u, L, L };
			return decode_bbwt_mtf_rice( p, m );
		},
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) {
			BBWTMeta m { 1u, 2u, L, L };
			return decode_bbwt_mtf_rice( p, m );
		},
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) {
			BBWTMeta m { 4u, 2u, L, L };
			return decode_bbwt_mtf_rice( p, m );
		},
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) {
			BBWTMeta m { 8u, 2u, L, L };
			return decode_bbwt_mtf_rice( p, m );
		},
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) {
			BBWTMeta m { 16u, 2u, L, L };
			return decode_bbwt_mtf_rice( p, m );
		},
		// 7 lz77
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) { return decode_lz77( p, L ); },
		// 8 lfsr predictor
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) { return decode_lfsr_predict( p, L ); },
		// 9 Re-Pair
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) { return repair_decompress( p, L ); },
		// 10 V2 new pipeline
		[]( const ByteVector& p, std::size_t L, const MetaMap& ) { return decode_new_pipeline( p, L, MetaMap {} ); },
	};
	return v;
}

// Tiny POD for a selection result (method id + payload).
struct _Best
{
	std::uint8_t method_id;
	ByteVector	 payload;
};

std::vector<size_t> _active_methods( size_t total_methods )
{
	std::vector<std::size_t> ids( total_methods );
	std::iota( ids.begin(), ids.end(), 0 );

	// 1) --only 优先：只保留一个（非法则兜底 raw=0）
	if ( G_ONLY_ENABLED )
	{
		if ( G_ONLY_METHOD < total_methods )
			return { G_ONLY_METHOD };
		return { 0 };
	}

	// 2) 否则再应用 --no-lz77 过滤
	std::vector<std::size_t> out;
	out.reserve( ids.size() );
	for ( auto id : ids )
	{
		if ( G_NO_LZ77 && id == 7 )
			continue;
		out.push_back( id );
	}
	if ( out.empty() )
		out.push_back( 0 );
	return out;
}

// Choose the smallest encoding for one block (MDL selection).
_Best _select_best( ByteSpan block )
{
	const auto& candidates = _encoder_candidates();
	std::size_t best_cost = std::numeric_limits<std::size_t>::max();
	std::size_t best_id = 0;
	ByteVector	best_payload;

	for ( std::size_t mid : _active_methods( candidates.size() ) )
	{
		try
		{
			auto [ payload, meta ] = candidates[ mid ]( block ); ( void )meta;
			// Per-block container layout (after global header "KOLR" + mode/size + total_len + nblocks):
			//   [ 1 byte ] method_id
			//   [ 4 byte ] orig_len   (little-endian u32)
			//   [ 4 byte ] payload_len(little-endian u32)
			//   [ N byte ] payload bytes
			// -> Fixed per-block overhead = 1 + 4 + 4 = 9 bytes (excluding payload).
			const std::size_t cost = payload.size() + 9;
			if ( cost < best_cost || ( cost == best_cost && mid == 0 ) )
			{
				best_cost = cost;
				best_id = mid;
				best_payload = std::move( payload );
			}
		}
		catch ( ... )
		{
			// Ignore failing candidates.
		}
	}
	return _Best { static_cast<std::uint8_t>( best_id ), std::move( best_payload ) };
}

// Decode one block by method id (bounds-checked dispatch).
ByteVector _decode_by_id( std::uint8_t method_id, const ByteVector& payload, std::size_t orig_len )
{
	const auto& dec = _decoder_table();
	if ( method_id >= dec.size() )
		throw std::runtime_error( "decompress: unknown method id" );
	return dec[ method_id ]( payload, orig_len, MetaMap {} );
}

// =========================================
//
// === Round 5: write_toc() — build TOC header + three bitstreams and return bytes/bitlen =========
/**
 * @brief Build TOC header (bytes) and TOC bitstreams (bit-packed) for the given block sequences.
 * Matches Python V2 layout exactly:
 *   toc_header (ULEB128):
 *	 - n_runs
 *	 - K = size of codebook (appeared symbols)
 *	 - (sym, length) pairs sorted by (length, sym)
 *	 - k_method  (Rice k for run_len-1)
 *	 - if MODE_FIXED: last_orig_len
 *	   else (MODE_CDC): k_len (Rice k for ZigZag(orig_len - size_field))
 *   toc_bits (bit order MSB-first):
 *	 - Huffman-encoded run symbols (Stream A part 1)
 *	 - Rice-encoded run lengths (Stream A part 2)
 *	 - [MODE_CDC only] Rice-encoded orig_len ZigZag deltas (Stream B)
 *	 - Elias–Fano of payload cumulative ends (Stream C)
 *
 * @param mode		MODE_FIXED or MODE_CDC
 * @param size_field  size field from global header (base_block_size for fixed, avg_size for CDC)
 * @param method_ids  per-block method id sequence
 * @param orig_lens   per-block original lengths
 * @param payload_lens per-block payload lengths
 * @param toc_header (out) header bytes
 * @param toc_bits   (out) concatenated bitstreams, byte-aligned
 * @param toc_bitlen (out) number of valid bits in toc_bits
 * @param total_payload (out) sum(payload_lens)
 */
void write_toc( std::uint32_t mode, std::uint32_t size_field, const std::vector<std::uint8_t>& method_ids, const std::vector<std::uint32_t>& orig_lens, const std::vector<std::uint32_t>& payload_lens, ByteVector& toc_header, ByteVector& toc_bits, std::uint32_t& toc_bitlen, std::uint32_t& total_payload )
{
	// total payload
	total_payload = 0u;
	for ( auto l : payload_lens )
		total_payload = static_cast<std::uint32_t>( total_payload + l );

	// RLE IDs
	auto [ run_syms, run_lens ] = rle_ids( method_ids );
	const std::size_t n_runs = run_syms.size();

	// Huffman code lengths & canonical codes
	std::unordered_map<std::uint32_t, std::uint32_t> freq;
	freq.reserve( n_runs * 2 + 1 );
	for ( std::uint32_t s : run_syms )
		++freq[ s ];
	auto lengths = huff_lengths( freq );
	auto [ enc_tbl, dec_tbl, maxlen ] = huff_canonical( lengths );

	// Choose Rice k for run lengths (0..7)
	int			best_k = 0;
	std::size_t best_bits = std::numeric_limits<std::size_t>::max();
	for ( int k = 0; k < 8; ++k )
	{
		std::size_t bits = 0;
		for ( std::uint32_t r : run_lens )
		{
			std::uint32_t q = ( k > 0 ) ? ( r >> k ) : r;
			bits += static_cast<std::size_t>( q + 1 + k );
		}
		if ( bits < best_bits )
		{
			best_bits = bits;
			best_k = k;
		}
	}

	// CDC deltas (ZigZag) + choose k2 if needed
	int						   best_k2 = 0;
	std::vector<std::uint32_t> deltas;
	if ( mode == MODE_CDC )
	{
		deltas.reserve( orig_lens.size() );
		for ( std::uint32_t ol : orig_lens )
		{
			std::int32_t diff = static_cast<std::int32_t>( ol ) - static_cast<std::int32_t>( size_field );
			deltas.push_back( zigzag_encode_32( diff ) );
		}
		std::size_t best_bits2 = std::numeric_limits<std::size_t>::max();
		for ( int k = 0; k < 8; ++k )
		{
			std::size_t bits = 0;
			for ( std::uint32_t d : deltas )
			{
				std::uint32_t q = ( k > 0 ) ? ( d >> k ) : d;
				bits += static_cast<std::size_t>( q + 1 + k );
			}
			if ( bits < best_bits2 )
			{
				best_bits2 = bits;
				best_k2 = k;
			}
		}
	}

	// ---- Build toc_header (ULEB128) ----
	toc_header.clear();
	// n_runs
	{
		auto v = uleb128_encode( static_cast<std::uint64_t>( n_runs ) );
		toc_header.insert( toc_header.end(), v.begin(), v.end() );
	}
	// K and (sym,L) pairs in canonical order (by (length, sym))
	{
		// Build vector of pairs for sorting
		std::vector<std::pair<int, std::uint32_t>> pairs;
		pairs.reserve( lengths.size() );
		for ( const auto& kv : lengths )
			pairs.emplace_back( kv.second, kv.first );
		std::sort( pairs.begin(), pairs.end(), []( const auto& a, const auto& b ) {
			if ( a.first != b.first )
				return a.first < b.first;
			return a.second < b.second;
		} );
		auto K = static_cast<std::uint64_t>( pairs.size() );
		auto vK = uleb128_encode( K );
		toc_header.insert( toc_header.end(), vK.begin(), vK.end() );
		for ( const auto& pr : pairs )
		{
			auto vs = uleb128_encode( static_cast<std::uint64_t>( pr.second ) );
			toc_header.insert( toc_header.end(), vs.begin(), vs.end() );
			auto vL = uleb128_encode( static_cast<std::uint64_t>( pr.first ) );
			toc_header.insert( toc_header.end(), vL.begin(), vL.end() );
		}
	}
	// k_method for run lengths
	{
		auto vk = uleb128_encode( static_cast<std::uint64_t>( best_k ) );
		toc_header.insert( toc_header.end(), vk.begin(), vk.end() );
	}
	if ( mode == MODE_FIXED )
	{
		// store only last block orig len
		std::uint32_t last_len = orig_lens.empty() ? 0u : orig_lens.back();
		auto		  v = uleb128_encode( static_cast<std::uint64_t>( last_len ) );
		toc_header.insert( toc_header.end(), v.begin(), v.end() );
	}
	else
	{
		// store k_len for CDC deltas
		auto v = uleb128_encode( static_cast<std::uint64_t>( best_k2 ) );
		toc_header.insert( toc_header.end(), v.begin(), v.end() );
	}


	// ---- Build toc_bits (bitstreams) ----
	BitWriter bw;
	// Stream A.1: Huff(method run symbols)
	huff_encode_symbols( bw, enc_tbl, run_syms );
	// Stream A.2: Rice(run_len)
	rice_write_values( bw, run_lens, best_k );
	// Stream B (CDC only): Rice(deltas)
	if ( mode == MODE_CDC )
	{
		rice_write_values( bw, deltas, best_k2 );
	}
	// Stream C: EF payload ends
	std::vector<std::uint32_t> ends;
	ends.reserve( payload_lens.size() );
	{
		std::uint32_t s = 0u;
		for ( std::uint32_t l : payload_lens )
		{
			s = static_cast<std::uint32_t>( s + l );
			ends.push_back( s );
		}
	}
	EFParams efp = ef_choose_params( total_payload, static_cast<std::uint32_t>( payload_lens.size() ) );
	ef_write_positions( ends, efp, bw );
	// Export bits
	toc_bitlen = static_cast<std::uint32_t>( bw.bit_length() );
	bw.pad_to_byte();
	toc_bits = std::move( bw.buf );
}

// === TocDecoded: output bundle for read_toc ==================================================
struct TocDecoded
{
	std::vector<std::uint8_t>			   method_ids;	  // per-block method id
	std::vector<std::uint32_t>			   orig_lens;	  // per-block original length
	std::vector<std::uint32_t>			   payload_ends;  // cumulative payload end positions
	int									   k_method = 0;
	int									   k_len = 0;			 // only for CDC
	std::unordered_map<std::uint32_t, int> method_code_lengths;	 // symbol->length
};


// === Round 6: read_toc — parse header, decode three streams, reconstruct per-block metadata ======
/**
 * @param mode		MODE_FIXED or MODE_CDC
 * @param size_field  base_block_size (fixed) or avg_size (CDC)
 * @param nblocks	 number of blocks (u16 promoted)
 * @param toc_header  header bytes (as written by write_toc)
 * @param toc_bits	concatenated bitstreams (bit-packed)
 * @param toc_bitlen  valid number of bits in toc_bits
 * @param out		 filled with method_ids, orig_lens, payload_ends
 */
void read_toc( std::uint32_t mode, std::uint32_t size_field, std::uint32_t nblocks, std::uint32_t total_payload, const ByteVector& toc_header, const ByteVector& toc_bits, std::uint32_t toc_bitlen, TocDecoded& out )
{
	// ---- Parse header ----
	std::size_t off = 0;
	const auto	read_u = [ & ]() -> std::uint64_t {
		 auto pr = uleb128_decode_stream( toc_header, off );
		 off = pr.second;
		 return pr.first;
	};

	// n_runs
	std::uint64_t n_runs = read_u();
	// K & (sym,len)
	std::uint64_t						   K = read_u();
	std::unordered_map<std::uint32_t, int> lengths;
	lengths.reserve( static_cast<std::size_t>( K * 2 + 4 ) );
	for ( std::size_t i = 0; i < K; ++i )
	{
		std::uint32_t sym = static_cast<std::uint32_t>( read_u() );
		int			  len = static_cast<int>( read_u() );
		lengths[ sym ] = len;
	}
	// k_method
	out.k_method = static_cast<int>( read_u() );
	// mode-specific
	if ( mode == MODE_FIXED )
	{
		// last_orig_len (store for later)
		std::uint32_t last_orig_len = static_cast<std::uint32_t>( read_u() );
		out.k_len = -1;	 // unused
		// store last len temporarily in k_len (negative sentinel not used elsewhere)
		// we'll reconstruct orig_lens later using size_field and last_orig_len.
		// stash into method_code_lengths after lengths parsed
		( void )last_orig_len;
		// read EF params (N,M,L) as LE32
	}
	else
	{
		// k_len
		out.k_len = static_cast<int>( read_u() );
		// read EF params next
	}
	out.method_code_lengths = lengths;

	// ---- Decode streams ----
	BitReader br( toc_bits );
	// Stream A.1: Huff(run_syms)
	auto [ enc_tbl, dec_tbl, maxlen ] = huff_canonical( lengths );
	std::vector<std::uint32_t> run_syms;
	run_syms.reserve( static_cast<std::size_t>( n_runs ) );
	run_syms = huff_decode_symbols( br, dec_tbl, maxlen, static_cast<std::size_t>( n_runs ) );

	// Stream A.2: Rice(run_lens)
	std::vector<std::uint32_t> run_lens = rice_read_n( br, out.k_method, static_cast<std::size_t>( n_runs ) );

	// Expand to per-block method_ids
	out.method_ids.clear();
	for ( std::size_t i = 0; i < run_syms.size(); ++i )
	{
		std::uint8_t  sym = static_cast<std::uint8_t>( run_syms[ i ] );
		std::uint32_t cnt = run_lens[ i ];
		for ( std::uint32_t t = 0; t < cnt; ++t )
			out.method_ids.push_back( sym );
	}
	if ( out.method_ids.size() != nblocks )
		throw std::runtime_error( "read_toc: expanded method_ids size mismatch" );

	// Stream B (CDC only): Rice(deltas) then ZigZag^-1 to orig_lens
	out.orig_lens.clear();
	out.orig_lens.reserve( nblocks );
	if ( mode == MODE_CDC )
	{
		std::vector<std::uint32_t> deltas = rice_read_n( br, out.k_len, static_cast<std::size_t>( nblocks ) );
		for ( std::uint32_t d : deltas )
		{
			std::int32_t diff = zigzag_decode_32( d );
			std::int32_t v = static_cast<std::int32_t>( size_field ) + diff;
			out.orig_lens.push_back( static_cast<std::uint32_t>( v ) );
		}
	}
	else
	{
		// Fixed: all but last are 'size_field'; the last value is encoded in header (we'll re-read it)
		// Re-parse last_orig_len from header (we didn't store it above; rewind and parse again cleanly)
		// Simpler approach: scan header again quickly to fetch the last_orig_len.
		std::size_t off2 = 0;
		auto		pr0 = uleb128_decode_stream( toc_header, off2 );
		off2 = pr0.second;	// n_runs
		auto prK = uleb128_decode_stream( toc_header, off2 );
		off2 = prK.second;
		std::uint64_t K2 = prK.first;
		for ( std::size_t i = 0; i < K2; ++i )
		{
			auto prSym = uleb128_decode_stream( toc_header, off2 );
			off2 = prSym.second;
			auto prLen = uleb128_decode_stream( toc_header, off2 );
			off2 = prLen.second;
		}
		auto prX = uleb128_decode_stream( toc_header, off2 );
		off2 = prX.second;	// k_method
		auto prL = uleb128_decode_stream( toc_header, off2 );
		off2 = prL.second;
		std::uint32_t last_orig_len = static_cast<std::uint32_t>( prL.first );

		for ( std::uint32_t i = 0; i < nblocks; ++i )
		{
			if ( i + 1 == nblocks )
				out.orig_lens.push_back( last_orig_len );
			else
				out.orig_lens.push_back( size_field );
		}
	}

	// Stream C: Elias–Fano(N = total_payload, M = nblocks, L = choose)
	{
		EFParams ep = ef_choose_params( total_payload, nblocks );
		out.payload_ends = ef_read_positions( br, ep );
	}


	// Final sanity
	if ( out.payload_ends.size() != nblocks )
		throw std::runtime_error( "read_toc: payload_ends size mismatch" );
}

// Compression (CDC variant; header carries the mode bit)
// =========================================
ByteVector compress_blocks_cdc( const ByteVector& data, std::size_t min_size, std::size_t avg_size, std::size_t max_size )
{
	// Build the TOC‑based container for FastCDC chunking.
	const ByteVector magic = { 'K', 'O', 'L', 'R' };
	// Determine boundaries via strict FastCDC, merging orphan tail.
	BoundaryList boundaries = cdc_fast_boundaries_strict( data, min_size, avg_size, max_size, true );
	std::size_t	 nblocks = boundaries.size();
	// Prepare arrays to collect method IDs, orig lengths and payloads.
	std::vector<std::uint8_t> method_ids;
	method_ids.reserve( nblocks );
	std::vector<std::uint32_t> orig_lens;
	orig_lens.reserve( nblocks );
	std::vector<std::vector<std::uint8_t>> payloads;
	payloads.reserve( nblocks );
	std::vector<std::uint32_t> payload_lens;
	payload_lens.reserve( nblocks );
	// Select best coding for each block.
	for ( std::size_t bi = 0; bi < boundaries.size(); ++bi )
	{
		const auto& [ start, end ] = boundaries[ bi ];
		ByteSpan block { data.data() + start, end - start };
		auto	 best = _select_best( block );
		method_ids.push_back( best.method_id );
		orig_lens.push_back( static_cast<std::uint32_t>( block.size() ) );
		payload_lens.push_back( static_cast<std::uint32_t>( best.payload.size() ) );
		payloads.emplace_back( std::move( best.payload ) );

		if ( G_PROGRESS && ( ( bi + 1 ) % 1 == 0 ) )
		{
			std::fprintf( stderr, "\r[Fast CDC Compress] block %zu/%zu ...", bi + 1, boundaries.size() );
			std::fflush( stderr );
		}
	}
	if ( G_PROGRESS )
	{
		std::fprintf( stderr, "\r[Fast CDC Compress]   block %zu/%zu done.\n", boundaries.size(), boundaries.size() );
	}

	// ---- Build TOC using unified helper ----
	ByteVector	  toc_header;
	ByteVector	  toc_bits;
	std::uint32_t toc_bitlen = 0;
	std::uint32_t total_payload = 0;
	write_toc( MODE_CDC, static_cast<std::uint32_t>( avg_size ), method_ids, orig_lens, payload_lens, toc_header, toc_bits, toc_bitlen, total_payload );

	// ---- Assemble container ----
	ByteVector out;
	out.insert( out.end(), magic.begin(), magic.end() );
	// packed mode/size (MODE_CDC + avg_size)
	write_le32( out, pack_mode_and_size( MODE_CDC, static_cast<std::uint32_t>( avg_size ) ) );
	// total original length
	write_le32( out, static_cast<std::uint32_t>( data.size() ) );
	// number of blocks (u16)
	if ( nblocks > 0xFFFFu )
		throw std::runtime_error( "Too many blocks for 16-bit field" );
	out.push_back( static_cast<std::uint8_t>( nblocks & 0xFFu ) );
	out.push_back( static_cast<std::uint8_t>( ( nblocks >> 8 ) & 0xFFu ) );
	// toc header length, toc bit length, total payload (ULEB128)
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( toc_header.size() ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( toc_bitlen ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( total_payload ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	// append toc_header and toc_bits
	out.insert( out.end(), toc_header.begin(), toc_header.end() );
	out.insert( out.end(), toc_bits.begin(), toc_bits.end() );
	// append payloads
	for ( std::size_t i = 0; i < nblocks; ++i )
		out.insert( out.end(), payloads[ i ].begin(), payloads[ i ].end() );

	return out;
}

// =========================================
// Compression (fixed-size variant; header carries the mode bit)
// =========================================
ByteVector compress_blocks_fixed( const ByteVector& data, std::size_t block_size )
{
	// Build the TOC-based container for fixed block size.
	const ByteVector magic = { 'K', 'O', 'L', 'R' };
	// Determine boundaries via fixed boundaries and merge tiny tail if needed.
	BoundaryList boundaries = fixed_boundaries( data, block_size );
	// Merge small tail block (same logic as Python: if last block shorter than min(block_size/2, 128), merge into previous)
	if ( boundaries.size() >= 2 )
	{
		auto [ ls, le ] = boundaries.back();
		std::size_t last_len = le - ls;
		std::size_t min_tail = std::min<std::size_t>( block_size / 2, static_cast<std::size_t>( 128 ) );
		if ( last_len < min_tail )
		{
			auto [ ps, pe ] = boundaries[ boundaries.size() - 2 ];
			boundaries[ boundaries.size() - 2 ] = { ps, le };
			boundaries.pop_back();
		}
	}
	std::size_t nblocks = boundaries.size();
	// Collect method IDs, orig lengths and payloads.
	std::vector<std::uint8_t> method_ids;
	method_ids.reserve( nblocks );
	std::vector<std::uint32_t> orig_lens;
	orig_lens.reserve( nblocks );
	std::vector<ByteVector> payloads;
	payloads.reserve( nblocks );
	std::vector<std::uint32_t> payload_lens;
	payload_lens.reserve( nblocks );
	for ( std::size_t bi = 0; bi < boundaries.size(); ++bi )
	{
		const auto& [ start, end ] = boundaries[ bi ];
		ByteSpan block { data.data() + start, end - start };
		auto	 best = _select_best( block );
		method_ids.push_back( best.method_id );
		orig_lens.push_back( static_cast<std::uint32_t>( block.size() ) );
		payload_lens.push_back( static_cast<std::uint32_t>( best.payload.size() ) );
		payloads.emplace_back( std::move( best.payload ) );

		if ( G_PROGRESS && ( ( bi + 1 ) % 1 == 0 ) )
		{
			std::fprintf( stderr, "\r[FIXED Compress] block %zu/%zu ...", bi + 1, boundaries.size() );
			std::fflush( stderr );
		}
	}
	if ( G_PROGRESS )
	{
		std::fprintf( stderr, "\r[FIXED Compress] block %zu/%zu done.\n", boundaries.size(), boundaries.size() );
	}
	// Compute total payload.
	std::uint32_t total_payload = 0;
	for ( auto l : payload_lens )
		total_payload = static_cast<std::uint32_t>( total_payload + l );
	
	// ---- Build TOC using unified helper ----
	ByteVector toc_header;
	ByteVector toc_bits;
	std::uint32_t toc_bitlen = 0;
	
	write_toc( MODE_FIXED, static_cast<std::uint32_t>(block_size),
			   method_ids, orig_lens, payload_lens,
			   toc_header, toc_bits, toc_bitlen, total_payload );
// Assemble container.
	ByteVector out;
	out.reserve( 16 + toc_header.size() + toc_bits.size() + total_payload );
	out.insert( out.end(), magic.begin(), magic.end() );
	// packed mode/size
	write_le32( out, pack_mode_and_size( MODE_FIXED, static_cast<std::uint32_t>( block_size ) ) );
	// total original length
	write_le32( out, static_cast<std::uint32_t>( data.size() ) );
	// number of blocks (u16)
	if ( nblocks > 0xFFFFu )
		throw std::runtime_error( "Too many blocks for 16-bit field" );
	out.push_back( static_cast<std::uint8_t>( nblocks & 0xFFu ) );
	out.push_back( static_cast<std::uint8_t>( ( nblocks >> 8 ) & 0xFFu ) );
	// toc header length, toc bit length, total payload (uleb128)
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( toc_header.size() ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( toc_bitlen ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	{
		auto tmp = uleb128_encode( static_cast<std::uint64_t>( total_payload ) );
		out.insert( out.end(), tmp.begin(), tmp.end() );
	}
	// append toc_header and toc_bits.
	out.insert( out.end(), toc_header.begin(), toc_header.end() );
	out.insert( out.end(), toc_bits.begin(), toc_bits.end() );
	// append payloads.
	for ( std::size_t i = 0; i < nblocks; ++i )
	{
		const auto& payload = payloads[ i ];
		out.insert( out.end(), payload.begin(), payload.end() );
	}
	return out;
}

// =========================================
// Decompression (compatible; reads mode bit)
// =========================================
ByteVector decompress( const ByteVector& data )
{
	// Decompress container built using the TOC format (no backward compatibility).
	std::size_t pos = 0;
	// verify magic
	if ( data.size() < 4 || std::string( data.begin(), data.begin() + 4 ) != "KOLR" )
		throw std::runtime_error( "Invalid magic" );
	pos = 4;
	// read packed (mode,size)
	if ( pos + 4 > data.size() )
		throw std::runtime_error( "decompress: truncated packed field" );
	std::uint32_t packed = read_le32( data, pos );
	pos += 4;
	std::uint32_t mode = 0;
	std::uint32_t size_field = 0;
	unpack_mode_and_size( packed, mode, size_field );
	// read total original length
	if ( pos + 4 > data.size() )
		throw std::runtime_error( "decompress: truncated total length" );
	std::uint32_t total_len = read_le32( data, pos );
	pos += 4;
	// read nblocks (u16)
	if ( pos + 2 > data.size() )
		throw std::runtime_error( "decompress: truncated nblocks" );
	std::uint16_t nblocks = static_cast<std::uint16_t>( data[ pos ] | ( static_cast<std::uint16_t>( data[ pos + 1 ] ) << 8 ) );
	pos += 2;
	// read toc header length, toc bit length, total payload via uleb128
	auto [ toc_hdr_len, n1 ] = uleb128_decode_stream( data, pos );
	pos = n1;
	auto [ toc_bitlen, n2 ] = uleb128_decode_stream( data, pos );
	pos = n2;
	auto [ total_payload, n3 ] = uleb128_decode_stream( data, pos );
	pos = n3;
	// grab toc_header
	if ( pos + toc_hdr_len > data.size() )
		throw std::runtime_error( "decompress: truncated toc header" );
	ByteVector toc_header( data.begin() + pos, data.begin() + pos + static_cast<std::size_t>( toc_hdr_len ) );
	pos += static_cast<std::size_t>( toc_hdr_len );
	// toc_bits length in bytes = ceil(toc_bitlen/8)
	std::size_t toc_bytes = static_cast<std::size_t>( ( toc_bitlen + 7u ) >> 3 );
	if ( pos + toc_bytes > data.size() )
		throw std::runtime_error( "decompress: truncated toc bits" );
	ByteVector toc_bits( data.begin() + pos, data.begin() + pos + toc_bytes );
	pos += toc_bytes;

	// === Use unified read_toc(...) to parse header + decode three streams ===
	// Pass total_payload explicitly so read_toc can compute EF parameters.  Without
	// supplying total_payload the function cannot reconstruct the payload_ends sequence.
	TocDecoded td;
	read_toc( mode, size_field, static_cast<std::uint32_t>( nblocks ), static_cast<std::uint32_t>( total_payload ), toc_header, toc_bits, static_cast<std::uint32_t>( toc_bitlen ), td );

	if ( G_PROGRESS )
	{
		std::cerr << "[Decompress] mode=" << ( mode == MODE_CDC ? "Fast CDC" : "FIXED" ) << " blocks=" << nblocks << " payload=" << total_payload << "Byte"  << " toc_bits=" << toc_bitlen << '\n';
	}

	const std::vector<std::uint8_t>&  method_ids = td.method_ids;
	const std::vector<std::uint32_t>& orig_lens = td.orig_lens;
	const std::vector<std::uint32_t>& payload_ends = td.payload_ends;
	// read payload area and decompress each block
	if ( pos + static_cast<std::size_t>( total_payload ) > data.size() )
		throw std::runtime_error( "decompress: truncated payload area" );
	ByteVector out;
	out.reserve( total_len );
	std::size_t	  payload_pos = pos;
	std::uint32_t prev_end = 0u;
	for ( std::size_t i = 0; i < nblocks; ++i )
	{
		std::uint32_t cur_end = payload_ends[ i ];
		if ( cur_end < prev_end )
			throw std::runtime_error( "decompress: non‑monotone EF ends" );
		std::uint32_t plen = static_cast<std::uint32_t>( cur_end - prev_end );
		std::uint32_t ol = orig_lens[ i ];
		std::uint8_t  mid = method_ids[ i ];
		if ( payload_pos + plen > data.size() )
			throw std::runtime_error( "decompress: truncated block payload" );
		ByteVector payload( data.begin() + payload_pos, data.begin() + payload_pos + plen );
		payload_pos += plen;
		ByteVector block = _decode_by_id( mid, payload, static_cast<std::size_t>( ol ) );
		if ( block.size() != static_cast<std::size_t>( ol ) )
			throw std::runtime_error( "decompress: block length mismatch" );
		out.insert( out.end(), block.begin(), block.end() );
		prev_end = cur_end;

		if ( G_PROGRESS && ( ( i + 1 ) % 1 == 0 ) )
		{
			std::fprintf( stderr, "\r[Decompress] block %zu/%zu ...", i + 1, nblocks );
			std::fflush( stderr );
		}
	}
	if ( G_PROGRESS )
	{
		std::fprintf( stderr, "\r[Decompress] block %zu/%zu done.\n", nblocks, nblocks );
	}
	if ( out.size() != static_cast<std::size_t>( total_len ) )
		throw std::runtime_error( "decompress: output length mismatch" );
	if ( prev_end != total_payload )
		throw std::runtime_error( "decompress: EF ends do not match total payload" );
	if ( payload_pos != pos + static_cast<std::size_t>( total_payload ) )
		throw std::runtime_error( "decompress: payload length mismatch" );
	if ( payload_pos != data.size() )
	{
		std::size_t extra = data.size() - payload_pos;
		throw std::runtime_error( "decompress: extra trailing " + std::to_string( extra ) + " bytes after container end" );
	}
	return out;
}

namespace SelfTest
{
	// === Block summary & helpers (declarations) ===
	struct BlockSummary
	{
		std::uint8_t  method_id;
		std::uint32_t orig_len;
		std::uint32_t payload_len;
	};

	// === Block summary & helpers (definition) ===
	std::vector<BlockSummary> parse_container_blocks( const ByteVector& data )
	{
		std::vector<BlockSummary> blocks;
		std::size_t				  pos = 0;
		// verify magic
		if ( data.size() < 4 || std::string( data.begin(), data.begin() + 4 ) != "KOLR" )
			throw std::runtime_error( "parse: invalid magic" );
		pos = 4;
		// read packed mode and size
		if ( pos + 4 > data.size() )
			throw std::runtime_error( "parse: truncated packed field" );
		std::uint32_t packed = read_le32( data, pos );
		pos += 4;
		std::uint32_t mode = 0, size_field = 0;
		unpack_mode_and_size( packed, mode, size_field );
		bool is_cdc = ( mode == MODE_CDC );
		// read total_len
		if ( pos + 4 > data.size() )
			throw std::runtime_error( "parse: truncated total length" );
		std::uint32_t total_len = read_le32( data, pos );
		pos += 4;
		// read nblocks
		if ( pos + 2 > data.size() )
			throw std::runtime_error( "parse: truncated nblocks" );
		std::uint16_t nblocks = static_cast<std::uint16_t>( data[ pos ] | ( static_cast<std::uint16_t>( data[ pos + 1 ] ) << 8 ) );
		pos += 2;
		// read toc header length, toc bit length, total payload
		auto [ toc_hdr_len, n1 ] = uleb128_decode_stream( data, pos );
		pos = n1;
		auto [ toc_bitlen, n2 ] = uleb128_decode_stream( data, pos );
		pos = n2;
		auto [ total_payload, n3 ] = uleb128_decode_stream( data, pos );
		pos = n3;
		if ( pos + toc_hdr_len > data.size() )
			throw std::runtime_error( "parse: truncated toc header" );
		ByteVector toc_header( data.begin() + pos, data.begin() + pos + static_cast<std::size_t>( toc_hdr_len ) );
		pos += static_cast<std::size_t>( toc_hdr_len );
		std::size_t toc_bytes = static_cast<std::size_t>( ( toc_bitlen + 7u ) >> 3 );
		if ( pos + toc_bytes > data.size() )
			throw std::runtime_error( "parse: truncated toc bits" );
		ByteVector toc_bits( data.begin() + pos, data.begin() + pos + toc_bytes );
		pos += toc_bytes;
		// parse toc_header
		std::size_t hpos = 0;
		auto		read_uleb = [ & ]( std::size_t& off ) -> std::uint64_t {
			   auto [ v, l ] = uleb128_decode_stream( toc_header, off );
			   off = l;
			   return v;
		};
		std::uint64_t						   n_runs64 = read_uleb( hpos );
		std::size_t							   n_runs = static_cast<std::size_t>( n_runs64 );
		std::uint64_t						   K64 = read_uleb( hpos );
		std::size_t							   K = static_cast<std::size_t>( K64 );
		std::unordered_map<std::uint32_t, int> lengths;
		lengths.reserve( K );
		for ( std::size_t i = 0; i < K; ++i )
		{
			std::uint64_t sym = read_uleb( hpos );
			std::uint64_t len = read_uleb( hpos );
			lengths[ static_cast<std::uint32_t>( sym ) ] = static_cast<int>( len );
		}
		auto [ enc_tbl, dec_tbl, maxlen ] = huff_canonical( lengths );
		std::uint64_t k_run64 = read_uleb( hpos );
		int			  k_run = static_cast<int>( k_run64 );
		int			  k_orig = 0;
		std::uint32_t last_orig = 0;
		if ( is_cdc )
		{
			std::uint64_t k2 = read_uleb( hpos );
			k_orig = static_cast<int>( k2 );
		}
		else
		{
			std::uint64_t last64 = read_uleb( hpos );
			last_orig = static_cast<std::uint32_t>( last64 );
		}
		// decode toc bits
		BitReader				   br( toc_bits );
		std::vector<std::uint32_t> run_syms = huff_decode_symbols( br, dec_tbl, maxlen, n_runs );
		std::vector<std::uint32_t> run_lens = rice_read_n( br, k_run, n_runs );
		// expand method_ids
		std::vector<std::uint8_t> method_ids;
		method_ids.reserve( nblocks );
		std::size_t idx = 0;
		for ( std::size_t i = 0; i < n_runs; ++i )
		{
			std::uint32_t sym = run_syms[ i ];
			std::uint32_t rl = run_lens[ i ];
			for ( std::uint32_t j = 0; j < rl; ++j )
			{
				if ( idx >= static_cast<std::size_t>( nblocks ) )
					throw std::runtime_error( "parse: runs exceed nblocks" );
				method_ids.push_back( static_cast<std::uint8_t>( sym ) );
				++idx;
			}
		}
		if ( method_ids.size() != static_cast<std::size_t>( nblocks ) )
			throw std::runtime_error( "parse: run-length expansion mismatch" );
		// decode orig_lens
		std::vector<std::uint32_t> orig_lens;
		orig_lens.reserve( nblocks );
		if ( is_cdc )
		{
			std::vector<std::uint32_t> deltas = rice_read_n( br, k_orig, nblocks );
			for ( std::size_t i = 0; i < nblocks; ++i )
			{
				std::int32_t diff = zigzag_decode_32( deltas[ i ] );
				std::int32_t base = static_cast<std::int32_t>( size_field );
				std::int32_t ol = base + diff;
				if ( ol < 0 )
					throw std::runtime_error( "parse: negative orig_len" );
				orig_lens.push_back( static_cast<std::uint32_t>( ol ) );
			}
		}
		else
		{
			std::uint32_t block_size32 = static_cast<std::uint32_t>( size_field );
			for ( std::size_t i = 0; i < nblocks; ++i )
			{
				if ( i + 1 == nblocks )
					orig_lens.push_back( last_orig );
				else
					orig_lens.push_back( block_size32 );
			}
		}
		// decode positions
		std::vector<std::uint32_t> ends = ef_read_positions( br, ef_choose_params( static_cast<std::uint32_t>( total_payload ), nblocks ) );
		std::vector<std::uint32_t> payload_lens;
		payload_lens.reserve( nblocks );
		std::uint32_t prev_end = 0u;
		for ( std::size_t i = 0; i < nblocks; ++i )
		{
			std::uint32_t e = ends[ i ];
			payload_lens.push_back( static_cast<std::uint32_t>( e - prev_end ) );
			prev_end = e;
		}
		// assemble block summaries
		blocks.reserve( nblocks );
		for ( std::size_t i = 0; i < nblocks; ++i )
		{
			blocks.push_back( BlockSummary { method_ids[ i ], orig_lens[ i ], payload_lens[ i ] } );
		}
		// ensure position matches container length exactly
		std::size_t payload_pos = pos + static_cast<std::size_t>( total_payload );
		if ( payload_pos != data.size() )
			throw std::runtime_error( "parse: extra trailing bytes" );
		return blocks;
	}

	std::string format_model_histogram(const std::vector<BlockSummary>& blocks)
	{
		std::map<std::string, int> kv_map; // 排序输出更稳定
		for (const auto& b : blocks) ++kv_map[method_name_from_id(b.method_id)];

		if (kv_map.empty()) return "-";
		std::ostringstream oss;
		bool first = true;
		for (const auto& kv : kv_map) {
			if (!first) oss << ", ";
			oss << kv.first << " - " << kv.second;
			first = false;
		}
		return oss.str();
	}

	std::string _summarize_container_methods(const ByteVector& container)
	{
		return format_model_histogram(parse_container_blocks(container));
	}

	// 简要头部信息（仅打印三张表所需字段）
	struct TocBrief 
	{
		bool		is_cdc = false;
		std::uint32_t size31 = 0;	  // FIXED: block_size; CDC: avg_size
		std::uint32_t nblocks = 0;
		std::uint64_t toc_hdr_len = 0;
		std::uint64_t toc_bitlen  = 0;
		std::uint64_t total_payload = 0;
	};

	// 安全解析容器头；失败时返回 nullopt，不抛异常
	std::optional<TocBrief> get_toc_brief(const ByteVector& container)
	{
		try {
			std::size_t pos = 0;
			if (container.size() < 4 || std::string(container.begin(), container.begin()+4) != "KOLR")
				return std::nullopt;
			pos = 4;
			if (pos + 4 > container.size()) return std::nullopt;
			std::uint32_t packed = read_le32(container, pos);
			pos += 4;
			std::uint32_t mode=0, size31=0;
			unpack_mode_and_size(packed, mode, size31);

			if (pos + 4 > container.size()) return std::nullopt;
			(void)read_le32(container, pos); // total_in（本表不打印，用不到）
			pos += 4;

			if (pos + 2 > container.size()) return std::nullopt;
			std::uint32_t nblocks = std::uint32_t(container[pos]) | (std::uint32_t(container[pos+1])<<8);
			pos += 2;

			auto rd = [&](std::size_t& off)->std::uint64_t {
				auto pr = uleb128_decode_stream(container, off);
				off = pr.second; return pr.first;
			};
			std::uint64_t toc_hdr_len  = rd(pos);
			std::uint64_t toc_bitlen   = rd(pos);
			std::uint64_t total_pay	= rd(pos);

			TocBrief tb;
			tb.is_cdc = (mode == MODE_CDC);
			tb.size31 = size31;
			tb.nblocks = nblocks;
			tb.toc_hdr_len = toc_hdr_len;
			tb.toc_bitlen  = toc_bitlen;
			tb.total_payload = total_pay;
			return tb;
		} catch (...) {
			return std::nullopt;
		}
	}

	// Forward declaration to allow toc_brief() to call summarize_container_methods()
	// before its definition.  Without this, the compiler complains that
	// summarize_container_methods is not declared when used in toc_brief.
	// Produce a concise TOC summary line (for self-test / diagnostics)
	std::string summarize_container_methods( const ByteVector& container )
	{
		try {
			std::size_t pos = 0;
			if (container.size() < 4 || std::string(container.begin(), container.begin()+4) != "KOLR")
				return "invalid-magic";
			pos = 4;
			// packed
			if (pos + 4 > container.size()) return "truncated-pack";
			std::uint32_t packed = read_le32(container, pos);
			pos += 4; 
			std::uint32_t mode=0, size31=0;
			unpack_mode_and_size(packed, mode, size31);
			// total_input_len
			if (pos + 4 > container.size()) 
				return "truncated-total";
			std::uint32_t total_in = read_le32(container, pos);
			pos += 4; 
			// nblocks
			if (pos + 2 > container.size()) 
				return "truncated-nblocks";
			std::uint32_t nblocks = std::uint32_t(container[pos]) | (std::uint32_t(container[pos+1])<<8);
			pos += 2;
			// toc hdr len / bitlen / total_payload (ULEB128)
			auto rd = [&]() -> std::uint64_t { auto pr = uleb128_decode_stream(container, pos); pos = pr.second; return pr.first; };
			std::uint64_t toc_hdr_len = rd();
			std::uint64_t toc_bitlen  = rd();
			std::uint64_t total_pay   = rd();
			// Compose
			std::ostringstream oss;
			oss << (mode==MODE_CDC ? "CDC" : "FIXED") << "(size=" << size31 << "), "
				<< "N=" << total_in << ", blocks=" << nblocks
				<< ", TOC: hdr=" << toc_hdr_len << "B, bits=" << toc_bitlen
				<< ", payload=" << total_pay << "B; Methods: "
				<< _summarize_container_methods(container);
			return oss.str();
		} catch (...) {
			return "parse-error";
		}
	}

	void run_self_test()
	{
		using clock = std::chrono::high_resolution_clock;

		struct Dataset { std::string name; std::vector<std::uint8_t> data; };
		std::vector<Dataset> datasets;

		// === 数据集（与原版一致） ===
		{
			std::string para = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
							   "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
							   "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
							   "hobbit-hole, and that means comfort.";
			std::string repeated;
			for (int i=0;i<10;++i) repeated += para;
			datasets.push_back({"text", std::vector<std::uint8_t>(repeated.begin(), repeated.end())});
		}
		{
			std::string para = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
							   "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
							   "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
							   "hobbit-hole, and that means comfort.";
			std::string repeated;
			for (int i=0;i<200;++i) repeated += para;
			datasets.push_back({"text_big", std::vector<std::uint8_t>(repeated.begin(), repeated.end())});
		}
		{
			std::vector<std::uint8_t> rnd(10240);
			std::mt19937 rng(123456789);
			std::uniform_int_distribution<int> dist(0,255);
			for (auto& b : rnd) b = static_cast<std::uint8_t>(dist(rng));
			datasets.push_back({"random", std::move(rnd)});
		}
		datasets.push_back({"repetitive", std::vector<std::uint8_t>(20480, (std::uint8_t)'a')});
		{
			std::vector<std::uint8_t> v; v.reserve(20000);
			for (int i=0;i<10000;++i){ v.push_back('a'); v.push_back('b'); }
			datasets.push_back({"abab", std::move(v)});
		}
		{
			std::vector<std::uint8_t> v; v.reserve(18000);
			for (int i=0;i<6000;++i){ v.push_back('a'); v.push_back('b'); v.push_back('c'); }
			datasets.push_back({"abcabc", std::move(v)});
		}
		datasets.push_back({"zero", std::vector<std::uint8_t>(16384, 0)});
		{
			std::vector<std::uint8_t> v(8192);
			for (std::size_t i=0;i<v.size();++i) v[i] = static_cast<std::uint8_t>(i & 0xFF);
			datasets.push_back({"ramp", std::move(v)});
		}
		{
			std::string s = "数据压缩 data compression 可逆性 reversibility —— Kolmogorov-style.";
			std::string rep; for (int i=0;i<200;++i) rep += s;
			datasets.push_back({"utf8_mixed", std::vector<std::uint8_t>(rep.begin(), rep.end())});
		}

		const std::vector<std::string> mode_names = {"FIXED","FastCDC"};

		struct Row {
			std::string dataset;
			std::string mode;
			std::size_t orig_size = 0;
			std::size_t blob_size = 0;
			double ratio = 0.0;
			double comp_ms = 0.0;
			double decomp_ms = 0.0;
			std::string ok = "OK";
			ByteVector blob; // 用于后续表B/表C解析
		};
		std::vector<Row> rows; rows.reserve(datasets.size()*2);

		struct Best { double ratio = 1e100; std::string mode; std::size_t size=0; double c=0,d=0; };
		std::unordered_map<std::string, Best> best_of;

		const std::size_t fixed_block = 2048;
		const std::size_t avg = std::max<std::size_t>(64, fixed_block);
		const std::size_t half = (avg>=2)?(avg/2):64;
		const std::size_t min_size = std::max<std::size_t>(64, std::min<std::size_t>(avg, half));
		const std::size_t max_size = std::max<std::size_t>(avg, avg*2);

		auto do_one = [&](const Dataset& ds, bool use_cdc){
			Row r;
			r.dataset = ds.name;
			r.mode = use_cdc ? mode_names[1] : mode_names[0];
			r.orig_size = ds.data.size();

			try {
				auto t0 = clock::now();
				ByteVector blob = use_cdc ? compress_blocks_cdc(ds.data, min_size, avg, max_size)
										  : compress_blocks_fixed(ds.data, fixed_block);
				auto t1 = clock::now();
				r.comp_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()/1000.0;

				r.blob_size = blob.size();
				r.ratio = (r.orig_size==0) ? 1.0 : double(r.blob_size)/double(r.orig_size);

				// 解压校验
				t0 = clock::now();
				ByteVector rec = decompress(blob);
				t1 = clock::now();
				r.decomp_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()/1000.0;
				if (rec != ds.data) r.ok = "FAIL";

				r.blob = std::move(blob);
			} catch (const std::exception& e) {
				r.ok = "ERR";
				r.ratio = std::numeric_limits<double>::infinity();
				std::cerr << "[SelfTest] FATAL: " << e.what() << "\n";
			} catch (...) {
				r.ok = "ERR";
				r.ratio = std::numeric_limits<double>::infinity();
				std::cerr << "[SelfTest] FATAL: unknown exception\n";
			}

			// 更新best
			if (r.ok=="OK" && r.ratio < best_of[r.dataset].ratio) {
				best_of[r.dataset] = {r.ratio, r.mode, r.blob_size, r.comp_ms, r.decomp_ms};
			}

			rows.push_back(std::move(r));
		};

		for (const auto& ds : datasets) {
			do_one(ds, /*use_cdc=*/false);
			do_one(ds, /*use_cdc=*/true);
		}

		// =========================
		// 表A：压缩/解压结果总表
		// =========================
		std::cout << std::left << std::setw(12) << "Dataset"
				  << std::left << std::setw(10) << "Mode"
				  << std::right<< std::setw(12) << "Unfolded"
				  << std::setw(12) << "Folded"
				  << std::setw(8)  << "Ratio"
				  << std::setw(14) << "Compress(ms)"
				  << std::setw(14) << "Decomp(ms)"
				  << std::setw(8)  << "OK/ERR" << "\n";
		std::cout << std::string(90,'-') << "\n";
		for (const auto& r : rows) {
			std::cout << std::left  << std::setw(12) << r.dataset
					  << std::left  << std::setw(10) << r.mode
					  << std::right << std::setw(12) << r.orig_size
					  << std::setw(12) << r.blob_size
					  << std::setw(8)  << std::fixed << std::setprecision(3) << r.ratio
					  << std::setw(14) << std::fixed << std::setprecision(3) << r.comp_ms
					  << std::setw(14) << std::fixed << std::setprecision(3) << r.decomp_ms
					  << std::setw(8)  << r.ok << "\n";
		}

		// =========================
		// 表B：容器与TOC摘要（精简字段）
		// =========================
		std::cout << "\n";
		std::cout << std::left << std::setw(12) << "Dataset"
				  << std::left << std::setw(10) << "Mode"
				  << std::left << std::setw(16) << "Container"
				  << std::right<< std::setw(8)  << "Blocks"
				  << std::setw(10) << "TOC_hdrB"
				  << std::setw(10) << "TOC_bits"
				  << std::setw(12) << "PayloadB"
				  << "\n";
		std::cout << std::string(88,'-') << "\n";
		for (const auto& r : rows) {
			std::string cbrief = "-";
			std::uint32_t blocks = 0;
			std::uint64_t hdrB=0, bits=0, pay=0;

			if (!r.blob.empty()) {
				if (auto tb = get_toc_brief(r.blob)) {
					std::ostringstream oss;
					oss << (tb->is_cdc ? "CDC(" : "FIXED(") << "size=" << tb->size31 << ")";
					cbrief = oss.str();
					blocks = tb->nblocks; hdrB = tb->toc_hdr_len; bits = tb->toc_bitlen; pay = tb->total_payload;
				} else {
					// 兜底：用 summarize_container_methods 的字符串（不拆字段）
					cbrief = summarize_container_methods(r.blob);
				}
			}

			std::cout << std::left  << std::setw(12) << r.dataset
					  << std::left  << std::setw(10) << r.mode
					  << std::left  << std::setw(16) << cbrief
					  << std::right << std::setw(8)  << blocks
					  << std::setw(10) << hdrB
					  << std::setw(10) << bits
					  << std::setw(12) << pay
					  << "\n";
		}

		// =========================
		// 表C：方法直方图（来自 parse_container_blocks）
		// =========================
		std::cout << "\n";
		std::cout << std::left << std::setw(12) << "Dataset"
				  << std::left << std::setw(10) << "Mode"
				  << std::left << std::setw(48) << "Methods(histogram)"
				  << "\n";
		std::cout << std::string(72,'-') << "\n";
		for (const auto& r : rows) {
			std::string hist = "-";
			try {
				if (!r.blob.empty()) {
					hist = format_model_histogram(parse_container_blocks(r.blob));
				}
			} catch (...) {
				hist = "parse-error";
			}
			if (hist.size() > 46) { // 控制列宽
				hist.resize(46);
				hist += "…";
			}
			std::cout << std::left << std::setw(12) << r.dataset
					  << std::left << std::setw(10) << r.mode
					  << std::left << std::setw(48) << hist
					  << "\n";
		}

		// （可选）保留原先“best of”结尾总结
		std::cout << "\nBest mode per dataset (by ratio):\n";
		for (const auto& kv : best_of) {
			const auto& b = kv.second;
			std::cout << "  " << std::left << std::setw(12) << kv.first
					  << " -> " << std::left << std::setw(10) << b.mode
					  << " size=" << b.size
					  << " ratio=" << std::fixed << std::setprecision(3) << b.ratio
					  << " comp(ms)=" << std::fixed << std::setprecision(3) << b.c
					  << " decomp(ms)=" << std::fixed << std::setprecision(3) << b.d
					  << "\n";
		}
	}

}  // namespace SelfTest


#define BUILD_KOLM_MAIN
#ifdef BUILD_KOLM_MAIN

// ---------------------------------------------
// Tiny argparse-like CLI
// ---------------------------------------------
struct ArgumentParser
{
	// flags
	bool help = false, decompress = false, fastcdc = false, experiment = false, progress = false;
	bool no_lz77 = false;
	// options
	std::string container = "TOC";	// TOC|SIMPLE
	std::string input, output, only;
	std::size_t block = 2048;  // align Python default

	bool parse( int argc, char** argv, std::string& err )
	{
		auto need_val = [ & ]( int& i ) -> const char* {
			if ( i + 1 >= argc )
			{
				err = std::string( "Missing value for " ) + argv[ i ];
				return nullptr;
			}
			return argv[ ++i ];
		};
		for ( int i = 1; i < argc; ++i )
		{
			std::string a = argv[ i ];

			// help
			if ( a == "-h" || a == "--help" )
			{
				help = true;
				continue;
			}

			// decompression / i-o
			else if ( a == "-d" || a == "--decompress" )
			{
				decompress = true;
			}
			else if ( a == "-i" || a == "--input" )
			{
				if ( auto v = need_val( i ) )
					input = v;
				else
					return false;
			}
			else if ( a == "-o" || a == "--output" )
			{
				if ( auto v = need_val( i ) )
					output = v;
				else
					return false;
			}

			// block / FastCDC
			else if ( a == "-b" || a == "--block" )
			{
				if ( auto v = need_val( i ) )
				{
					try
					{
						block = static_cast<std::size_t>( std::stoull( v ) );
					}
					catch ( ... )
					{
						err = "Invalid numeric value for --block";
						return false;
					}
				}
				else
					return false;
			}
			else if ( a == "--FastCDC" || a == "--fastcdc" )
			{
				fastcdc = true;
			}

			// experiment / progress
			else if ( a == "--experiment" )
			{
				experiment = true;
			}
			else if ( a == "--progress" )
			{
				progress = true;
			}

			// model toggles
			else if ( a == "--no-lz77" )
			{
				no_lz77 = true;
			}
			else if ( a == "--only" )
			{
				if ( auto v = need_val( i ) )
					only = v;
				else
					return false;
			}

			// container
			else if ( a == "--container" )
			{
				if ( auto v = need_val( i ) )
				{
					std::string s = v;
					if ( s == "TOC" || s == "toc" )
						container = "TOC";
					else if ( s == "SIMPLE" || s == "simple" )
						container = "SIMPLE";
					else
					{
						err = "Unknown --container: " + s + " (expected TOC|SIMPLE)";
						return false;
					}
				}
				else
					return false;
			}
			else if ( !a.empty() && a[ 0 ] == '-' )
			{
				err = "Unknown option: " + a;
				return false;
			}
			else
			{
				// bare input (compat with `prog data.bin`)
				if ( input.empty() )
					input = a;
				else
				{
					err = "Unexpected positional argument: " + a;
					return false;
				}
			}
		}
		return true;
	}
};


static void print_help( const char* prog )
{
	std::cerr << "Kolmogorov researched compressor (C++)\n\n"
				 "Usage:\n"
			  << "  " << prog << " [options] <input>\n"
			  << "  " << prog << " --experiment [--no-lz77] [--only <name|id>] [--progress]\n\n"
			  << "Modes:\n"
			  << "  -d, --decompress              Decompress container\n"
			  << "  -o, --output <file>           Output file\n"
			  << "  -b, --block <N>               FIXED block size (default 8192) or FastCDC avg_size\n"
			  << "      --FastCDC, --fastcdc      Use Fast Content-Defined Chunking (avg_size = --block)\n"
			  << "      --experiment              Run built-in experiment (self-test) and exit\n"
			  << "      --no-lz77                 Disable LZ77 candidate\n"
			  << "      --container TOC|SIMPLE    Container format (default TOC)\n"
			  << "      --only <name|id>          Only use one method (raw/xor/bbwt/.../v2 or 0..10)\n"
			  << "  -h, --help                    Show this help\n\n"
			  << "Examples:\n"
			  << "  " << prog << " data.bin\n"
			  << "  " << prog << " --FastCDC -b 16384 data.bin\n"
			  << "  " << prog << " --no-lz77 data.bin\n"
			  << "  " << prog << " --only v2 data.bin\n"
			  << "  " << prog << " -d data.bin.kolr\n";
}

int main( int argc, char** argv )
{
	// 关掉 stdout 缓冲（保底看见表格）
#if defined( _WIN32 )
	setvbuf( stdout, nullptr, _IONBF, 0 );
#endif
	std::ios::sync_with_stdio( false );

	ArgumentParser cli;
	std::string	   perr;
	if ( !cli.parse( argc, argv, perr ) )
	{
		std::cerr << "[cli] " << perr << "\n";
		print_help( argv[ 0 ] );
		return 1;
	}
	if ( cli.help )
	{
		print_help( argv[ 0 ] );
		return 0;
	}

	// 全局开关
	G_PROGRESS = cli.progress;
	if ( !cli.only.empty() )
	{
		if ( auto id = method_id_from_name( cli.only ) )
		{
			set_only_method( *id );
			set_no_lz77( false );
		}
		else
		{
			try
			{
				set_only_method( static_cast<std::size_t>( std::stoul( cli.only ) ) );
				set_no_lz77( false );
			}
			catch ( ... )
			{
				std::cerr << "[cli] Unknown method for --only: " << cli.only << "\n";
				return 1;
			}
		}
	}
	else
	{
		set_no_lz77( cli.no_lz77 );
		clear_only();
	}

	// 1) 先跑 experiment（无需输入文件）
	if ( cli.experiment )
	{
		SelfTest::run_self_test();
		std::cout.flush();
		std::fflush( nullptr );
		return 0;
	}

	// 2) 再检查是否缺少输入
	if ( cli.input.empty() )
	{
		print_help( argv[ 0 ] );
		return 0;
	}

	// --- 文件 I/O ---
	auto read_all = []( const std::string& path ) -> std::vector<std::uint8_t> {
		std::vector<std::uint8_t> buf;
		if ( std::FILE* f = std::fopen( path.c_str(), "rb" ) )
		{
			std::fseek( f, 0, SEEK_END );
			long sz = std::ftell( f );
			if ( sz > 0 )
			{
				std::fseek( f, 0, SEEK_SET );
				buf.resize( static_cast<std::size_t>( sz ) );
				std::fread( buf.data(), 1, buf.size(), f );
			}
			std::fclose( f );
		}
		return buf;
	};
	auto write_all = []( const std::string& path, const std::vector<std::uint8_t>& buf ) -> bool {
		if ( std::FILE* f = std::fopen( path.c_str(), "wb" ) )
		{
			if ( !buf.empty() )
				std::fwrite( buf.data(), 1, buf.size(), f );
			std::fclose( f );
			return true;
		}
		return false;
	};

	const std::string in_name = cli.input;
	const bool		  used_cli_out = !cli.output.empty();
	const std::string default_out = cli.decompress ? ( in_name.rfind( '.' ) != std::string::npos ? in_name.substr( 0, in_name.find_last_of( '.' ) ) + ".out" : in_name + ".out" ) : ( in_name + ".kolr" );
	const std::string out_name = used_cli_out ? cli.output : default_out;

	// 解压
	if ( cli.decompress )
	{
		auto blob = read_all( in_name );
		if ( blob.empty() )
		{
			std::cerr << "Failed to read input file: " << in_name << "\n";
			return 1;
		}
		auto out = decompress( blob );
		if ( !write_all( out_name, out ) )
		{
			std::cerr << "Failed to open output file: " << out_name << "\n";
			return 1;
		}
		try
		{
			std::cerr << "[info] input : " << std::filesystem::absolute( in_name ).string() << "\n";
			std::cerr << "[info] output: " << std::filesystem::absolute( out_name ).string() << ( used_cli_out ? "  (from -o)" : "  (default: input + .out)" ) << "\n";
		}
		catch ( ... )
		{
		}
		std::cout << "Decompressed " << blob.size() << " bytes to " << out.size() << " bytes \xE2\x86\x92 " << out_name << "\n";
		std::cout.flush();
		std::fflush( nullptr );
		return 0;
	}

	// 压缩
	if ( cli.container == "SIMPLE" )
	{
		std::cerr << "SIMPLE container is not supported in this build. Use --container TOC.\n";
		return 2;
	}
	auto data = read_all( in_name );
	if ( data.empty() )
	{
		std::cerr << "Failed to read input file: " << in_name << "\n";
		return 1;
	}

	std::vector<std::uint8_t> blob;
	std::string				  mode_desc;
	if ( cli.fastcdc )
	{
		std::size_t avg = std::max<std::size_t>( 64, cli.block );
		std::size_t min_size = std::max<std::size_t>( 64, ( avg >= 2 ? avg / 2 : 64 ) );
		std::size_t max_size = std::max<std::size_t>( avg, avg * 2 );
		blob = compress_blocks_cdc( data, min_size, avg, max_size );
		mode_desc = "FastCDC(min=" + std::to_string( min_size ) + ", avg=" + std::to_string( avg ) + ", max=" + std::to_string( max_size ) + ")";
	}
	else
	{
		blob = compress_blocks_fixed( data, cli.block );
		mode_desc = "FIXED(block=" + std::to_string( cli.block ) + ")";
	}

	try
	{
		std::cerr << "[info] input : " << std::filesystem::absolute( in_name ).string() << "\n";
		std::cerr << "[info] output: " << std::filesystem::absolute( out_name ).string() << ( used_cli_out ? "  (from -o)" : "  (default: input + .kolr)" ) << "\n";
	}
	catch ( ... )
	{
	}

	if ( !write_all( out_name, blob ) )
	{
		std::cerr << "Failed to open output file: " << out_name << "\n";
		return 1;
	}

	double ratio = data.empty() ? 1.0 : double( blob.size() ) / double( data.size() );
	std::cout << "[" << mode_desc << "] Compressed " << data.size() << " bytes to " << blob.size() << " bytes (ratio " << std::fixed << std::setprecision( 3 ) << ratio << ") " << out_name << "\n";
	std::cout.flush();
	std::fflush( nullptr );
	return 0;
}
#endif	// BUILD_KOLM_MAIN
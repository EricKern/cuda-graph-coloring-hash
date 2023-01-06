/**
 * \file coloring.hpp
 *
 * \brief  This file contains functions for coloring a graph in CSR format
 * 
 * @author Niklas Balterheimer, I Wayan Aditya Swardiana, Paul Grosse-Bley
 *
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <type_traits>

#include <cusparse.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace asc_hash_graph_coloring {

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

static constexpr int num_hash_functions{15};
static constexpr int first_hash_k_param{3};

/** Functor brev_classic
 * This functor will reverse the least significant bit to the most significant bit
 */
template <typename T>
struct brev_classic {
	static_assert(std::is_unsigned_v<T> && sizeof(T) == 4);
	__forceinline__ __host__ __device__ T operator()(T a) const noexcept {
		a = (a >> 16) | (a << 16); // swap halfwords
		T m{0x00ff00ff};
		a = ((a >> 8) & m) | ((a << 8) & ~m); // swap bytes
		m = m ^ (m << 4);
		a = ((a >> 4) & m) | ((a << 4) & ~m); // swap nibbles
		m = m ^ (m << 2);
		a = ((a >> 2) & m) | ((a << 2) & ~m);
		m = m ^ (m << 1);
		a = ((a >> 1) & m) | ((a << 1) & ~m);
		return a;
	}
};

/** Functor brev_classic_comp
 * This functor will compare two values while reversing their least significant bit to their most significant bit
 */
template <typename T>
struct brev_classic_comp {
	static_assert(std::is_integral_v<T>);
	__forceinline__ __host__ __device__ bool operator()(T a,
	                                                    T b) const noexcept {
		using UT = std::make_unsigned_t<T>;
		brev_classic<UT> brev{};
		return brev(static_cast<UT>(a)) < brev(static_cast<UT>(b));
	}
};

/** struct HashWrapper
 * This struct is used for wrap multiple hashes for mglbk computation
 */
struct HashWrapper {
	using value_type = uint32_t;
	value_type maximum[num_hash_functions]{};
};

/** Functor maximum_mglbk
 * This functor will compute maximum of mglbk from multiple hashes
 */
struct Max_mglbk {
	__forceinline__ __host__ __device__ HashWrapper
	operator()(HashWrapper lhs, HashWrapper rhs) const noexcept {
		HashWrapper result;

		for (int i = 0; i < num_hash_functions; ++i) {
			result.maximum[i] = (lhs.maximum[i] > rhs.maximum[i])
			                        ? lhs.maximum[i]
			                        : rhs.maximum[i];
		}
		return result;
	}
};

template <typename IndexType>
__forceinline__ __device__ std::uint32_t hash(IndexType val,
                                              int k_param) noexcept {
	auto const uval = static_cast<std::make_unsigned_t<IndexType>>(val);
	auto const divisor = (1u << k_param) - 1u;
	return (uval / divisor) << k_param | (uval % divisor);
}

__forceinline__ __device__ std::uint32_t minimum_differing_pow2(
    uint32_t hash1,
    uint32_t hash2) noexcept {
	// extract least significant differing bit
	auto const hash_xor = hash1 ^ hash2;
	return static_cast<uint32_t>(-hash_xor) & hash_xor;
}

template <typename IndexType>
class MaxColorCollisionsDist1 {
	IndexType const* row_offsets_{};
	IndexType const* cols_{};

   public:
	MaxColorCollisionsDist1(thrust::device_ptr<IndexType const> row_offsets,
	                        thrust::device_ptr<IndexType const> cols)
	    : row_offsets_{thrust::raw_pointer_cast(row_offsets)},
	      cols_{thrust::raw_pointer_cast(cols)} {
	}

	__forceinline__ __device__ HashWrapper
	operator()(IndexType row) const noexcept {
		HashWrapper local_mglbk{};

		auto const row_begin = row_offsets_[row];
		auto const row_end = row_offsets_[row + 1];

		auto const max = thrust::maximum<HashWrapper::value_type>{};

		for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
			auto const col = cols_[col_idx];

			for (auto hash_idx = 0; hash_idx < num_hash_functions; ++hash_idx) {
				auto const k_param = hash_idx + first_hash_k_param;
				auto const row_hash = hash(row, k_param);
				auto const col_hash = hash(col, k_param);

				// extract least significant differing bit
				auto const medge = minimum_differing_pow2(row_hash, col_hash);

				local_mglbk.maximum[hash_idx] =
				    max(local_mglbk.maximum[hash_idx], medge);
			}
		}
		return local_mglbk;
	}
};

/** Function coloring_distance1
 * This function will calculate minimum distance 1 color of a graph using hashing algorithm
 * @param row_offsets Row Pointer.
 * @param cols cols.
 * @param warmup Number of warm up cycles.
 */
template <typename IndexType>
int coloring_distance1(thrust::device_vector<IndexType> const& row_offsets,
                       thrust::device_vector<IndexType> const& cols,
                       int warmup) {
	auto min_mglbk{std::numeric_limits<std::uint32_t>::max()};
	int min_hash_k{};

	for (int i = 0; i < (warmup + 1); ++i) {
		auto const start = Clock::now();

		auto const medge_iter = thrust::make_counting_iterator(0);

		auto const data_ia = row_offsets.data();
		auto const data_ja = cols.data();

		auto const mglbk = thrust::transform_reduce(
		    medge_iter,
		    medge_iter + row_offsets.size() - 1,
		    MaxColorCollisionsDist1<IndexType>{data_ia, data_ja},
		    HashWrapper{},
		    Max_mglbk{});

		auto const min_it = std::min_element(std::cbegin(mglbk.maximum),
		                                     std::cend(mglbk.maximum));
		if (*min_it < min_mglbk) {
			min_mglbk = *min_it;
			min_hash_k = static_cast<int>(std::distance(
			                 std::cbegin(mglbk.maximum), min_it)) +
			             first_hash_k_param;
		}

		Duration const runtime = Clock::now() - start;
		if (i == warmup) {
			std::cout << "color1_time: " << runtime.count() << '\n';
			std::cout << "color1_colors: " << 2 * min_mglbk << '\n';
			std::cout << "color1_hash: " << min_hash_k << '\n';
		}
	}

	if (warmup == 0) {
		std::cout << "Computing distance-1 coloring with hashing method\n";
		std::cout << "Number of bits (mglb) for valid coloring = " << min_mglbk
		          << '\n';
		std::cout << "Number of colors = " << 2 * min_mglbk << '\n';
		std::cout << "Hash = " << min_hash_k << '\n';
	}

	return 2 * min_mglbk;
}

template <typename IndexType>
class MaxColorCollisionsDist2 {
	IndexType const* row_offsets_{};
	IndexType const* cols_{};
	IndexType* workspace_{};

   public:
	MaxColorCollisionsDist2(thrust::device_ptr<IndexType const> row_offsets,
	                        thrust::device_ptr<IndexType const> cols,
	                        thrust::device_ptr<IndexType> workspace)
	    : row_offsets_{thrust::raw_pointer_cast(row_offsets)},
	      cols_{thrust::raw_pointer_cast(cols)},
	      workspace_{thrust::raw_pointer_cast(workspace)} {
	}

	__forceinline__ __device__ HashWrapper
	operator()(IndexType row) const noexcept {
		HashWrapper local_mglbk{};

		auto const row_begin = row_offsets_[row];
		auto const row_end = row_offsets_[row + 1];
		auto const local_workspace_begin = row_begin + row;
		auto const local_workspace_end = row_end + row;

		auto const num_local_vertices = (row_end - row_begin) + 1;

		auto const max = thrust::maximum<HashWrapper::value_type>{};

		for (auto hash_idx = 0; hash_idx < num_hash_functions; ++hash_idx) {
			auto const k_param = hash_idx + first_hash_k_param;
			// put hashes into workspace_
			workspace_[local_workspace_begin] = hash(row, k_param);

			for (auto col_idx = row_begin; col_idx < row_end; ++col_idx) {
				workspace_[col_idx + row + 1] = hash(cols_[col_idx], k_param);
			}

			// sort hashes in workspace_
			thrust::sort(thrust::seq,
			             workspace_ + local_workspace_begin,
			             workspace_ + local_workspace_end,
			             brev_classic_comp<IndexType>{});

			// compare only neighboring hashes from workspace_
			auto hash = workspace_[local_workspace_begin];
			for (auto vertex_idx = 1; vertex_idx < num_local_vertices;
			     ++vertex_idx) {
				auto const next_hash =
				    workspace_[local_workspace_begin + vertex_idx];

				// extract least significant differing bit
				auto const medge = minimum_differing_pow2(hash, next_hash);

				local_mglbk.maximum[hash_idx] =
				    max(local_mglbk.maximum[hash_idx], medge);
				hash = next_hash;
			}
		}
		return local_mglbk;
	}
};

/** Function coloring_distance2
 * This function will calculate minimum distance 2 color of a graph using hashing algorithm
 * @param row_offsets Row Pointer.
 * @param cols cols.
 * @param warmup Number of warm up cycles.
 */
template <typename IndexType>
int coloring_distance2(thrust::device_vector<IndexType> const& row_offsets,
                       thrust::device_vector<IndexType> const& cols,
                       int warmup) {
	auto min_mglbk{std::numeric_limits<std::uint32_t>::max()};
	int min_hash_k{};

	auto const medge_iter = thrust::make_counting_iterator(IndexType{0});

	thrust::device_vector<IndexType> p(cols.size() + (row_offsets.size() - 1));

	auto const data_ia = row_offsets.data();
	auto const data_ja = cols.data();
	auto const data_p = p.data();

	for (int i = 0; i < (warmup + 1); i++) {
		auto start = Clock::now();

		auto const mglbk = thrust::transform_reduce(
		    medge_iter,
		    medge_iter + row_offsets.size() - 1,
		    MaxColorCollisionsDist2{data_ia, data_ja, data_p},
		    HashWrapper{},
		    Max_mglbk());

		auto const min_it = std::min_element(std::cbegin(mglbk.maximum),
		                                     std::cend(mglbk.maximum));
		if (*min_it < min_mglbk) {
			min_mglbk = *min_it;
			min_hash_k = static_cast<int>(std::distance(
			                 std::cbegin(mglbk.maximum), min_it)) +
			             first_hash_k_param;
		}

		Duration const runtime = Clock::now() - start;
		if (i == warmup) {
			std::cout << "color2_time: " << runtime.count() << '\n';
			std::cout << "color2_colors: " << 2 * min_mglbk << '\n';
			std::cout << "color2_hash: " << min_hash_k << '\n';
		}
	}

	if (warmup == 0) {
		std::cout << "Computing distance-2 coloring with hashing method"
		          << '\n';
		std::cout << "Number of bits (mglb) for valid coloring = " << min_mglbk
		          << '\n';
		std::cout << "Number of colors = " << 2 * min_mglbk << '\n';
		std::cout << "Hash = " << min_hash_k << '\n';
	}

	return 2 * min_mglbk;
}

static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

static const char* _cudaGetErrorEnum(cusparseStatus_t error) {
	return cusparseGetErrorString(error);
}

template <typename T>
void check(T result,
           char const* const func,
           const char* const file,
           int const line) {
	if (result) {
		std::fprintf(stderr,
		             "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
		             file,
		             line,
		             static_cast<unsigned int>(result),
		             _cudaGetErrorEnum(result),
		             func);
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/** Function cusparse_distance1
 * This function will calculate minimum distance 1 color of a graph using CuSPARSE library
 * @param A Non-zero values of graph.
 * @param row_offsets Row Pointer.
 * @param cols cols.
 * @param node_size Number of nodes inside the graph.
 * @param warmup Number of warm up cycles.
 */
template <typename IndexType>
void cusparse_distance1(thrust::device_vector<double> const& A,
                        thrust::device_vector<IndexType> const& row_offsets,
                        thrust::device_vector<IndexType> const& cols,
                        int warmup) {
	cusparseHandle_t handle;
	checkCudaErrors(cusparseCreate(&handle));

	cusparseMatDescr_t descG;
	// creates descriptor for 0-based indexing and general matrix by default
	checkCudaErrors(cusparseCreateMatDescr(&descG));

	cusparseColorInfo_t info;
	checkCudaErrors(cusparseCreateColorInfo(&info));

	thrust::device_vector<IndexType> coloring(row_offsets.size() - 1);

	// fraction of vertices that has to be colored iteratively before falling back
	// to giving every leftover node an unique color
	constexpr double fraction = 1.0;

	int num_colors; // will be updated by cusparse

	for (int i = 0; i < (warmup + 1); i++) {
		checkCudaErrors(cudaDeviceSynchronize());
		auto const start = Clock::now();

		checkCudaErrors(
		    cusparseDcsrcolor(handle,
		                      coloring.size(),
		                      A.size(),
		                      descG,
		                      thrust::raw_pointer_cast(A.data()),
		                      thrust::raw_pointer_cast(row_offsets.data()),
		                      thrust::raw_pointer_cast(cols.data()),
		                      &fraction,
		                      &num_colors,
		                      thrust::raw_pointer_cast(coloring.data()),
		                      nullptr, // don't need reordering
		                      info));

		checkCudaErrors(cudaDeviceSynchronize());
		Duration runtime = Clock::now() - start;

		if (i == warmup) {
			std::cout << "cusparse_time: " << runtime.count() << '\n';
			std::cout << "cusparse_colors: " << num_colors << '\n';
		}
	}
	if (warmup == 0) {
		std::cout << "Computing distance-1 coloring with cuSPARSE" << '\n';
		std::cout << "Number of colors = " << num_colors << '\n';
	}

	checkCudaErrors(cusparseDestroyColorInfo(info));
	checkCudaErrors(cusparseDestroyMatDescr(descG));
	checkCudaErrors(cusparseDestroy(handle));
}

// clean up
#undef checkCudaErrors

} // namespace asc_hash_graph_coloring


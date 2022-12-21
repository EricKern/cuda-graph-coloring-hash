/** 
 * \file main.cpp
 * @brief Main file with simpleTiling example
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <cstdio> //! fprintf
#include <cstdlib> //! atoi

#include <cpumultiply.hpp> //! header file for tiling
#include <tiling.hpp> //! header file for tiling

/**
 * @brief Main entry point for all CPU versions
 * @param[in] m_rows			: number of rows of matrix
 * @param[in] number_of_tiles	: desired number of tiles each layer is partitioned into
 * @param[in] row_ptr			: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr			: integer array of column indices of the nonzero elements of matrix
 * @param[out] slices_dptr		: pointer to integer array of m_rows elements containing row numbers
 * @param[out] ndc_dptr			: pointer to integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] offsets_dptr		: pointer to array with indices for each tile marking line between nodes with and without redundant values
 */
void simple_tiling(const int m_rows,
                   const int number_of_tiles,
                   const int* const row_ptr,
                   const int* const col_ptr,
                   int** const slices_dptr,
                   int** const ndc_dptr,
                   int** const offsets_dptr) {
	int indices_[] = {0, m_rows};
	int number_of_slices = 1;
	int* slices_ = new int[m_rows];
	int* layers_ = new int[m_rows];
	for (int i = 0; i < m_rows; i++) {
		slices_[i] = i;
		layers_[i] = 0;
	}
	*slices_dptr = slices_;
	tiling_partitioning(m_rows,
	                    number_of_tiles,
	                    number_of_slices,
	                    row_ptr,
	                    col_ptr,
	                    layers_,
	                    indices_,
	                    slices_,
	                    ndc_dptr,
	                    slices_dptr,
	                    nullptr,
	                    offsets_dptr,
	                    nullptr,
	                    &number_of_slices,
	                    0);
	delete[] layers_;
}

/**
 * @brief Main function 
 * @param[in] inputMat			: path to the file with matrix information(<name>.mtx)
 * @param[in] number_of_tiles	: number of tiles matrix is partitioned into
 */
int double_multiply(const char* const inputMat, const int number_of_tiles) {
	int* row_ptr;
	int* col_ptr;
	double* val_ptr; // create pointers for matrix in csr format
	const int m_rows =
	    cpumultiplyDloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);
	double* mult_ptr = new double[m_rows];
	double* sol_ptr = new double[m_rows];
	double* prm_ptr = new double[m_rows];
	int* ndc_; // array with indices of each tile in all slices
	int* slices_; // array with nodes grouped in slices
	int* offsets_;
	// initialise reference array
	for (int i = 0; i < m_rows; i++) {
		mult_ptr[i] = static_cast<double>(i) * 0.1;
	}
	simple_tiling(
	    m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_, &offsets_);
	// spMV without permutation
	cpumultiplyDspmv(
	    number_of_tiles, row_ptr, col_ptr, val_ptr, mult_ptr, ndc_, sol_ptr);
	// spMV with previous permutation and successive unpermutation
	cpumultiplyDpermuteMatrix(number_of_tiles,
	                          1,
	                          ndc_,
	                          slices_,
	                          row_ptr,
	                          col_ptr,
	                          val_ptr,
	                          &row_ptr,
	                          &col_ptr,
	                          &val_ptr,
	                          true);
	cpumultiplyDpermuteVector(
	    number_of_tiles, 1, ndc_, slices_, mult_ptr, &mult_ptr, true);
	cpumultiplyDspmv(
	    number_of_tiles, row_ptr, col_ptr, val_ptr, mult_ptr, ndc_, prm_ptr);
	cpumultiplyDpermuteVector(
	    number_of_tiles, 1, ndc_, slices_, prm_ptr, &prm_ptr, false);
	// compare of results with and without permutation
	cpumultiplyDsummary(number_of_tiles, ndc_, sol_ptr, prm_ptr);
	delete[] row_ptr;
	delete[] col_ptr;
	delete[] val_ptr;
	delete[] ndc_;
	delete[] slices_;
	delete[] offsets_;
	delete[] mult_ptr;
	delete[] sol_ptr;
	delete[] prm_ptr;
	return EXIT_SUCCESS;
}

/**
 * @brief Main function 
 * @param[in] inputMat			: path to the file with matrix information(<name>.mtx)
 * @param[in] number_of_tiles	: number of tiles matrix is partitioned into
 */
int float_multiply(const char* const inputMat, const int number_of_tiles) {
	int* row_ptr;
	int* col_ptr;
	float* val_ptr; // create pointers for matrix in csr format
	const int m_rows =
	    cpumultiplySloadMTX(inputMat, &row_ptr, &col_ptr, &val_ptr);
	float* mult_ptr = new float[m_rows];
	float* sol_ptr = new float[m_rows];
	float* prm_ptr = new float[m_rows];
	int* ndc_; // array with indices of each tile in all slices
	int* slices_; // array with nodes grouped in slices
	int* offsets_;
	// initialise reference array
	for (int i = 0; i < m_rows; i++) {
		mult_ptr[i] = static_cast<float>(i) * 0.1f;
	}
	simple_tiling(
	    m_rows, number_of_tiles, row_ptr, col_ptr, &slices_, &ndc_, &offsets_);
	// spMV without permutation
	cpumultiplySspmv(
	    number_of_tiles, row_ptr, col_ptr, val_ptr, mult_ptr, ndc_, sol_ptr);
	// spMV with previous permutation and successive unpermutation
	cpumultiplySpermuteMatrix(number_of_tiles,
	                          1,
	                          ndc_,
	                          slices_,
	                          row_ptr,
	                          col_ptr,
	                          val_ptr,
	                          &row_ptr,
	                          &col_ptr,
	                          &val_ptr,
	                          true);
	cpumultiplySpermuteVector(
	    number_of_tiles, 1, ndc_, slices_, mult_ptr, &mult_ptr, true);
	cpumultiplySspmv(
	    number_of_tiles, row_ptr, col_ptr, val_ptr, mult_ptr, ndc_, prm_ptr);
	cpumultiplySpermuteVector(
	    number_of_tiles, 1, ndc_, slices_, prm_ptr, &prm_ptr, false);
	// compare of results with and without permutation
	cpumultiplySsummary(number_of_tiles, ndc_, sol_ptr, prm_ptr);
	delete[] row_ptr;
	delete[] col_ptr;
	delete[] val_ptr;
	delete[] ndc_;
	delete[] slices_;
	delete[] offsets_;
	delete[] mult_ptr;
	delete[] sol_ptr;
	delete[] prm_ptr;
	return EXIT_SUCCESS;
}

/**
 * @brief Main entry point
 * @param[in] argc	: number of input elements
 * @param[in] argv	: array of input strings
 */
int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::fprintf(stderr,
		             "USAGE: %s matrix-path number-of-tiles [double]\n",
		             argv[0]);
		return EXIT_FAILURE;
	}
	char* inputMat = argv[1];
	const int number_of_tiles = std::atoi(argv[2]);
	return (argc == 3) ? float_multiply(inputMat, number_of_tiles)
	                   : double_multiply(inputMat, number_of_tiles);
}

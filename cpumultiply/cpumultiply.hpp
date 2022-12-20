/** 
 * \file cpumultiply.hpp
 * @brief Header file of shared library
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

int // returns number of rows in generated or loaded matrix
cpumultiplyDloadMTX(const char * const inputMat, // path to the file with matrix information(<name>.mtx) or cuboid dimensions(XxYxZ) for matrix generation
             int ** const row_dptr, // pointer to matrix row array (can be set to NULL, if no output required)
             int ** const col_dptr, // pointer to matrix column array (can be set to NULL, if no output required)
             double ** const val_dptr); // pointer to matrix value array (can be set to NULL, if no output required)

int  // returns number of rows in generated or loaded matrix
cpumultiplySloadMTX(const char * const inputMat, // path to the file with matrix information(<name>.mtx) or cuboid dimensions(XxYxZ) for matrix generation
             int ** const row_dptr, // pointer to matrix row array (can be set to NULL, if no output required)
             int ** const col_dptr, // pointer to matrix column array (can be set to NULL, if no output required)
             float ** const val_dptr); // pointer to matrix value array (can be set to NULL, if no output required)

double // returns time measurement for permutation of vector using first touch approach
cpumultiplyDpermuteVector(const int number_of_tiles, // number of tiles each layer is partitioned into
                         const int n_z, // number of layers
                         const int * const ndc_, // array of number_of_tiles*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
                         const int * const slices_, // array of m_rows(ndc_[number_of_tiles*n_z]) elements containing row numbers used for permutation (index output, slices[index] input)
                         const double * const orig_arr, // pointer to array of m_rows elements on which permutation should be applied
                         double ** const out_arr_dptr, // pointer to resulting array of m_rows elements after permutation
                         const bool prmt); // flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)

double // returns time measurement for permutation of matrix using first touch approach
cpumultiplyDpermuteMatrix(const int number_of_tiles, // number of tiles each layer is partitioned into
                         const int n_z, // number of layers
                         const int * const ndc_, // array of number_of_tiles*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
                         const int * const slices_, // array of m_rows(ndc_[number_of_tiles*n_z]) elements containing row numbers used for permutation (index: new order , value: old order)
                         const int * const row_ptr, // array of m_rows+1 elements that contains the start of every row and the end of the last row plus one on which permutation should be applied
                         const int * const col_ptr, // array of nnz(row_ptr[m_rows]) column indices of the nonzero elements of matrix on which permutation should be applied
                         const double * const val_ptr, // array of nnz values of the nonzero elements of matrix on which permutation should be applied
                         int ** const out_row_dptr, // pointer to resulting array of m_rows+1 elements that contains the start of every row and the end of the last row plus one after permutation
                         int ** const out_col_dptr, // pointer to resulting array of nnz column indices of the nonzero elements of matrix after permutation
                         double ** const out_val_dptr, // pointer to resulting array of nnz values of the nonzero elements of matrix after permutation
                         const bool prmt); // flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)

double // returns time measurement for permutation of vector using first touch approach
cpumultiplySpermuteVector(const int number_of_tiles, // number of tiles each layer is partitioned into
                         const int n_z, // number of layers
                         const int * const ndc_, // array of number_of_tiles*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
                         const int * const slices_, // array of m_rows(ndc_[number_of_tiles*n_z]) elements containing row numbers used for permutation (index output, slices[index] input)
                         const float * const orig_arr, // array of m_rows elements on which permutation should be applied
                         float ** const out_arr_dptr, // pointer to resulting array of m_rows elements after permutation
                         const bool prmt); // flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)

double // returns time measurement for permutation of matrix using first touch approach
cpumultiplySpermuteMatrix(const int number_of_tiles, // number of tiles each layer is partitioned into
                         const int n_z, // number of layers
                         const int * const ndc_, // array of number_of_tiles*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
                         const int * const slices_, // array of m_rows(ndc_[number_of_tiles*n_z]) elements containing row numbers used for permutation (index: new order , value: old order)
                         const int * const row_ptr, // array of m_rows+1 elements that contains the start of every row and the end of the last row plus one on which permutation should be applied
                         const int * const col_ptr, // array of nnz column indices of the nonzero elements of matrix on which permutation should be applied
                         const float * const val_ptr, // array of nnz values of the nonzero elements of matrix on which permutation should be applied
                         int ** const out_row_dptr, // pointer to resulting array of m_rows+1 elements that contains the start of every row and the end of the last row plus one after permutation
                         int ** const out_col_dptr, // pointer to resulting array of nnz column indices of the nonzero elements of matrix after permutation
                         float ** const out_val_dptr, // pointer to resulting array of nnz values of the nonzero elements of matrix after permutation
                         const bool prmt); // flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)

double // returns time measurements for algorithm execution
cpumultiplyDspmv(const int number_of_tiles, // number of tiles each layer is partitioned into
                const int *const row_ptr, // array of ndc_[number_of_tiles]+1 elements that contains the start of every row and the end of the last row plus one
                const int *const col_ptr, // array of nnz column indices of the nonzero elements of matrix
                const double *const val_ptr, // array of nnz values of the nonzero elements of matrix
                const double *const ref_ptr, // reference array
                const int * const ndc_, // array of number_of_tiles+1 elements that contains the start of every tile
                double *const sol_ptr); // calculated solution array (can be set to NULL, if no output required)

double // returns time measurements for algorithm execution
cpumultiplySspmv(const int number_of_tiles, // number of tiles each layer is partitioned into
                const int *const row_ptr, // array of ndc_[number_of_tiles]+1 elements that contains the start of every row and the end of the last row plus one
                const int *const col_ptr, // array of nnz column indices of the nonzero elements of matrix
                const float *const val_ptr, // array of nnz values of the nonzero elements of matrix
                const float *const ref_ptr, // reference array
                const int * const ndc_, // array of number_of_tiles+1 elements that contains the start of every tile
                float *const sol_ptr); // calculated solution array (can be set to NULL, if no output required)

double // returns time measurements for algorithm execution
cpumultiplyDsummary(const int number_of_tiles, // number of tiles each layer is partitioned into
                const int * const ndc_, // array of number_of_tiles+1 elements that contains the start of every tile
                const double *const ref_ptr, // reference array
                const double *const sol_ptr); // calculated solution array (can be set to NULL, if no output required)

double // returns time measurements for algorithm execution
cpumultiplySsummary(const int number_of_tiles, // number of tiles each layer is partitioned into
                const int * const ndc_, // array of number_of_tiles+1 elements that contains the start of every tile
                const float *const ref_ptr, // reference array
                const float *const sol_ptr); // calculated solution array (can be set to NULL, if no output required)

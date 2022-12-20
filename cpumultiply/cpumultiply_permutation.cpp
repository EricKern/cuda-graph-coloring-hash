/** 
 * \file cpumultiply_permutation.cpp
 * @brief File with interfaces for permutation functions
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Sums up row sizes to get row array in csr format
 * @param[in] m_rows			: domain size
 * @param[in,out] perm_row		: array containing row sizes
 */
void row_sums(const int m_rows, int* const perm_row) {
	int row_sz = perm_row[1]; // load size of first row in matrix
	for (int prt_id = 2; prt_id <= m_rows;
	     prt_id++) // go over the following rows
	{
		row_sz += perm_row[prt_id]; // sum up row sizes
		perm_row[prt_id] =
		    row_sz; // write out calculated sum for particular row
	}
}
/**
 * @brief Permute elements of matrix according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (i new index, slices[i] old index)
 * @param[in] in_row_ptr		: integer array of m+1 elements that contains the start of every row and the end of the last row plus one on which permutation should be applied
 * @param[in] in_col_ptr		: array of nnz column indices of the nonzero elements of matrix on which permutation should be applied
 * @param[in] in_val_ptr		: array of nnz values of the nonzero elements of matrix on which permutation should be applied
 * @param[out] out_row_dptr		: pointer to resulting array of m+1 elements that contains the start of every row and the end of the last row plus one after permutation
 * @param[out] out_col_dptr		: pointer to resulting array of nnz column indices of the nonzero elements of matrix after permutation
 * @param[out] out_val_dptr		: pointer to resulting array of nnz values of the nonzero elements of matrix after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of matrix using first touch approach
 */
template <class prcsn>
double cpumultiply_permuteMatrix(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const int* const in_row_ptr,
                                 const int* const in_col_ptr,
                                 const prcsn* const in_val_ptr,
                                 int** const out_row_dptr,
                                 int** const out_col_dptr,
                                 prcsn** const out_val_dptr,
                                 const bool prmt) {
	if (NULL == out_row_dptr && NULL == out_col_dptr && NULL == out_val_dptr)
		return 0.0;
	assert(tn > 0); // number of tiles is at least one
	assert(n_z > 0); // overall number of layers is at least one
	assert(NULL != ndc_); // array with tile indices must be set
	assert(NULL !=
	       slices_); // array with row numbers used for permutation must be set
	assert(NULL != in_row_ptr); // pointer to input row indices must be set
	assert(NULL != in_col_ptr); // pointer to input column indices must be set
	assert(NULL != in_val_ptr); // pointer to input values must be set
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ =
	    Clock::now(); // time points for measurements

	const int tile_amo = tn * n_z; // overall number of tiles
	const int m_rows = ndc_[tile_amo]; // domain size
	const int nnz = in_row_ptr[m_rows]; // number of non zero elements in matrix

	int* const __restrict__ perm_row =
	    new int[m_rows + 1]; // array with permuted row indices
	int* const __restrict__ perm_col =
	    new int[nnz]; // array with permuted column indices
	prcsn* const __restrict__ perm_val =
	    new prcsn[nnz]; // array with permuted matrix entries

	perm_row[0] = 0;

	if (prmt) {
		int* const __restrict__ deslices_ = new int
		    [m_rows]; // array used for permutation of column entries (index old, value new)
#pragma omp parallel num_threads(tn)
		{
			const int t_id = omp_get_thread_num(); // thread own gang id
			const int end_border =
			    (t_id + 1) *
			    n_z; // end of area for processing of current threads
			for (
			    int n_idx = end_border - n_z; n_idx < end_border;
			    n_idx++) // go over rows in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over the entries of a tile in one layer
				{
					const int node_ = slices_[node_idx]; // read in node number
					deslices_[node_] =
					    node_idx; // create permutation array (old->new)
					perm_row[node_idx + 1] =
					    in_row_ptr[node_ + 1] -
					    in_row_ptr
					        [node_]; // calculate number of non-zero entries in a row and write it to heap memory
				}
			}
#pragma omp barrier // 1. ESP (explicit synchronization point)
#pragma omp single
			row_sums(m_rows,
			         perm_row); // 1. ISP (implicit synchronization point)
			for (
			    int n_idx = end_border - n_z, wrt_idx; n_idx < end_border;
			    n_idx++) // go over tiles in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in the array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over rows in tile
				{
					const int node_ = slices_[node_idx]; // read in node number
					wrt_idx =
					    perm_row[node_idx]; // read in permuted write offset
					const int row_end =
					    in_row_ptr[node_ +
					               1]; // index to the beginning of the next row
					for (int row = in_row_ptr[node_]; row < row_end;
					     row++, wrt_idx++) // go over row entries
					{
						perm_col[wrt_idx] = deslices_
						    [in_col_ptr
						         [row]]; // permute column entry and write out
						perm_val[wrt_idx] =
						    in_val_ptr[row]; // copy corresponding value entry
					}
				}
			}
		} // end parallel
		delete[] deslices_;
	}
	else {
#pragma omp parallel num_threads(tn)
		{
			const int t_id = omp_get_thread_num(); // thread own gang id
			const int end_border =
			    (t_id + 1) *
			    n_z; // end of area for processing of current threads
			for (
			    int n_idx = end_border - n_z; n_idx < end_border;
			    n_idx++) // go over rows in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over the entries of a tile in one layer
					perm_row[slices_[node_idx] + 1] =
					    in_row_ptr[node_idx + 1] -
					    in_row_ptr
					        [node_idx]; // calculate number of non-zero entries in a row and write it to heap memory
			}
#pragma omp barrier // 1. ESP (explicit synchronization point)
#pragma omp single
			row_sums(m_rows,
			         perm_row); // 1. ISP (implicit synchronization point)
			for (
			    int n_idx = end_border - n_z, wrt_idx; n_idx < end_border;
			    n_idx++) // go over tiles in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in the array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over rows in tile
				{
					wrt_idx = perm_row
					    [slices_[node_idx]]; // read in permuted write offset
					const int row_end =
					    in_row_ptr[node_idx +
					               1]; // index to the beginning of the next row
					for (int row = in_row_ptr[node_idx]; row < row_end;
					     row++, wrt_idx++) // go over row entries
					{
						perm_col[wrt_idx] = slices_
						    [in_col_ptr
						         [row]]; // permute column entry and write out
						perm_val[wrt_idx] =
						    in_val_ptr[row]; // copy corresponding value entry
					}
				}
			}
		} // end parallel
	}
	if (NULL != out_row_dptr) {
		if (in_row_ptr ==
		    *out_row_dptr) // if input and output row pointers points to the same memory location replacement of input with output desired
			delete[] * out_row_dptr; // delete input row data
		*out_row_dptr =
		    perm_row; // redirect output row pointer to output memory location with permuted data
	}
	else
		delete[] perm_row;
	if (NULL != out_col_dptr) {
		if (in_col_ptr ==
		    *out_col_dptr) // if input and output col pointers points to the same memory location replacement of input with output desired
			delete[] * out_col_dptr; // delete input col data
		*out_col_dptr =
		    perm_col; // redirect output col pointer to output memory location with permuted data
	}
	else
		delete[] perm_col;
	if (NULL != out_val_dptr) {
		if (in_val_ptr ==
		    *out_val_dptr) // if input and output val pointers points to the same memory location replacement of input with output desired
			delete[] * out_val_dptr; // delete input val data
		*out_val_dptr =
		    perm_val; // redirect output val pointer to output memory location with permuted data
	}
	else
		delete[] perm_val;

	return DSeconds(Clock::now() - start_)
	    .count(); // return elapsed time spend for matrix permutation
}

/**
 * @brief Permute elements of matrix according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (i new index, slices[i] old index)
 * @param[in] in_row_ptr		: integer array of m+1 elements that contains the start of every row and the end of the last row plus one on which permutation should be applied
 * @param[in] in_col_ptr		: array of nnz column indices of the nonzero elements of matrix on which permutation should be applied
 * @param[in] in_val_ptr		: array of nnz values of the nonzero elements of matrix on which permutation should be applied
 * @param[out] out_row_dptr		: pointer to resulting array of m+1 elements that contains the start of every row and the end of the last row plus one after permutation
 * @param[out] out_col_dptr		: pointer to resulting array of nnz column indices of the nonzero elements of matrix after permutation
 * @param[out] out_val_dptr		: pointer to resulting array of nnz values of the nonzero elements of matrix after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of matrix using first touch approach
 */
double cpumultiplyDpermuteMatrix(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const int* const in_row_ptr,
                                 const int* const in_col_ptr,
                                 const double* const in_val_ptr,
                                 int** const out_row_dptr,
                                 int** const out_col_dptr,
                                 double** const out_val_dptr,
                                 const bool prmt) {
	return cpumultiply_permuteMatrix<double>(tn,
	                                         n_z,
	                                         ndc_,
	                                         slices_,
	                                         in_row_ptr,
	                                         in_col_ptr,
	                                         in_val_ptr,
	                                         out_row_dptr,
	                                         out_col_dptr,
	                                         out_val_dptr,
	                                         prmt);
}

/**
 * @brief Permute elements of matrix according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (i new index, slices[i] old index)
 * @param[in] in_row_ptr		: integer array of m+1 elements that contains the start of every row and the end of the last row plus one on which permutation should be applied
 * @param[in] in_col_ptr		: array of nnz column indices of the nonzero elements of matrix on which permutation should be applied
 * @param[in] in_val_ptr		: array of nnz values of the nonzero elements of matrix on which permutation should be applied
 * @param[out] out_row_dptr		: pointer to resulting array of m+1 elements that contains the start of every row and the end of the last row plus one after permutation
 * @param[out] out_col_dptr		: pointer to resulting array of nnz column indices of the nonzero elements of matrix after permutation
 * @param[out] out_val_dptr		: pointer to resulting array of nnz values of the nonzero elements of matrix after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of matrix using first touch approach
 */
double cpumultiplySpermuteMatrix(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const int* const in_row_ptr,
                                 const int* const in_col_ptr,
                                 const float* const in_val_ptr,
                                 int** const out_row_dptr,
                                 int** const out_col_dptr,
                                 float** const out_val_dptr,
                                 const bool prmt) {
	return cpumultiply_permuteMatrix<float>(tn,
	                                        n_z,
	                                        ndc_,
	                                        slices_,
	                                        in_row_ptr,
	                                        in_col_ptr,
	                                        in_val_ptr,
	                                        out_row_dptr,
	                                        out_col_dptr,
	                                        out_val_dptr,
	                                        prmt);
}

/**
 * @brief Permute elements of vector according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (index output, slices[index] input)
 * @param[in] orig_arr			: array of m elements on which permutation should be applied
 * @param[out] out_arr_dptr		: pointer to resulting array of m elements after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of array using first touch approach
 */
template <class prcsn>
double cpumultiply_permuteVector(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const prcsn* const orig_arr,
                                 prcsn** const out_arr_dptr,
                                 const bool prmt) {
	if (NULL == out_arr_dptr)
		return 0.0;
	assert(tn > 0); // number of tiles is at least one
	assert(n_z > 0); // number of layers is at least one
	assert(NULL != ndc_); // array with tile indices must be set
	assert(NULL != slices_); // array with row numbers must be set
	assert(NULL != orig_arr); // input must be set
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ =
	    Clock::now(); // time points for measurements
	prcsn* const __restrict__ perm_arr =
	    new prcsn[ndc_[tn * n_z]]; // allocate space for output
	if (prmt) {
#pragma omp parallel num_threads(tn)
		{
			const int end_border =
			    (omp_get_thread_num() + 1) *
			    n_z; // end of area for processing of current threads
			for (
			    int n_idx = end_border - n_z; n_idx < end_border;
			    n_idx++) // go over rows in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in the array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over the entries of a tile in one layer
					perm_arr[node_idx] = orig_arr
					    [slices_[node_idx]]; // permutation of reference array
			}
		}
	}
	else {
#pragma omp parallel num_threads(tn)
		{
			const int end_border =
			    (omp_get_thread_num() + 1) *
			    n_z; // end of area for processing of current threads
			for (
			    int n_idx = end_border - n_z; n_idx < end_border;
			    n_idx++) // go over rows in bar (tiles with same tile index in every layer)
			{
				const int node_end =
				    ndc_[n_idx +
				         1]; // index to the end of a tile in the array slices
				for (int node_idx = ndc_[n_idx]; node_idx < node_end;
				     node_idx++) // go over the entries of a tile in one layer
					perm_arr[slices_[node_idx]] =
					    orig_arr[node_idx]; // permutation of reference array
			}
		}
	}
	if (orig_arr ==
	    *out_arr_dptr) // if input and output points to the same memory location replacement of input with output desired
		delete[] * out_arr_dptr; // delete input data
	*out_arr_dptr =
	    perm_arr; // redirect output pointer to output memory location
	return DSeconds(Clock::now() - start_)
	    .count(); // return elapsed time spend for vector permutation
}

/**
 * @brief Permute elements of vector according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (index output, slices[index] input)
 * @param[in] orig_arr			: array of m elements on which permutation should be applied
 * @param[out] out_arr_dptr		: pointer to resulting array of m elements after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of array using first touch approach
 */
double cpumultiplyDpermuteVector(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const double* const orig_arr,
                                 double** const out_arr_dptr,
                                 const bool prmt) {
	return cpumultiply_permuteVector<double>(
	    tn, n_z, ndc_, slices_, orig_arr, out_arr_dptr, prmt);
}

/**
 * @brief Permute elements of vector according to rearranging information stored in array slices_ using first touch approach
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: oaverall number of layers
 * @param[in] ndc_				: array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile plus one
 * @param[in] slices_			: array of m elements containing row numbers used for permutation (index output, slices[index] input)
 * @param[in] orig_arr			: array of m elements on which permutation should be applied
 * @param[out] out_arr_dptr		: pointer to resulting array of m elements after permutation
 * @param[in] prmt				: flag signals permutation or unpermutation execution (false : unpermutation , true : permutation)
 * @return						: time measurement for permutation of array using first touch approach
 */
double cpumultiplySpermuteVector(const int tn,
                                 const int n_z,
                                 const int* const ndc_,
                                 const int* const slices_,
                                 const float* const orig_arr,
                                 float** const out_arr_dptr,
                                 const bool prmt) {
	return cpumultiply_permuteVector<float>(
	    tn, n_z, ndc_, slices_, orig_arr, out_arr_dptr, prmt);
}

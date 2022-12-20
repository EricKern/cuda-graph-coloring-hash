/** 
 * \file cpumultiply_spmv.cpp
 * @brief File with interfaces for multiplication function
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Performs preconditioned CG algorithm on permuted matrix according to slicing information
 * @param[in] tn		: number of tiles in each layer
 * @param[in] row_ptr	: matrix row pointer
 * @param[in] col_ptr	: matrix column pointer
 * @param[in] val_ptr	: matrix value pointer
 * @param[in] ref_ptr	: reference array
 * @param[in] ndc_		: array with indices of each tile in all slices
 * @param[out] sol_ptr	: calculated solution array (can be set to NULL, if no output required)
 * @return				: time measurements for SpMV execution
 */
template <class prcsn>
double cpumultiply_spmv(const int tn,
                        const int* const row_ptr,
                        const int* const col_ptr,
                        const prcsn* const val_ptr,
                        const prcsn* const ref_ptr,
                        const int* const ndc_,
                        prcsn* const sol_ptr) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ =
	    Clock::now(); // time point for measurements
#pragma omp parallel num_threads(tn)
	{
		float w_thread;
		const int t_id = omp_get_thread_num();
		const int node_end = ndc_[t_id + 1];
		for (int node_ = ndc_[t_id]; node_ < node_end;
		     node_++) // go over the entries of a tile in one layer
		{
			w_thread = 0.0;
			const int row_end = row_ptr
			    [node_ +
			     1]; // index to the beginning of the next row in the matrix
			for (int row = row_ptr[node_]; row < row_end; row++) // go over row
				w_thread +=
				    val_ptr[row] *
				    ref_ptr
				        [col_ptr
				             [row]]; // multiply the matrix value with the precalculated vector value
			sol_ptr[node_] =
			    w_thread; // write SpMV results intermediate value out into heap
		}
	}
	return DSeconds(Clock::now() - start_).count(); // return duration
}

/**
 * @brief Performs preconditioned CG algorithm on permuted matrix according to slicing information
 * @param[in] tn		: number of tiles in each layer
 * @param[in] row_ptr	: matrix row pointer
 * @param[in] col_ptr	: matrix column pointer
 * @param[in] val_ptr	: matrix value pointer
 * @param[in] ref_ptr	: reference array
 * @param[in] ndc_		: array with indices of each tile in all slices
 * @param[out] sol_ptr	: calculated solution array (can be set to NULL, if no output required)
 * @return				: time measurements for SpMV execution
 */
double cpumultiplyDspmv(const int tn,
                        const int* const row_ptr,
                        const int* const col_ptr,
                        const double* const val_ptr,
                        const double* const ref_ptr,
                        const int* const ndc_,
                        double* const sol_ptr) {
	return cpumultiply_spmv<double>(
	    tn, row_ptr, col_ptr, val_ptr, ref_ptr, ndc_, sol_ptr);
}

/**
 * @brief Performs preconditioned CG algorithm on permuted matrix according to slicing information
 * @param[in] tn		: number of tiles in each layer
 * @param[in] row_ptr	: matrix row pointer
 * @param[in] col_ptr	: matrix column pointer
 * @param[in] val_ptr	: matrix value pointer
 * @param[in] ref_ptr	: reference array
 * @param[in] ndc_		: array with indices of each tile in all slices
 * @param[out] sol_ptr	: calculated solution array (can be set to NULL, if no output required)
 * @return				: time measurements for SpMV execution
 */
double cpumultiplySspmv(const int tn,
                        const int* const row_ptr,
                        const int* const col_ptr,
                        const float* const val_ptr,
                        const float* const ref_ptr,
                        const int* const ndc_,
                        float* const sol_ptr) {
	return cpumultiply_spmv<float>(
	    tn, row_ptr, col_ptr, val_ptr, ref_ptr, ndc_, sol_ptr);
}

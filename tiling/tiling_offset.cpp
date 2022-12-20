/** 
 * \file offset.cpp
 * @brief Main file with implementation of ordering within each tile in domain
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Sort of nodes within each tile in a way that those nodes causing redundant calculations are grouped in front
 * @param[in] tn			: number of started threads
 * @param[in] n_z			: number of slices
 * @param[in] ndc_max		: size of biggest tile
 * @param[in] row_ptr		: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr		: integer array of column indices of the nonzero elements of matrix
 * @param[in] ndc_			: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_		: array of vectors containing node numbers
 * @param[in] tiles_		: array with corresponding tile number for each node
 * @param[out] write_ptr	: array containing node numbers grouped in slices and subdivided within each slice into tiles
 * @return offsets_			: array with indices for each tile marking line between nodes with and without redundant values
 */
int* offset_multi(const int tn,
                  const int n_z,
                  const int ndc_max,
                  const int* const row_ptr,
                  const int* const col_ptr,
                  const int* const ndc_,
                  const int* const slices_,
                  const int* const tiles_,
                  int* const write_ptr) {
	int* const __restrict__ offsets_ = new int[tn * n_z];
#pragma omp parallel num_threads(tn)
	{
		int* const __restrict__ cpy_arr =
		    new int[ndc_max]; // each thread allocates memory space for copying
		const int t_id = omp_get_thread_num(); // thread own gang id
		const int t_end =
		    n_z * (t_id + 1); // end index for reading from ndc array

		for (int t_idx = n_z * t_id, current_tile, next_tile = ndc_[t_idx];
		     t_idx < t_end;
		     t_idx++) // go over the layers
		{
			current_tile = next_tile;
			next_tile = ndc_[t_idx + 1];
			int cpy_begin = -1,
			    cpy_end =
			        next_tile -
			        current_tile; // indices for writing into copy array (redundant in front)
			for (int ndc_idx = current_tile; ndc_idx < next_tile;
			     ndc_idx++) // go over the entries of a tile in one layer
			{
				const int node_ = slices_[ndc_idx]; // read in the node number
				bool covered =
				    true; // assumption that node has no redundant entries
				const int node_tile = tiles_[node_];
				const int row_end =
				    row_ptr[node_ + 1]; // end of a row in matrix
				for (int row = row_ptr[node_]; row < row_end;
				     row++) // go over row entries
				{
					const int nei_ =
					    col_ptr[row]; // read in the node number of the neighbor
					if (node_tile !=
					    tiles_
					        [nei_]) // if the neighbor and the node have different tile numbers
					{
						covered =
						    false; // signal that the node has redundant entry
						cpy_arr[++cpy_begin] =
						    node_; // write node in front part of copy array
						break;
					}
				}
				if (covered) // if no redundant entries were found
					cpy_arr[--cpy_end] =
					    node_; // write node in back part of copy array
			}
			offsets_[t_idx] =
			    cpy_end +
			    current_tile; // write out index marking line between nodes with and without redundant values for current tile
			for (int ndc_idx = current_tile; ndc_idx < next_tile;
			     ndc_idx++) // go over the entries of a tile in one layer
				write_ptr[ndc_idx] = cpy_arr
				    [ndc_idx -
				     current_tile]; // write ordered tile into array slices_
		}
		delete[] cpy_arr; // delete copying array
	} // end #pragma omp parallel
	return offsets_;
}
/**
 * @brief Sort of nodes within each tile in a way that those nodes causing redundant calculations are grouped in front
 * @param[in] n_z			: number of slices
 * @param[in] ndc_			: integer array of n_z+1 elements that contains the start of every tile and the end of the last tile
 * @return offsets_			: array with indices for each tile marking line between nodes with and without redundant values
 */
int* offset_single(const int n_z, const int* const ndc_) {
	int* const __restrict__ offsets_ = new int[n_z];
	memcpy(offsets_, ndc_, sizeof(int) * n_z);
	return offsets_;
}

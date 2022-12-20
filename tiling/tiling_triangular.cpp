/** 
 * \file triangular.cpp
 * @brief Main file with implementation of slicing procedure for lower triangular matrix
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <cmath> // ceil

/**
 * @brief Function to register a node for a layer
 * @param[in] row_idx		: node number
 * @param[in] m_rows		: domain size
 * @param[in] thr			: negative if parallel and positive or zero if sequential
 * @param[in] row_ptr		: matrix row pointer
 * @param[in] col_ptr		: matrix column pointer
 * @param[in,out] bin_ptr	: layer pointer
 * @param[out] mtx_flags	: array of flags to recognize which node was already registered
 */
void node_reg(const int row_idx,
              const int m_rows,
              const int thr,
              const int* const row_ptr,
              const int* const col_ptr,
              int* const bin_ptr,
              bool* const mtx_flags) {
	const int tn =
	    (thr < 0)
	        ? bin_ptr[m_rows]
	        : thr; // if in parallel extract the number from the array else use thr
	bin_ptr
	    [m_rows]++; // increase number of participating threads/nodes in a layer
	bin_ptr[m_rows + tn +
	        1]++; // increase number of participating threads/nodes in a tile
	const int invalid =
	    m_rows + tn; // calculate invalidation flag (m_rows+thread ID)
	const int row_end = row_ptr[row_idx] - 1; // stop criteria
	for (int col_idx = row_ptr[row_idx + 1] - 1; col_idx > row_end;
	     col_idx--) // go over the entries of a row
		bin_ptr[col_ptr[col_idx]] = invalid; // invalidate each entry
	bin_ptr[row_idx] = tn; // mark the row by inserting thread ID
	mtx_flags[row_idx] = false; // invalidate global flag
}

#define INVALID_TILE -1
/**
 * @brief Function to determine if a node can be processed sequentially by a node in a layer and register it if so
 * @see node_reg
 * @param[in] row_idx		: node number
 * @param[in] m_rows		: domain size
 * @param[in] tn			: number of started threads
 * @param[in] row_ptr		: matrix row pointer
 * @param[in] col_ptr		: matrix column pointer
 * @param[in,out] bin_ptr	: layer pointer
 * @param[out] mtx_flags	: array of flags to recognize which node was already registered
 * @return					: false if registration was successful else true
 */
bool node_seq(const int row_idx,
              const int m_rows,
              const int tn,
              const int* const row_ptr,
              const int* const col_ptr,
              int* const bin_ptr,
              bool* const mtx_flags) {
	int thr = INVALID_TILE; // thread id for sequential execution
	int rlv; // read in layer value
	for (int col_idx = row_ptr[row_idx]; col_idx < row_ptr[row_idx + 1];
	     col_idx++) // go over the entries of a row
	{
		rlv =
		    bin_ptr[col_ptr[col_idx]]; // read in corresponding entry in a layer
		if (0 > rlv) // if the entry is not set
			continue; // proceed with next entry
		if (m_rows <= rlv) // if the entry is a invalidation flag
			rlv -= m_rows; // substract domain size to get thread ID
		if (0 > thr) // if thread id is not reserved so far
			thr = rlv; // set thread id for this node
		else if (
		    thr !=
		    rlv) // if thread id is already set and it is unequal to the one founded
			return true; // cancel the search process
	}
	if (INVALID_TILE ==
	    thr) // current node can be executed in parallel to all nodes in all tiles
	{
		thr = 0; // set index to first tile
		rlv = bin_ptr[m_rows + 1]; // set size of first tile as initial value
		int* tile_ptr =
		    &bin_ptr[m_rows +
		             2]; // define additional pointer to go over tile sizes
		for (int tile_idx = 1; tile_idx < tn;
		     tile_idx++, tile_ptr++) // go over tiles
			if (rlv > *tile_ptr) // if tile size is smaller than stored one
			{
				rlv = *tile_ptr; // overwrite stored tile size
				thr = tile_idx; // overwrite corresponding tile index
			}
	}
	node_reg(row_idx,
	         m_rows,
	         thr,
	         row_ptr,
	         col_ptr,
	         bin_ptr,
	         mtx_flags); // register the result of successful search
	return false; // return that the search was successful
}

/**
 * @brief Function to determine if a node can be processed in parallel with other nodes in a layer
 * @param[in] row_idx	: node number
 * @param[in] row_ptr	: matrix row pointer
 * @param[in] col_ptr	: matrix column pointer
 * @param[in] bin_ptr	: layer pointer
 * @return				: true if in parallel else false
 */
bool node_par(const int row_idx,
              const int* const row_ptr,
              const int* const col_ptr,
              int* const bin_ptr) {
	for (int col_idx = row_ptr[row_idx]; col_idx < row_ptr[row_idx + 1];
	     col_idx++) // go over the entries of a row
		if (bin_ptr[col_ptr[col_idx]] >
		    INVALID_TILE) // if one entry is already set
			return false; // parallel execution is not possible
	return true; // if none of entries was set parallel execution is possible
}

/**
 * @brief Builds a vector with slices of vertices
 * @see node_reg
 * @see node_par
 * @see node_seq
 * @param[in] m_rows			: domain size
 * @param[in] row_ptr			: matrix row pointer
 * @param[in] col_ptr			: matrix column pointer
 * @param[out] slices_dptr		: pointer to array with slice indices (can be set to NULL, if no output required)
 * @param[out] ndc_dptr			: pointer to an array with indices of each tile in all slices (can be set to NULL, if no output required)
 * @param[out] n_z_ptr			: pointer to number of layers (can be set to NULL, if no output required)
 * @param[out] tn_ptr			: pointer to number of tiles (can be set to NULL, if no output required)
 * @return						: time measurements for slicing process
 */
double tiling_triangular(const int m_rows,
                         const int* const row_ptr,
                         const int* const col_ptr,
                         int** const slices_dptr,
                         int** const ndc_dptr,
                         int* const n_z_ptr,
                         int* const tn_ptr) {
	if (NULL == slices_dptr && NULL == ndc_dptr && NULL == n_z_ptr &&
	    NULL == tn_ptr)
		return 0.0;
	assert(m_rows > 0); // matrix consist of at least one row
	assert(NULL != row_ptr); // pointer to row indices must be set
	assert(NULL != col_ptr); // pointer to column indices must be set
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ =
	    Clock::now(); // time points for measurements
	int layer = 0; // internal counter for layers
	int row_idx;
	int tn = (NULL == tn_ptr) ? m_rows : *tn_ptr;
	std::vector<int*> bin_arr;
	bool* mtx_flags = new bool
	    [m_rows]; // an array of flags to recognize which node was already registered
	for (row_idx = 0; row_idx < m_rows; row_idx++) // go over the nodes
		mtx_flags[row_idx] = true; // mark all nodes as unregistered
	bin_arr.push_back(new int[m_rows + 1 + tn]);
	for (row_idx = 0; row_idx < m_rows; row_idx++) // go over the layer entries
		bin_arr[layer][row_idx] = INVALID_TILE; // invalidate them
	for (row_idx = m_rows + tn; row_idx >= m_rows;
	     row_idx--) // go over the tile sizes
		bin_arr[layer][row_idx] = 0; // initialize them with zero

	node_reg(m_rows - 1,
	         m_rows,
	         INVALID_TILE,
	         row_ptr,
	         col_ptr,
	         bin_arr[layer],
	         mtx_flags); // register the first node
	for (row_idx = m_rows - 2; bin_arr[layer][m_rows] < tn && row_idx > -1;
	     row_idx--) // go over the nodes
		if (node_par(
		        row_idx,
		        row_ptr,
		        col_ptr,
		        bin_arr
		            [layer])) // check if the node can be processed in parallel with other registered nodes
			node_reg(row_idx,
			         m_rows,
			         INVALID_TILE,
			         row_ptr,
			         col_ptr,
			         bin_arr[layer],
			         mtx_flags); // register the node if so
	tn = bin_arr[0][m_rows];
	row_idx = -1;
	do {
		for (; row_idx > -1; row_idx--) // go over the nodes
			if (mtx_flags[row_idx]) // if the node is unregistered
				if (node_seq(
				        row_idx,
				        m_rows,
				        tn,
				        row_ptr,
				        col_ptr,
				        bin_arr[layer - 1],
				        mtx_flags)) // if node can not be sequetially executed in previously created layer
					if (bin_arr[layer][m_rows] < tn)
						if (node_par(
						        row_idx,
						        row_ptr,
						        col_ptr,
						        bin_arr
						            [layer])) // check if the node can be processed in parallel with other registered nodes
							node_reg(row_idx,
							         m_rows,
							         INVALID_TILE,
							         row_ptr,
							         col_ptr,
							         bin_arr[layer],
							         mtx_flags); // register the node if so
		layer++; // increment number of layers
		row_idx = m_rows - 2;
		for (; row_idx > -1; row_idx--) // go over the nodes
			if (mtx_flags[row_idx]) // if the node is unregistered
				if (node_seq(
				        row_idx,
				        m_rows,
				        tn,
				        row_ptr,
				        col_ptr,
				        bin_arr[layer - 1],
				        mtx_flags)) // if node can not be sequetially executed in previously created layer
				{
					bin_arr.push_back(new int[m_rows + 1 + tn]);
					for (int i = 0; i < m_rows;
					     i++) // go over the layer entries
						bin_arr[layer][i] =
						    INVALID_TILE; // initialize them with minus one
					for (int i = m_rows + tn; i >= m_rows;
					     i--) // go over the layer entries
						bin_arr[layer][i] = 0; // initialize them with zero
					node_reg(row_idx,
					         m_rows,
					         INVALID_TILE,
					         row_ptr,
					         col_ptr,
					         bin_arr[layer],
					         mtx_flags); // register the node
					break;
				}
	} while (-1 < row_idx);
	delete[] mtx_flags; // delete registration array

	if (NULL != slices_dptr &&
	    NULL !=
	        ndc_dptr) // if valid pointers for writing of nodes in slices and for writing of slices indices were provided
	{
		int* const ndc_bar =
		    new int[layer * tn + 1]; // allocate sufficient space for ndc array
		ndc_bar[0] = 0; // first entry in ndc is always zero
		int ndc_idx = 1;
		for (row_idx = 0; row_idx < tn; row_idx++) // go over tiles
		{
			const int t_ile = row_idx + m_rows + 1; // tile index in bin_arr
			for (int l_idx = 0; l_idx < layer;
			     l_idx++, ndc_idx++) // go over layers
				ndc_bar[ndc_idx] =
				    ndc_bar[ndc_idx - 1] +
				    bin_arr[l_idx]
				           [t_ile]; // sum up tile sizes to create tile indices
		}
		if (NULL !=
		    slices_dptr) // if valid pointer for writing of nodes in slices was provided
		{
			int* slices_ =
			    new int[m_rows]; // allocate sufficient space for slices array
#pragma omp parallel num_threads(layer)
			{
				int** const wrt_ptr =
				    new int*[tn]; // an array of pointers to slices_ in order to write tile entries
				const int t_id = omp_get_thread_num(); // thread own gang id
				for (int t_idx = 0; t_idx < tn; t_idx++) // go over tiles
					wrt_ptr[t_idx] =
					    &slices_
					        [ndc_bar
					             [t_id +
					              t_idx *
					                  layer]]; // navigate write pointer to an address within slices where current tile begins
				int* const bin_ptr = bin_arr
				    [t_id]; // each thread initialise pointer to its own layer
				for (int b_idx = 0; b_idx < m_rows; b_idx++) // go over nodes
				{
					const int node =
					    bin_ptr[b_idx]; // read in an entry from a layer
					if (node < tn && -1 < node) // if the entry is set
						*(wrt_ptr[node]++) =
						    b_idx; // write out node number to slices_
				}
			}
			*slices_dptr = slices_; // redirect output pointer to created array
		}
		*ndc_dptr = ndc_bar; // redirect output pointer to created array
	}
	for (row_idx = 0; row_idx < layer; row_idx++) // go over the layers
		delete[] bin_arr[row_idx]; // delete layer container
	if (NULL !=
	    n_z_ptr) // if valid pointer for writing of number of layers was provided
		*n_z_ptr = layer; // overwrite number of slices
	if (NULL != tn_ptr)
		*tn_ptr = tn;
	return DSeconds(Clock::now() - start_).count(); // return duration
}
#undef INVALID_TILE

/** 
 * \file coarsening.cpp
 * @brief Main file with implementation of 'fine' input flag
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Perform merging of subsequent slices if the size of combined tiles for all threads is not bigger than maximal one.
 * @param[in] tn			: number of started threads
 * @param[in] ndc_max		: size of biggest tile
 * @param[in] n_z			: number of slices
 * @param[in] ndc_			: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_		: array of vectors containing node numbers
 * @param[out] ndc_ptr		: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] write_ptr	: array containing node numbers grouped in slices and subdivided within each slice into tiles
 * @param[out] til_sum_dptr	: pointer to array with accumulated bar sizes (can be set to NULL, if no output required)
 * @return					: number of slices after possible merging
 */
int coarsening_multi( const int tn, const int ndc_max, int n_z, const int * const ndc_, const int * const slices_, int * const ndc_ptr, int * const write_ptr, int ** const til_sum_dptr )
{
	const int tile_amo=tn*n_z+1;										// overall number of tiles
	int * const __restrict__ til_sum=new int[tn];						// allocate sufficient space for array til_sum
	bool ndc_coarse=true;												// flag which marks if coarsening should happen
	int * const __restrict__ combined_sz=new int[tn];					// contains for each thread the size of tiles for current and previous slices
	int reduced;
	#pragma omp parallel num_threads(tn)
	{
		const int t_id=omp_get_thread_num();							// thread own gang id
		int * const __restrict__ cpy_arr=new int[ndc_max];				// each thread allocates memory space for coarsed tile
		int write_idx=t_id+1;											// thread own index for writing in global array
		int read_idx=write_idx+tn;										// thread own index for reading in global array
		int prev_sz=ndc_[write_idx]-ndc_[write_idx-1];					// tile size for previous slice
		til_sum[t_id]=prev_sz;											// initialize number of nodes in a bar for each thread to tile size of previous slice
		for(; read_idx<tile_amo; read_idx+=tn) 							// go over the entries in ndc
		{
			const int curr_ndc=ndc_[read_idx];							// read in current index out of global array
			const int curr_sz=curr_ndc-ndc_[read_idx-1];				// tile size for current slice
			til_sum[t_id]+=curr_sz;										// add to number of nodes in a bar for each thread tile size of current slice
			combined_sz[t_id]=curr_sz+prev_sz;							// tile size of coarsed tile is stored
			if(combined_sz[t_id]>ndc_max && ndc_coarse)					// if the size of coarsed tile is bigger than the maximal one
				ndc_coarse=false;										// tile coarsening should not be made
			#pragma omp barrier											// 1. ESP (explicit synchronization point)
			if(!ndc_coarse)												// if tile coarsening should not be made
			{
				prev_sz=curr_sz;										// overwrite the size of previous tile for following iteration
				write_idx+=tn;											// set write index to next slice
				ndc_ptr[write_idx]=curr_ndc;							// overwrite the indices of previous tile with the beginning index<	
			}
			else	// coarsening
			{
				if(0==t_id)												// only first thread
					ndc_coarse=true;									// reset the flag for following iterations
				std::memcpy(cpy_arr, &slices_[ndc_[write_idx-1]], prev_sz*sizeof(int));// copy nodes of previous slice in combined array
				std::memcpy(cpy_arr+prev_sz, &slices_[ndc_[read_idx-1]], curr_sz*sizeof(int));// copy nodes of current slice in combined array
				prev_sz=combined_sz[t_id];								// overwrite the size of previous tile for following iteration with combined size
				int tile_sz=prev_sz;									// initialize the size of tile with combined size
				for(int prt_id=0; prt_id<t_id; prt_id++)				// go over the previously stored tile sizes
					tile_sz+=combined_sz[prt_id];						// calculate how many tiles are in front
				tile_sz+=ndc_[write_idx-t_id-1];						// add the beginning index to determine the write address
				#pragma omp barrier										// 2. ESP (explicit synchronization point)
				ndc_ptr[write_idx]=tile_sz;								// overwrite the indices of previous tile with the beginning index of combined tile
				std::memcpy(&write_ptr[tile_sz-prev_sz], cpy_arr, combined_sz[t_id]*sizeof(int));// write content of combined array back to global ndc array
			}
		}	// end for
		#pragma omp barrier												// 3. ESP (explicit synchronization point)
		if(0==t_id)
			reduced=(read_idx-write_idx)/tn-1;							// calculate number of merged slices
		delete [] cpy_arr;												// free thread own memory space for coarsed tile
		prev_sz=til_sum[t_id];											// load bar size of each thread
		for(int prt_id=0; prt_id<t_id; prt_id++)						// go over the preceeding tiles
			prev_sz+=til_sum[prt_id];									// calculate the starting point of following thread for ndc and slices
		#pragma omp barrier												// 4. ESP (explicit synchronization point)
		til_sum[t_id]=prev_sz;											// initialize number of nodes in a bar for each thread to tile size of previous slice
	}
	delete [] combined_sz;												// free arrays
	*til_sum_dptr=til_sum;												// write out accumulated bar sizes
	if(0<reduced)
		printf("tiling: %d slices merged\n",reduced);					// coarsening confirmation message
	return n_z-reduced;													// return possibly changed number of slices
}

/**
 * @brief Perform merging of subsequent slices if the size of combined layers is not bigger than maximal one.
 * @param[in] ndc_max			: size of biggest layer
 * @param[in] n_z				: number of slices
 * @param[in] ndc_				: integer array of n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] ndc_ptr			: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @return						: number of slices after possible merging
 */
int coarsening_single( const int ndc_max, int n_z, const int * const ndc_, int * const ndc_ptr )
{
	int write_idx = 1;													// index for writing in global array
	int read_idx = write_idx+1;											// index for reading in global array
	for(int prev_sz=ndc_[write_idx]-ndc_[write_idx-1]; read_idx<=n_z; read_idx++)
	{
		const int curr_ndc=ndc_[read_idx];								// read in current index out of global array
		const int curr_sz=curr_ndc-ndc_[read_idx-1];					// tile size for current slice
		const int combined_sz=curr_sz+prev_sz;							// size of combined tiles
		if(combined_sz>ndc_max)											// if the size of combined tiles is bigger than the maximal one
		{
			prev_sz=curr_sz;											// overwrite the size of previous tile for following iteration
			ndc_ptr[++write_idx]=curr_ndc;								// write out current index to the next entry in global array
		}
		else	// coarsening
		{
			prev_sz=combined_sz;										// overwrite the size of previous tile for following iteration with combined size
			ndc_ptr[write_idx]=curr_ndc;								// write out current index to the same entry in global array
		}
	}
	int reduced = read_idx-write_idx-1;									// number of merged slices
	if(0<reduced)
		printf("%d indices merged\n",reduced);							// coarsening confirmation message
	return n_z-reduced;													// return overall number of slices after merging
}

/**
 * @brief Determine number of nodes in each bar and write it out
 * @param[in] tn			: number of started threads
 * @param[in] n_z			: overall number of slices
 * @param[in] ndc_			: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] til_sum_dptr	: pointer to array with bar sizes
 */
void create_til_sum(const int tn, const int n_z, const int * const ndc_, int ** const til_sum_dptr )
{
	int * const til_sum=new int[tn];									// allocate sufficient space for array with bar sizes
	const int tile_amo=tn*n_z;
	#pragma omp parallel num_threads(tn)
	{
		const int t_id=omp_get_thread_num();							// thread own gang id
		til_sum[t_id]=0;												// initialize bar size with zero
		for(int t_idx = t_id; t_idx<tile_amo; t_idx+=tn)				// go over the entries in ndc
			til_sum[t_id]+=ndc_[t_idx+1]-ndc_[t_idx];					// increment bar size by size of current tile
	}
	for(int idx=1; idx<tn; idx++ )										// go over bars
		til_sum[idx]+=til_sum[idx-1];									// initialize bar size with zero
	*til_sum_dptr=til_sum;												// write out accumulated bar sizes
}

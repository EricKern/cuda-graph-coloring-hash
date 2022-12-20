/** 
 * \file accessories.cpp
 * @brief Main file with implementation of metising and sectioning helping functions
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Print information about decision process during tiling
 * @param[in] tn			: the number of started threads
 * @param[in] stage			: stage number
 * @param[in] prt_     		: array with numbers of registered nodes for all tiles
 * @param[out] reports_		: array for printing of additional information
 * @return					: either NULL or pointer to the begin of free space in array reports_
 */
char * stage_handout( const int tn, const int stage, const int * const prt_, char * reports_)
{
	if(NULL==reports_)													// if provided pointer is NULL output on stdout is desired
	{
		printf("      stage %d: %d %d", stage, prt_[0], prt_[1]);
		for(int idx=2; idx<tn; idx++)
			printf(" %d", prt_[idx]);
		printf("\n");
	}
	else																// if provided pointer points to memory location data are written to it
	{
		reports_+=sprintf(reports_, "      stage %d: %d %d", stage, prt_[0], prt_[1]);
		for(int idx=2; idx<tn; idx++)
			reports_+=sprintf(reports_, " %d", prt_[idx]);
		reports_+=sprintf(reports_, "\n");
	}
	return reports_;													// either NULL or pointer to the begin of free space in array reports_
}

/**
 * @brief Reorganize the order in which the nodes are stored within slices in order to match the tiles subdivision and print statistic data if desired
 * @param[in] s_id				: index of the current slice
 * @param[in] tn				: number of started threads
 * @param[in] slice_start		: marks the start position of a slice (index to the slices array)
 * @param[in] tiles_			: array with corresponding tile number for each node
 * @param[in] indices_			: integer array of n_z+1 elements that contains the start of every layer in slices array and the end of the last layer in slices array
 * @param[in] slices_			: integer array of m elements containing row numbers grouped in layers
 * @param[in] reports_			: array for printing of additional information (possibly NULL)
 * @param[in,out] prt_			: array with number of nodes in the each tile
 * @param[out] stats_dptr		: pointer to array of chars intended to hold printing information (possibly NULL)
 * @param[out] ndc_				: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] write_ptr		: array containing node numbers grouped in layers and subdivided within each layer into tiles (possibly NULL)
 * @param[in] s_info			: directive to print additional information about each tile(layer number and size as well as average, minimal and maximal tile size within this layer)
 */
void form_tiles( const int s_id, const int tn, const int slice_start, const int * const tiles_, const int * const slices_, const char * const reports_, int * const prt_, char ** const stats_dptr, int * const ndc_, int * const write_ptr, const bool s_info )
{
	int * const __restrict__ cp_v=new int[tn+1];						// array to hold the copies of tile indices
	int slice_end=slice_start;											// index of the first entry of the slice within the array slices
	for( int n_idx=0, var_idx=s_id*tn; n_idx<tn; n_idx++)				// go over the tiles
	{
		cp_v[n_idx]=slice_end;											// store the copy of one index into the array
		ndc_[var_idx++]=slice_end;										// save the index in the global array ndc_
		slice_end+=prt_[n_idx];											// read in the number of nodes in the next tile and increment the index accordingly
	}
	cp_v[tn]=slice_end;													// store last index which marks the beginning of following slice

	if(NULL!=write_ptr)													// if pointer for output writing is NULL no output is desired
	{
		int * const __restrict__ hlp_v=new int[slice_end-slice_start];	// array to hold the copies of node identification numbers
		#pragma omp parallel num_threads(tn)
		{
			const int t_id=omp_get_thread_num();						// each thread register nodes from one tile
			const int * rd_ptr=slices_+slice_start-1;					// set read pointer
			int * wrt_ptr=hlp_v+(cp_v[t_id]-slice_start);				// set write pointer
			const int * const stp_ptr=hlp_v+(cp_v[t_id+1]-slice_start);		// set stop pointer
			while(wrt_ptr<stp_ptr)										// as long as write pointer is in front of stop pointer
			{
				const int node_=*(++rd_ptr);							// read in node number
				if(t_id!=tiles_[node_])									// if node is not from the same tile as thread number
					continue;											// go to the next entry in slices_
				*(wrt_ptr++)=*rd_ptr;									// overwrite the node identification numbers in slices with sorted ones from hlp_v array
			}
			#pragma omp barrier											// 1. ESP (explicit synchronization point)
			rd_ptr=hlp_v+(cp_v[t_id]-slice_start);
			wrt_ptr=write_ptr+cp_v[t_id];
			while(rd_ptr<stp_ptr)
				*(wrt_ptr++)=*(rd_ptr++);
		}	// end parallel
		delete [] hlp_v;
	}	// end write
	delete [] cp_v;

	if(NULL!=reports_||s_info)											// if tinf or sinf flags were set
	{
		if(NULL==*stats_dptr)											// if stdout should be used for printing
		{
			printf( "statistic for slice %d:\n", s_id);
			if(s_info)													// if optional argument in order to print messages about layers is set to true
			{
				const int prt_init=prt_[0];
				int prt_min=prt_init;
				int prt_max=prt_init;
				double prt_sum=prt_init;
				for( int prt_idx=1; prt_idx<tn; prt_idx++)				// go over the tiles
				{
					const int prt_val=prt_[prt_idx];
					if(prt_val>prt_max)									// if tile size is bigger than the maximal one
						prt_max=prt_val;								// overwrite the maximal value
					if(prt_val<prt_min)									// if tile size is bigger than the maximal one
						prt_min=prt_val;								// overwrite the maximal value
					prt_sum+=prt_val;
				}
				printf( "   slicing summary:\n"
						"      size\t%d\n"
						"      avg\t%.2f\n"
						"      min\t%d\n"
						"      max\t%d\n"
						, slice_end-slice_start, prt_sum/tn, prt_min, prt_max);
			}
			if(NULL!=reports_)											// if optional argument in order to print messages about tiles is set to true
				printf("%s", reports_);
		}
		else															// if printing should be done to stats
		{
			*stats_dptr+=sprintf(*stats_dptr, "statistic for slice %d:\n", s_id);
			if(s_info)													// if optional argument in order to print messages about layers is set to true
			{
				const int prt_init=prt_[0];
				int prt_min=prt_init;
				int prt_max=prt_init;
				double prt_sum=prt_init;
				for( int prt_idx=1; prt_idx<tn; prt_idx++)				// go over the tiles
				{
					const int prt_val=prt_[prt_idx];
					if(prt_val>prt_max)									// if tile size is bigger than the maximal one
						prt_max=prt_val;								// overwrite the maximal value
					if(prt_val<prt_min)									// if tile size is bigger than the maximal one
						prt_min=prt_val;								// overwrite the maximal value
					prt_sum+=prt_val;
				}
				*stats_dptr+=sprintf(*stats_dptr, "   slicing summary:\n"
												"      size\t%d\n"
												"      avg\t%.2f\n"
												"      min\t%d\n"
												"      max\t%d\n"
												, slice_end-slice_start, prt_sum/tn, prt_min, prt_max);
			}
			if(NULL!=reports_)											// if optional argument in order to print messages about tiles is set to true
			{
				strcat(*stats_dptr, reports_);
				*stats_dptr=strchr(*stats_dptr,'\0');
			}
		}	// end stats
	}	// end tinf or sinf
}

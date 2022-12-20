/** 
 * \file sectioning.cpp
 * @brief Main file with implementation of sectioning procedure
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <algorithm>		// std::sort

#define SET_INVALID -1
#define SET_MULTIPLE -2
/**
 * @brief Finally subdivide dynamically all but maximal slice into tiles by counting the connections of neighbouring layers
 * @see stage_handout
 * @param[in] node_layer	: index of the current slice
 * @param[in] tn			: number of started threads
 * @param[in] slice_start	: marks the start position of a slice (index to the slices array)
 * @param[in] slice_end		: marks the end position of a slice (index to the slices array)
 * @param[in] ndc_max		: average number of nodes in each tile if divided evenly
 * @param[in] borders		: array of start and end borders for each thread in csr format
 * @param[in] row_ptr		: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr		: integer array of column indices of the nonzero elements of matrix
 * @param[in] layers_		: array with corresponding layer number for each node
 * @param[in,out] slices_	: array of vectors containing node numbers
 * @param[in,out] tiles_	: array with corresponding tile number for each node
 * @param[in,out] prt_		: counts the number of nodes in tiles
 * @param[out] reports_		: array for printing of additional information (possibly NULL)
 * @param[out] part_		: array which stores the index of a tile with maximal connections the node should be registered to
 * @param[out] conn_		: array which stores maximal number of connections to tile specified in part_
 * @param direct_global		: global array which stores the number of connections to the nodes in tiles in the neighbouring slice for each participating thread
 * @param indirect_global	: global array which stores the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer for each participating thread
 */
inline void final_sectioning(const int node_layer, const int tn, const int slice_start, const int slice_end, const int ndc_max, const int * const borders_, const int *const row_ptr, const int *const col_ptr, const int * const layers_, const int * const slices_, int * const __restrict__ tiles_, int * const __restrict__ prt_, char * reports_, int * const __restrict__ part_, int * const __restrict__ conn_, int * const direct_global, int * const indirect_global)
{
	const int slice_sz=slice_end-slice_start;							// size of a slice
	bool first_exit;													// flag to exit the first stage
	#pragma omp parallel num_threads(tn)								// check the first condition (maximal connections for a tile in neighbouring layer) and the second condition (maximal connections to a tile in neighbouring layer from the adjacent nodes in the same layer)
	{
		const int t_id=omp_get_thread_num();							// each thread gets its number in the gang
		const int end_border=borders_[t_id+1];							// end of area for processing of current threads
		for(int n_idx = borders_[omp_get_thread_num()]; n_idx<end_border; n_idx++)	// go over the entries
			conn_[n_idx]=SET_INVALID;									// number of valid connections to nodes in other tiles is set to negative value
		int * const __restrict__ direct_ = &direct_global[tn*t_id];		// counts the number of connections to the nodes in tiles in the neighbouring slice
		int * const __restrict__ indirect_ = &indirect_global[tn*t_id];	// counts the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer
		std::vector< std::pair <int,int> > tile_vec;					// vector containing pairs of number of connections and tile identification
		do{
			#pragma omp single											// 1. ISP (implicit synchronization point)
				first_exit=true;										// exit flag is set to true (no nodes to register)
			for(int n_idx = borders_[t_id]; n_idx<end_border; n_idx++)	// go over the nodes in current layer
			{
				if(SET_INVALID<part_[n_idx])							// all validated nodes
					continue;											// will be disregarded
				std::memset(direct_, 0, sizeof(int)*tn);				// initialize direct_ array with zeroes
				std::memset(indirect_, 0, sizeof(int)*tn);				// initialize indirect_ array with zeroes
				const int node_=slices_[slice_start+n_idx];				// read in node number
				
				const int row_end=row_ptr[node_+1];						// determine the begin of following row
				for(int row=row_ptr[node_]; row<row_end; row++)			// go over non-zero elements in current row
				{
					const int nei_ = col_ptr[row];						// read in the node number of the neighbor
					const int nei_layer=layers_[nei_];					// layer number the neighbour is in
					const int nei_tile=tiles_[nei_];					// tile number the neighbour is in
					if(0>nei_layer||0>nei_tile)							// if a node is not registered for a layer or a tile
						continue;										// go to the next node
					if(ndc_max>prt_[nei_tile])
					{
						const int nei_dist= node_layer-nei_layer;		// determine the distance between current layer and layer the neighbour is in
						if(1==nei_dist||-1==nei_dist)					// if a node is in the neighboring layer
							direct_[nei_tile]++;						// register for a tile its connected to
						else if(0==nei_dist)							// if a node is in the same layer
							indirect_[nei_tile]++;						// register for a tile its connected to
					}
				}
				bool multiple=false;									// flag to show if multiple maximal entries exist
				int part_idx=0;											// initial index to go through vector second
				int max_val=direct_[part_idx];							// set initial value of maximal index for direct connections
				int sup_val=indirect_[part_idx];						// set initial value of maximal index for indirect connections
				for(int m_idx=1; m_idx<tn; m_idx++)						// go through vector third and find first occurance of minimal entry amoung the direct_ values
				{
					const int parity=direct_[m_idx]-max_val;			// difference in connections to different tiles in neighbouring layer
					const int equality=indirect_[m_idx]-sup_val;		// difference in connections to different tiles in neighbouring layer through the nodes in the same layer
					if(0<parity||(0==parity&&0<equality))				// if one tile has less registered nodes than the other or same number of registered nodes but more connecitons through neighbors 
					{
						part_idx=m_idx;									// overwrite the index
						max_val=direct_[m_idx];							// reset value of maximal direct index
						sup_val=indirect_[m_idx];						// reset value of maximal indirect index
						if(multiple)									// if multiple tiles have the same number of connections but a tile with more connections to it was found
						{
							multiple=false;								// reset the multiple flag
							part_[n_idx]=SET_INVALID;					// reset mark
						}
					}
					else if(0==parity&&0==equality)						// if both tiles have the same number of registered nodes and same number of connections through neighbors
					{
						multiple=true;									// set the multiple flag
						if(0!=max_val||0!=sup_val)						// only those nodes should be counted that have actual connections to multiple tiles
							part_[n_idx]=SET_MULTIPLE;					// set mark
					}
				}
				if(!multiple)											// if indeed there is only one maximal entry
				{
					if(first_exit)										// if at least one node is entered for registration
						first_exit=false;								// set the exit flag to false
					part_[n_idx]=part_idx;								// store the index of a tile the node should be registered to
					conn_[n_idx]=direct_[part_idx];						// store the number of connections to the tile
				}
			}
			#pragma omp barrier											// 1. ESP (explicit synchronization point)
			if(first_exit)												// if the exit flag was not set to false
				break;													// there are no nodes to register, exit the outer loop
			#pragma omp barrier											// 2. ESP (explicit synchronization point)
			for(int ind_=0; ind_<slice_sz; ind_++)						// go over the nodes
				if((SET_INVALID!=conn_[ind_])&&(t_id==part_[ind_]))		// if the node should be registered (number of connections is set) and it is right tile number
					tile_vec.push_back(std::make_pair(conn_[ind_],ind_));	// register connections to a specific tile and position of the node within current slice
			int cntr=tile_vec.size();									// determine the number of nodes to register
			const int rest=ndc_max-prt_[t_id];							// determine how many vacant places are there
			std::sort(tile_vec.rbegin(), tile_vec.rend());				// sort the entries according to found connections in descending order
			if(cntr>=rest)												// if there are more or equal nodes to register than vacant places in a tile
			{
				for(int tl_idx=rest; tl_idx<cntr; tl_idx++)				// go over the tiles
				{
					const int index = tile_vec[tl_idx].second;			// determine internal index of the entry to register
					part_[index]=SET_INVALID;							// reset the tile number
					conn_[index]=SET_INVALID;							// reset connection amount
				}
				cntr=rest;												// set the number of nodes to registrate to highest possible
			}
			for(int tl_idx=0; tl_idx<cntr; tl_idx++)					// go over the nodes to registrate
			{
				const int index = tile_vec[tl_idx].second;				// determine internal index of the entry to register
				tiles_[slices_[slice_start+index]]=t_id;				// register the node for a particular tile
				conn_[index]=SET_INVALID;								// reset the flag
			}
			tile_vec.clear();											// erases all elements from the container, but leaves the capacity() of the vector unchanged
			prt_[t_id]+=cntr;											// increment internal counter for a particular tile
		}while(true);
	} // end parallel
	
	if(NULL!=reports_)													// if optional argument in order to print messages about tiles is set to true
	{
		int rest=0, multi=0;
		#pragma omp parallel for reduction(+: rest) reduction(+: multi)
		for (int n_idx = 0; n_idx < slice_sz; n_idx++)
		{
			const int node_state=part_[n_idx];
			if(SET_INVALID==node_state)
				rest += 1;
			else if(SET_MULTIPLE==node_state)
				multi += 1;
		}
		reports_=stage_handout( tn, 1, prt_, strchr(reports_,'\0') );
		reports_+=sprintf(reports_, "         remaining nodes: %d of %d \n"
									"         multiple nodes: %d of %d \n"
									, rest, slice_sz, multi, slice_sz);
	}
	int registered=prt_[0];												// summation result of already registered nodes is initialized with amount of registered nodes in first tile
	for( int n_idx=1; n_idx<tn; n_idx++)								// go over the tiles
		registered+=prt_[n_idx];										// sum up amount of registered nodes over all tiles
	if(!(slice_sz>registered))											// if summation result is smaller than amount of nodes in slice there are still nodes to register
		return;															// otherwise no further sectioning steps are needed
	
	int max_sim=0, random_calls=0, tile_idx;
	int * const __restrict__ indirect_ = new int[tn];					// counts the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer
	for(int n_idx=0; n_idx<slice_sz; n_idx++)							// go over the nodes in a slice
	{
		if(SET_INVALID<part_[n_idx])									// all validated nodes from previous stage
			continue;													// will be disregarded
		const int node_=slices_[slice_start+n_idx];						// node number
		if(SET_INVALID<tiles_[node_])									// all validated nodes from current stage
			continue;													// will be also disregarded
		std::memset(indirect_, 0, sizeof(int)*tn);						// initialize indirect_ array with zeroes
		std::vector<int> neis_;											// vector with neighbours from the same layer for registration
		neis_.push_back(node_);											// the node itself is put on the begin of registration queue
		const int row_end=row_ptr[node_+1];								// end of a row in matrix
		for(int row=row_ptr[node_]; row<row_end; row++)					// go over the elements of a row
		{
			const int nei_ = col_ptr[row];								// read in the node number of the neighbor
			if(nei_==node_)												// if it is diagonal element
				continue;												// proceed with next neighbour
			if(node_layer==layers_[nei_])								// if neighbour is in the same layer as the node
			{
				if(0>tiles_[nei_])										// if a neighbour is not registered for a tile
					neis_.push_back(nei_);								// record its number
				else if(ndc_max>prt_[tiles_[nei_]])						// if a neighbor belongs to a nonfull tile
					indirect_[tiles_[nei_]]++;							// register for a tile its connected to
			}
		}
		int indir_min=indirect_[0];										// the number of indirect connections to tile zero is taken as initial value for searching of minimal entry
		int indir_max=indir_min;										// the same value is also set for searching of maximal entry
		tile_idx=0;														// corresponding tile index is also set to zero
		for( int indir_idx=1; indir_idx<tn; indir_idx++)				// go over entries in array
		{
			const int indir_val=indirect_[indir_idx];					// read in another tile size
			if(indir_val>indir_max)										// if tile size is bigger than the maximal one
			{
				indir_max=indir_val;									// overwrite the maximal value
				tile_idx=indir_idx;										// set corresponding index to tile number
			}
			if(indir_val<indir_min)										// if tile size is smaller than the minimal one
				indir_min=indir_val;									// overwrite the minimal value
		}
		if(indir_min==indir_max)										// if all entries of array have same values
		{
			random_calls++;												// random tile selection take place
			int min_val=prt_[0];										// amount of nodes in tile zero is taken as initial value for searching of tile with minimal number of registered nodes
			for(int prt_idx=1; prt_idx<tn; prt_idx++)					// go over the tiles
			{
				const int cur_val=prt_[prt_idx];						// get the size of a tile
				if(cur_val<min_val)										// if this tile is smaller than currently stored
				{
					min_val=cur_val;									// set new minimal size
					tile_idx=prt_idx;									// rewrite tile number appropriately
				}
			}
		}

		int cntr=neis_.size();											// determine the number of nodes to register
		const int rest=ndc_max-prt_[tile_idx];							// how many nodes can be added to this tile
		if(cntr>=rest)													// if there are more nodes to register than vacant places in tile
			cntr=rest;													// number of nodes to register now equals vacant places in tile
		for(int tl_idx=cntr-1; tl_idx>-1; tl_idx--)						// go over the neighbours
			tiles_[neis_[tl_idx]]=tile_idx;								// register the node for a particular tile
		prt_[tile_idx]+=cntr;											// increase the number of nodes registered for particular tile by vector size

		if(max_sim<cntr)												// if number of nodes added simultaneously is bigger than previous one
			max_sim=cntr;												// overwrite the maximal value
	}
	delete [] indirect_;
	
	if(NULL!=reports_)													// if optional argument in order to print messages about tiles is set to true
	{
		reports_=stage_handout( tn, 2, prt_, reports_ );
		reports_+=sprintf(reports_, "         calls for random assignment: %d\n"
									"         nodes added at ones: %d\n"
									, random_calls, max_sim);
	}
}

/**
 * Presubdivide dynamically all but maximal slice into tiles by counting the connections of neighbouring layers
 * @see stage_handout
 * @param[in] node_layer	: index of the current slice
 * @param[in] tn			: number of started threads
 * @param[in] slice_start	: marks the start position of a slice (index to the slices array)
 * @param[in] slice_end		: marks the end position of a slice (index to the slices array)
 * @param[in] ndc_max		: average number of nodes in each tile if divided evenly
 * @param[in] cond_ 		: 1 if allready partitioned slice is on the left side of the current slice else it is -1
 * @param[in] borders		: array of start and end borders for each thread in csr format
 * @param[in] row_ptr		: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr		: integer array of column indices of the nonzero elements of matrix
 * @param[in] layers_		: array with corresponding layer number for each node
 * @param[in] slices_		: array of vectors containing node numbers
 * @param[in,out] tiles_	: array with corresponding tile number for each node
 * @param[out] prt_			: contains amount of nodes already assigned to each tile/thread
 * @param[out] reports_		: array for printing of additional information (possibly NULL)
 * @param[out] part_		: array which stores the index of a tile with maximal connections the node should be registered to
 * @param[out] conn_		: array which stores maximal number of connections to tile specified in part_
 * @param direct_global		: global array which stores the number of connections to the nodes in tiles in the neighbouring slice for each participating thread
 * @param indirect_global	: global array which stores the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer for each participating thread
 */
inline bool init_sectioning(const int node_layer, const int tn, const int slice_start, const int slice_end, const int ndc_max, const int cond_, const int * const borders_, const int * const row_ptr, const int * const col_ptr, const int * const layers_, const int * const slices_, int * const tiles_, int * const prt_, char * reports_, int * const part_, int * const conn_, int * const direct_global, int * const indirect_global)
{
	const int slice_sz=slice_end-slice_start;							// size of a slice
	bool first_exit;													// flag to exit the first stage
	#pragma omp parallel num_threads(tn)								// check the first condition (maximal connections for a tile in neighbouring layer)  and the second condition
	{ 																	// (maximal connections to a tile in neighbouring layer from the adjacent nodes in the same layer)
		const int t_id=omp_get_thread_num();							// each thread gets its number in the gang
		const int end_border=borders_[t_id+1];							// end of area for processing of current threads
		prt_[t_id]=0;													// initialize all tile sizes with zero
		for(int n_idx = borders_[t_id]; n_idx<end_border; n_idx++)		// go over the entries
		{
			part_[n_idx]=SET_INVALID;									// set partition number to invalid value
			conn_[n_idx]=SET_INVALID;									// set number of connetions to invalid value
		}
		int * const __restrict__ direct_ = &direct_global[tn*t_id];		// counts the number of connections to the nodes in tiles in the neighbouring slice
		int * const __restrict__ indirect_ = &indirect_global[tn*t_id];	// counts the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer
		std::vector< std::pair <int,int> > tile_vec;					// vector containing pairs of number of connections and tile identification
		do{
			#pragma omp single											// 1. ISP (implicit synchronization point)
				first_exit=true;										// exit flag is set to true (no nodes to register)
			for(int n_idx = borders_[t_id]; n_idx<end_border; n_idx++)	// go over the entries
			{
				if(SET_INVALID<part_[n_idx])							// all validated nodes
					continue;											// will be disregarded
				std::memset(direct_, 0, sizeof(int)*tn);				// initialize direct_ array with zeroes
				std::memset(indirect_, 0, sizeof(int)*tn);				// initialize indirect_ array with zeroes
				const int node_=slices_[slice_start+n_idx];				// read in node number
				
				const int row_end=row_ptr[node_+1];						// index to the beginning of the next row in the matrix
				for(int row=row_ptr[node_]; row<row_end; row++)			// go over the row
				{
					const int nei_ = col_ptr[row];						// read in the node number of the neighbor
					const int nei_layer=layers_[nei_];					// layer number the neighbour is in
					const int nei_tile=tiles_[nei_];					// tile number the neighbour is in
					if(0>nei_layer||0>nei_tile)							// if a node is not registered for a layer or a tile
						continue;										// go to the next node
					if(ndc_max>prt_[nei_tile])
					{
						const int nei_dist= node_layer-nei_layer;		// determine the distance between current layer and layer the neighbour is in
						if(cond_==nei_dist)								// if a node is in the neighboring layer
							direct_[nei_tile]++;						// register for a tile its connected to
						else if(0==nei_dist)							// if a node is in the same layer
							indirect_[nei_tile]++;						// register for a tile its connected to
					}
				}
				bool multiple=false;									// flag to show if multiple maximal entries exist
				int part_idx=0;											// initial index to go through vector second
				int max_val=direct_[part_idx];							// set initial value of maximal index for direct connections
				int sup_val=indirect_[part_idx];						// set initial value of maximal index for indirect connections
				for(int m_idx=1; m_idx<tn; m_idx++)						// go through vector third and find first occurance of minimal entry amoung the direct_ values
				{
					const int parity=direct_[m_idx]-max_val;			// difference in connections to different tiles in neighbouring layer
					const int equality=indirect_[m_idx]-sup_val;		// difference in connections to different tiles in neighbouring layer through the nodes in the same layer
					if(parity>0||(0==parity&&equality>0))				// if one tile has less registered nodes than the other or same number of registered nodes but more connecitons through neighbors 
					{
						part_idx=m_idx;									// overwrite the index
						max_val=direct_[m_idx];							// reset value of maximal direct index
						sup_val=indirect_[m_idx];						// reset value of maximal indirect index
						if(multiple)									// if multiple tiles have the same number of connections but a tile with more connections to it was found
						{
							multiple=false;								// reset the multiple flag
							part_[n_idx]=SET_INVALID;					// reset mark
						}
					}
					else if(0==parity&&0==equality)						// if both tiles have the same number of registered nodes and same number of connections through neighbors
					{
						multiple=true;									// set the multiple flag
						if(0!=max_val||0!=sup_val)						// only those nodes should be counted that have actual connections to multiple tiles
							part_[n_idx]=SET_MULTIPLE;					// set mark
					}
				}
				if(!multiple)											// if indeed there is only one maximal entry
				{
					if(first_exit)										// if at least one node is entered for registration
						first_exit=false;								// set the exit flag to false
					part_[n_idx]=part_idx;								// store the index of a tile the node should be registered to
					conn_[n_idx]=direct_[part_idx];						// store the number of connections to the tile
				} // end inner for
			} // end outer for
			#pragma omp barrier											// 1. ESP (explicit synchronization point)
			if(first_exit)												// if the exit flag was not set to false
				break;													// there are no nodes to register, exit the outer loop
			#pragma omp barrier											// 2. ESP (explicit synchronization point)
			for(int ind_=0; ind_<slice_sz; ind_++)						// go over the nodes
				if((SET_INVALID!=conn_[ind_])&&(t_id==part_[ind_]))		// if the node should be registered (number of connections is set) and it is right tile number
					tile_vec.push_back(std::make_pair(conn_[ind_],ind_));
			int cntr=tile_vec.size();									// determine the number of nodes to register
			const int rest=ndc_max-prt_[t_id];							// how many nodes can be added to this tile
			std::sort(tile_vec.rbegin(), tile_vec.rend());				// sort the entries by connections
			if(cntr>=rest)												// if there are more nodes to register than vacant places in tile
			{
				for(int tl_idx=rest; tl_idx<cntr; tl_idx++)				// go over entries in vector which can not be registered for a tile
				{
					const int index = tile_vec[tl_idx].second;			// determine internal index of the entry to register
					part_[index]=SET_INVALID;							// reset the tile number
					conn_[index]=SET_INVALID;							// reset connection amount
				}
				cntr=rest;												// number of nodes to register now equals vacant places in tile
			}
			for(int tl_idx=0; tl_idx<cntr; tl_idx++)					// go over entries in vector which are to register
			{
				const int index = tile_vec[tl_idx].second;				// determine internal index of the entry to register
				tiles_[slices_[slice_start+index]]=t_id;				// register the node for a particular tile
				conn_[index]=SET_INVALID;								// reset the flag
			}
			tile_vec.clear();											// erases all elements from the container, but leaves the capacity() of the vector unchanged
			prt_[t_id]+=cntr;											// increment internal counter for a particular tile
		}while(true);
	} // end parallel
	if(NULL!=reports_)													// if optional argument in order to print messages about tiles is set to true
	{
		int rest=0, multi=0;
		#pragma omp parallel for reduction(+: rest) reduction(+: multi)
		for (int n_idx = 0; n_idx < slice_sz; n_idx++)
		{
			const int node_state=part_[n_idx];
			if(SET_INVALID==node_state)
				rest += 1;
			else if(SET_MULTIPLE==node_state)
				multi += 1;
		}
		reports_+=sprintf(reports_, "   tiling summary:\n");
		reports_=stage_handout( tn, 0, prt_, reports_ );
		reports_+=sprintf(reports_, "         remaining nodes: %d of %d \n"
									"         multiple nodes: %d of %d \n"
									, rest, slice_sz, multi, slice_sz);
	}
	int registered=prt_[0];												// summation result of already registered nodes is initialized with amount of registered nodes in first tile
	for( int n_idx=1; n_idx<tn; n_idx++)								// go over the tiles
		registered+=prt_[n_idx];										// sum up amount of registered nodes over all tiles
	return (slice_sz>registered)?true:false;							// if summation result is smaller than amount of nodes in slice there are still nodes to register
}

#undef SET_INVALID
#undef SET_MULTIPLE

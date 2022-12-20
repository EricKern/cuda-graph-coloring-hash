/** 
 * \file metising.cpp
 * @brief Main file with implementation of metising procedure
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <metis.h>			// METIS PartGraphKway API routine
/**
 * @brief Subdivision of layer by invoking external library metis
 * @param[in] tn				: desired number of tiles
 * @param[in] slice_sz			: size of slice to subdivide
 * @param[in] u_factor			: maximum allowed load imbalance among the partitions for metis call
 * @param[in] xadj				: array with amount of neighbors for each node
 * @param[in] adjncy			: array with edges
 * @param[in] adjwgt			: array with weights of edges (can be set to NULL, if all edges have same weight)
 * @param[out] part_			: array with partition information
 */
inline void metis_call(idx_t tn, idx_t slice_sz, const int u_factor, idx_t * xadj, idx_t * adjncy,  idx_t * adjwgt, idx_t * const __restrict__ part_)
{
	idx_t one_=1;						//
	idx_t objval;						// Upon successful completion, this variable stores the edge-cut or the total communication volume of the partitioning solution. The value returned depends on the partitioning’s objective function.
	idx_t options[METIS_NOPTIONS];		// define metis options structure
	METIS_SetDefaultOptions(options);	// set default options
	options[METIS_OPTION_UFACTOR] = u_factor;	// Specifies the maximum allowed load imbalance among the partitions. A value of x indicates that the allowed load imbalance is (1 + x)/1000. For -ptype=rb, the default value is 1 (i.e., load imbalance of 1.001) and for -ptype=kway, the default value is 30 (i.e., load imbalance of 1.03).
	options[METIS_OPTION_NCUTS] = 3;	// Specifies the number of different partitionings that it will compute. The final partitioning is the one that achieves the best edgecut or communication volume. Default is 1.
	options[METIS_OPTION_NUMBERING] = 0;// Used to indicate which numbering scheme is used for the adjacency structure of a graph or the element-node structure of a mesh.
	
	int ret_ = METIS_PartGraphKway(
			&slice_sz,	// nvtxs - The number of vertices in the graph.
			&one_,		// ncon - The number of balancing constraints. It should be at least 1.
			xadj,		// That is, for each vertex i, its adjacency list is stored in consecutive locations in the array adjncy, and the array xadj is used to point to where it begins and where it ends.
			adjncy,		//
			NULL,		// vwgt - The weights of the vertices.
			NULL,		// vsize - The size of the vertices for computing the total communication volume.
			adjwgt,		// adjwgt - The weights of the edges
			&tn,		// nparts - The number of parts to partition the graph.
			NULL,		// tpwgts - This is an array of size nparts×ncon that specifies the desired weight for each partition and constraint.
			NULL,		// ubvec - This is an array of size ncon that specifies the allowed load imbalance tolerance for each constraint.
			options,	// options - This is the array of options
			&objval,	// objval - Upon successful completion, this variable stores the edge-cut or the total communication volume of the partitioning solution. The value returned depends on the partitioning’s objective function.
			part_		// part - This is a vector of size nvtxs that upon successful completion stores the partition vector of the graph.
			);
	if(METIS_OK!=ret_)
		std::cerr << "metis error: " << ret_ << '\n';
}

/**
 * @brief Read in partition information and create tiles
 * @param[in] tn				: desired number of tiles
 * @param[in] slice_sz			: size of slice to subdivide
 * @param[in] slices_			: array with nodes grouped in slices shifted to read current slice
 * @param[in] part_				: array with partition information
 * @param[out] tiles_			: array with corresponding tile number for each node
 * @param[out] prt_				: array with tile sizes
 */
inline void metis_tiles(const int tn, const int slice_sz, const int * const __restrict__ slices_, const idx_t * const __restrict__ part_, int * const __restrict__ tiles_, int * const __restrict__ prt_)
{
	#pragma omp parallel num_threads(tn)
	{
		const int t_id=omp_get_thread_num();							// thread own gang id
		prt_[t_id]=0;													// initialize all tile sizes with zero
		const idx_t * rd_ptr=part_-1;									// set read pointer
		const idx_t * stp_ptr=part_+slice_sz;							// set stop pointer
		while(++rd_ptr<stp_ptr)											// as long as write pointer is in front of stop pointer
			if(t_id==*rd_ptr)											// if thread id equals tile number
			{
				tiles_[slices_[rd_ptr-part_]]=t_id;						// register the node for a particular tile
				prt_[t_id]++;											// increase the counter for the tile chosen
			}
	}
}

#define WEIGHT_UNASSIGNED 2 // one of by the edge connected nodes is still unassigned to a tile
#define WEIGHT_ASSIGNED_SAME 2 // both of by the edge connected nodes are assigned to the same tile
#define WEIGHT_ASSIGNED_DIFFERENT 1 // both of by the edge connected nodes are assigned to different tiles
#define NO_TILE_ASSIGNMENT -1
/**
 * @brief Subdivide dynamically all but maximal slice into tiles by counting the connections of neighbouring layers
 * @see metis_call
 * @see metis_tiles
 * @see stage_handout
 * @param[in] s_id			: index of the current slice
 * @param[in] tn			: desired number of tiles
 * @param[in] slice_start	: marks the start position of a slice (index to the slices array)
 * @param[in] slice_end		: marks the end position of a slice (index to the slices array)
 * @param[in] cond_ 		: 1 if allready partitioned slice is on the left side of the current slice else it is -1
 * @param[in] borders		: array of start and end borders for each thread in csr format
 * @param[in] row_ptr		: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr		: integer array of column indices of the nonzero elements of matrix
 * @param[in] layers_		: array with corresponding layer number for each node
 * @param[in] slice_spot	: array with distance for each node within a slice the node belongs to
 * @param[in,out] slices_	: array with nodes grouped in slices
 * @param[in,out] tiles_	: array with corresponding tile number for each node
 * @param[out] prt_			: contains amount of nodes already assigned to each tile/thread
 * @param[out] reports_		: array for printing of additional information (possibly NULL)
 * @param[out] part_		: array with partition information
 */
inline void metising( const int s_id, const int tn, const int slice_start, const int slice_end, const int cond_, int * const borders_, const int *const row_ptr, const int *const col_ptr, const int * const layers_, const int * const __restrict__ slice_spot, const int * const slices_, int * const tiles_, int * const __restrict__ prt_, char * reports_, idx_t * const __restrict__ part_)
{
	const int slice_sz=slice_end-slice_start;							// size of a slice
	get_borders(slice_sz, tn, borders_);
	#pragma omp parallel num_threads(tn)								// check the first condition (maximal connections for a tile in neighbouring layer)
	{
		const int t_id=omp_get_thread_num();							// thread own gang id
		int * const __restrict__ scn_ = new int[tn];					// each thread allocates memory space for counting the number of connections to the nodes in tiles in the neighbouring slice
		const int end_border=borders_[t_id+1]+slice_start;				// end of area for processing of current threads
		for(int n_idx =slice_start+borders_[t_id]; n_idx<end_border; n_idx++)	// go over the entries
		{
			std::memset(scn_, 0, sizeof(int)*tn);						// initialize scn array with zeroes
			const int node_=slices_[n_idx];								// node number
			const int row_end=row_ptr[node_+1];							// index where following row starts
			for(int row=row_ptr[node_]; row<row_end; row++)				// go over row entries
			{
				const int nei_ = col_ptr[row];							// read in the node number of the neighbor
				if(s_id-layers_[nei_]==cond_)							// if a node is in the neighboring layer
					scn_[tiles_[nei_]]++;								// register for a tile its connected to
			}
			int part_idx=NO_TILE_ASSIGNMENT;							// index to a tile if specific node is connected only with nodes from this tile
			for( int vl=0; vl<tn; vl++ )								// go over the gathered information about the connections
				if( 0!=scn_[vl] )										// if at least one connection is detected
				{
					if(NO_TILE_ASSIGNMENT!=part_idx)					// if the index is set, then connections to multiple tiles are detected
					{
						part_idx=NO_TILE_ASSIGNMENT;					// invalidate the index
						break;											// stop searching
					}else												// if the index is still unset
						part_idx=vl;									// set the index to the tile number
				}
			if(part_idx>NO_TILE_ASSIGNMENT)								// if node is connected only to nodes from one tile
				tiles_[node_]=part_idx;									// register the node for a particular tile
		}
		delete [] scn_;
	}
	std::vector<idx_t> xadj(slice_sz+1);								// In this format the adjacency structure of a graph with n vertices and m edges is represented using two arrays xadj and adjncy
	std::vector<idx_t> adjncy;											// The xadj array is of size n + 1 whereas the adjncy array is of size 2m
	std::vector<idx_t> adjwgt;											// The weights of the edges (if any) are stored in an additional array called adjwgt which contains 2m elements
	xadj[0]=0;															// first index is always zero
	for (int n_idx=slice_start, m_idx=0; n_idx<slice_end; n_idx++)		// go over the nodes in the largest slice
	{
		const int node_=slices_[n_idx];									// extract one node number
		const int node_layer=layers_[node_];							// read in layer number the node is in
		const int row_end=row_ptr[node_+1];								// index to the beginning of the next row in the matrix
		for(int row=row_ptr[node_]; row<row_end; row++)					// go over row
		{
			const int nei_ = col_ptr[row];								// read in the node number of the neighbor
			const int nei_layer=layers_[nei_];							// layer number the neighbour is in
			if((node_layer==nei_layer)&&(node_!=nei_))					// if neighbour is in the same slice and not diagonal element
			{
				adjncy.push_back(slice_spot[nei_]);						// store its index in the adjncy vector
				if(tiles_[node_]<0||tiles_[nei_]<0)						// if either the node or its neighbor is not assigned to a tile yet
					adjwgt.push_back(WEIGHT_UNASSIGNED);				// the weight of such an edge is small
				else
				{
					if(tiles_[node_]!=tiles_[nei_])						// if the nodes don't belong to the same tile
						adjwgt.push_back(WEIGHT_ASSIGNED_DIFFERENT);	// the weight of such an edge is medium
					else
						adjwgt.push_back(WEIGHT_ASSIGNED_SAME);			// the weight of an edge between two nodes of the same tile is large
				}
			}
		}
		xadj[++m_idx]=adjncy.size();									// conclude the count of neighbors by inserting the next entry to the xadj vector
	}
	metis_call(tn, slice_sz, 111, xadj.data(), adjncy.data(), adjwgt.data(), part_);
	metis_tiles(tn, slice_sz, slices_+slice_start, part_, tiles_, prt_);
	if(NULL!=reports_)													// if optional argument in order to print messages about tiles is set to true
	{
		reports_+=sprintf(reports_, "   tiling summary:\n");
		reports_=stage_handout( tn, 0, prt_, reports_ );
	}
}
#undef WEIGHT_UNASSIGNED
#undef WEIGHT_ASSIGNED_SAME
#undef WEIGHT_ASSIGNED_DIFFERENT
#undef NO_TILE_ASSIGNMENT

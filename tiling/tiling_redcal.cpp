/** 
 * \file redcal.cpp
 * @brief Main file with implementation of 'redcal' input flag
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Print out statistic about how many connections there are that cause redundant calculations during execution
 * @param[in] m_rows			: domain size
 * @param[in] tn				: number of started threads
 * @param[in] n_z				: number of slices
 * @param[in] row_ptr			: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr			: integer array of column indices of the nonzero elements of matrix
 * @param[in] ndc_				: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[in] slices_			: array of vectors containing node numbers
 * @param[in] tiles_			: array with corresponding tile number for each node
 * @param[out] stats_ptr		: array of chars intended to hold printing information (possibly NULL)
 */
void redcal_multi( const int m_rows, const int tn, const int n_z, const int *const row_ptr, const int *const col_ptr, const int * const ndc_, const int * const slices_, const int * const tiles_, char * stats_ptr )
{
	int * const __restrict__ redundant_nodes= new int[tn];				// stores the number of non-redundant nodes for each bar
	int * const __restrict__ outside_edges= new int[tn];				// stores the number of redundant connections for each bar
	int * const __restrict__ overall_edges= new int[tn];				// stores the number of overall connections for each bar
	#pragma omp parallel num_threads(tn)
	{
		int inside_edges_thread=0, outside_edges_thread=0, redundant_nodes_thread=0;	// threadwise variables to collect non-redundant and redundant connections for each bar
		const int t_id=omp_get_thread_num();							// thread own gang id
		const int t_end=n_z*(t_id+1) ;									// end index for reading from ndc array
		for(int t_idx = n_z*t_id, current_tile, next_tile=ndc_[t_idx];t_idx<t_end;t_idx++)	// go over the layers
		{
			current_tile=next_tile;
			next_tile=ndc_[t_idx+1];									// read in subsequent tile index
			for(int ndc_idx=current_tile;ndc_idx<next_tile;ndc_idx++)	// go over the entries of a tile in one layer
			{
				const int node_=slices_[ndc_idx];						// read in the node number
				bool covered=true;										// assumption that node has no redundant entries
				const int node_tile=tiles_[node_];						// read in the tile number the node is in
				const int row_end=row_ptr[node_+1];						// end of a row in matrix
				for(int row=row_ptr[node_]; row<row_end; row++)			// go over row entries
				{
					const int nei_ = col_ptr[row];						// read in the node number of the neighbor
					if(node_tile==tiles_[nei_])							// if the entry and the node have the same tile number
						inside_edges_thread++;							// increment the counter for non-redundant connections
					else												// if neighbour is in another bar
					{
						outside_edges_thread++;							// increment the counter for nodes with redundant connections
						if(covered)										// if at least one redundant connection was found
							covered=false;								// node is marked as redundant
					}
				}
				if(!covered)											// if none of found connections is redundant
					redundant_nodes_thread++;							// increment the counter for nodes without redundant connections
			}
		}
		overall_edges[t_id]=inside_edges_thread+outside_edges_thread;	// total number of edges in a bar
		outside_edges[t_id]=outside_edges_thread;						// number of redundant connections
		redundant_nodes[t_id]=redundant_nodes_thread;					// number of non-redundant nodes
	}// end #pragma omp parallel
	double sum=0.0, min=1, max=0, prc, redundant_nodes_sum=0.0;
	for(int bar_=0; bar_<tn; bar_++)									// go through bars
	{
		prc=(0==overall_edges[bar_])?0.0:((double)outside_edges[bar_])/overall_edges[bar_]; // calculated relative number of redundant connections for a bar
		if(prc<min)
			min=prc;
		if(prc>max)
			max=prc;
		sum+=prc;														// sum up relative numbers of redundant connections of all bars in domain
		redundant_nodes_sum+=redundant_nodes[bar_];						// sum up non-redundant nodes of all bars in domain
	}
	delete [] outside_edges;	delete [] overall_edges;
	delete [] redundant_nodes;
	if(NULL==stats_ptr)													// if stdout should be used for printing
	{
		printf( "Statistics for redundant calculations:\n"
				"   Redundant edges:\n"
				"      average:\t%.2f%%\n"								// average number of edges with redundant accesses over all bars in domain
				"      minimum:\t%.2f%%\n"								// minimal number of edges with redundant accesses over all bars in domain
				"      maximum:\t%.2f%%\n"								// maximal number of edges with redundant accesses over all bars in domain
				"   Redundant nodes:\t%.2f%%\n"							// number of nodes in domain with redundant accesses compared to all nodes in domain
				, sum*100/tn, min*100, max*100, redundant_nodes_sum*100/m_rows);
	}
	else																// if printing should be done to stockpile
	{
		sprintf( stats_ptr, "Statistics for redundant calculations:\n"
				"   Redundant edges:\n"
				"      average:\t%.2f%%\n"								// average number of edges with redundant accesses over all bars in domain
				"      minimum:\t%.2f%%\n"								// minimal number of edges with redundant accesses over all bars in domain
				"      maximum:\t%.2f%%\n"								// maximal number of edges with redundant accesses over all bars in domain
				"   Redundant nodes:\t%.2f%%\n"							// number of nodes in domain with redundant accesses compared to all nodes in domain
				, sum*100/tn, min*100, max*100, redundant_nodes_sum*100/m_rows);
	}
}

/**
 * @brief Print out statistic about how many connections there are that cause redundant calculations during execution
 * @param[out] stats_ptr		: array of chars intended to hold printing information (possibly NULL)
 */
void redcal_single(char * stats_ptr)
{
	if(NULL==stats_ptr)													// if stdout should be used for printing
	{
		printf( "Statistics for redundant calculations:\n"
				"   Redundant edges:\t0%%\n"							// number of redundant accesses compared to all matrix accesses during SpMV
				"   Redundant nodes:\t0%%\n");							// number of nodes in domain with redundant accesses compared to all nodes in domain
	}
	else																// if printing should be done to stats
	{
		sprintf( stats_ptr, "Statistics for redundant calculations:\n"
				"   Redundant edges:\t0%%\n"							// number of redundant accesses compared to all matrix accesses during SpMV
				"   Redundant nodes:\t0%%\n");							// number of nodes in domain with redundant accesses compared to all nodes in domain
	}
}

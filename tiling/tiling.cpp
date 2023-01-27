/** 
 * \file tiling.cpp
 * @brief Library main file for slicing and tiling process
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: rgutmann@stud.uni-heidelberg.de.
 **/

#include <assert.h> // assert
#include <omp.h> // include OpenMP pragmas
#include <chrono> // for duration<>
#include <cstring> // memset
#include <iostream> // for printing the messages on stdout
#include <vector> // vector
#include "../properties.hpp" // include tiling properties

using Clock = std::chrono::high_resolution_clock;
using DSeconds = std::chrono::duration<double, std::ratio<1>>;

#include "tiling_accessories.cpp" // no dependencies
#include "tiling_coarsening.cpp" // no dependencies
#include "tiling_equipartitioning.cpp" // no dependencies
#include "tiling_metising.cpp" // stage_handout from accessories.cpp , get_borders from equipartitioning.cpp
#include "tiling_offset.cpp" // no dependencies
#include "tiling_redcal.cpp" // no dependencies
#include "tiling_sectioning.cpp" // stage_handout from accessories.cpp , get_borders from equipartitioning.cpp
#include "tiling_triangular.cpp" // no dependencies

/**
 * @brief Print out number of layers and their sizes
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] ndc_				: integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @param[out] stats_ptr		: array of chars intended to hold printing information (possibly NULL)
 * @return						: array for printing of additional information (possibly NULL)
 */
char* print_linf(const int tn,
                 const int n_z,
                 const int* const ndc_,
                 char* stats_ptr) {
	if (NULL == stats_ptr) // if stdout should be used for printing
	{
		printf("Sizes of %d layers:", n_z);
		for (int i = tn, j = tn * n_z; i <= j; i += tn)
			printf(" %d", ndc_[i] - ndc_[i - tn]);
		printf("\n");
	}
	else // if printing should be done to stats
	{
		stats_ptr += sprintf(stats_ptr, "Sizes of %d layers:", n_z);
		for (int i = tn, j = tn * n_z; i <= j; i += tn)
			stats_ptr += sprintf(stats_ptr, " %d", ndc_[i] - ndc_[i - tn]);
		stats_ptr += sprintf(stats_ptr, "\n");
	}
	return stats_ptr;
}
/**
 * @brief Determine the size of overall statistics information array and allocate sufficient space in heap
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] max_sz			: number of nodes in biggest layer
 * @param[in] ndc_max			: average number of nodes in each tile if divided evenly
 * @param[in] report_sz			: size of tiling output for one layer
 * @param[out] stats_dptr		: pointer to array of chars intended to hold printing information (possibly NULL)
 * @param[in] til_prop			: hold information about settings for secondary and quaternary flags
 * @return						: array for printing of additional information (possibly NULL)
 */
char* alloc_stats(const int tn,
                  const int n_z,
                  const int max_sz,
                  const int ndc_max,
                  const int report_sz,
                  char** const stats_dptr,
                  const uint8_t til_prop) {
	if (NULL == stats_dptr)
		return NULL;
	int stats_amo = report_sz;
	// 20<statistic for slice:>+log10(n_z)+2< %d>+19<   slicing summary:>+2<newline>+1<\0> = 44+log10(n_z)
	//+10<      size>+(log10(max_sz)+2)<\t%d>+1<newline> = 13 + log10(max_sz)
	//+3*9<      min>+2*(log10(ndc_max)+2)<\t%d>+(log10(ndc_max)+5)<\t%.2f>+3<newline> = 39 + 3*log10(ndc_max)
	// = 96+3*log10(ndc_max)+log10(max_sz)+log10(n_z)
	stats_amo += (til_prop & tiling_properties::TILING::PRINT_SINF)
	                 ? (int) std::ceil(96 + 3 * std::log10(ndc_max) +
	                                   std::log10(max_sz) + std::log10(n_z))
	                 : 0;
	//log10(max_sz)+2< %d>
	stats_amo += (til_prop & tiling_properties::TILING::PRINT_LINF)
	                 ? log10(max_sz) + 2
	                 : 0;
	stats_amo *= n_z;
	//18+log10(n_z)<Sizes of %d layers:>
	stats_amo += (til_prop & tiling_properties::TILING::PRINT_LINF)
	                 ? 18 + log10(n_z)
	                 : 0;
	// 38<Statistics for redundant calculations:>+19<   Redundant edges:>+2<newline>+1<\0> = 60
	//+3*(14<      average:>+8<\t%.2f%%>+1<newline>) = 69
	//+19<   Redundant nodes:>+8<\t%.2f%%>+1<newline> = 28
	//= 157
	stats_amo += (til_prop & tiling_properties::TILING::PRINT_RINF) ? 157 : 0;
	return *stats_dptr = (0 == stats_amo) ? NULL : new char[stats_amo];
}

/**
 * @brief Determine the size of tiling statistics information array and allocate sufficient space in heap
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers
 * @param[in] max_sz			: number of nodes in biggest layer
 * @param[in] ndc_max			: average number of nodes in each tile if divided evenly
 * @param[in] til_prop			: hold information about settings for secondary and quaternary flags
 * @return						: array for printing of additional information (possibly NULL)
 */
int get_report_size(const int tn,
                    const int n_z,
                    const int max_sz,
                    const int ndc_max,
                    const uint8_t til_prop) {
	return (til_prop & tiling_properties::TILING::PRINT_TINF)
	           ? (til_prop & tiling_properties::TILING::METIS)
	                 ?
	                 // 20<statistic for slice:>+log10(n_z)+2< %d>+18<   tiling summary:>+14<      stage 0:>+<(log10(ndc_max)+2)*tn>+3<newline>+1<\0> [--tinf -m]
	                 // = 58 + log10(n_z) + (log10(ndc_max)+2)*tn
	                 (int) std::ceil(58 + std::log10(n_z) +
	                                 (std::log10(ndc_max) + 2) * tn)
	                 :
	                 // 20<statistic for slice:>+log10(n_z)+2< %d>+18<   tiling summary:>+2<newline>+1<\0> = 43+log10(n_z)
	                 //+3*(14<      stage 0:>+(log10(ndc_max)+2)*tn< %d... %d>+1<newline>) = 45 + 3*(log10(ndc_max)+2)*tn
	                 //+2*(28+(2*(log10(max_sz)+2))<        remaining nodes: 0 of 12>+27+(2*(log10(max_sz)+2))<        multiple nodes: 0 of 12>+2<newline>) = 114+8*(log10(max_sz)+2)
	                 //+37<         calls for random assignment:>+log10(max_sz)+2< %d>+24<         nodes added at ones:>+log10(max_sz)+2< %d>+2<newline> = 63+2*(log10(max_sz)+2)
	                 // = 265 + log10(n_z) + 3*(log10(ndc_max)+2)*tn + 10*(log10(max_sz)+2)
	                 // = 285 + log10(n_z) + 3*(log10(ndc_max)+2)*tn + 10*log10(max_sz)
	                 (int) std::ceil(285 + std::log10(n_z) +
	                                 3 * tn * (std::log10(ndc_max) + 2) +
	                                 10 * std::log10(max_sz))
	           : 0;
}

/**
 * @brief Subdivide layers in grouped rows called tiles
 * @see alloc_stats
 * @see get_report_size
 * @see coarsening_multi
 * @see coarsening_single
 * @see redcal_multi
 * @see offset_multi
 * @see form_tiles
 * @see init_sectioning
 * @see final_sectioning
 * @see create_til_sum
 * @see metis_call
 * @see metis_tiles
 * @param[in] m_rows			: number of rows of matrix
 * @param[in] tn				: number of tiles in each layer
 * @param[in] n_z				: overall number of layers after slicing
 * @param[in] row_ptr			: integer array that contains the start of every row and the end of the last row plus one
 * @param[in] col_ptr			: integer array of column indices of the nonzero elements of matrix
 * @param[in] layers_			: array with corresponding layer number for each node
 * @param[in] indices_			: integer array of n_z+1 elements that contains the start of every layer in slices array and the end of the last layer in slices array
 * @param[in] slices_			: pointer to integer array of m_rows elements containing row numbers
 * @param[out] ndc_dptr			: pointer to integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile (can be set to NULL, if no output required)
 * @param[out] tesserae_dptr	: pointer to array containing node numbers grouped in slices and subdivided within each slice into tiles (can be set to NULL, if no output required)
 * @param[out] tiles_dptr		: pointer to array with corresponding tile number for each node (can be set to NULL, if no output required)
 * @param[out] offsets_dptr		: pointer to array with indices for each tile marking line between nodes with and without redundant values (can be set to NULL, if no output required)
 * @param[out] stats_dptr		: pointer to array of chars intended to hold printing information if printing flags are set in til_prop (can be set to NULL to print to stdout)
 * @param[out] n_z_ptr			: pointer to overall number of layers (can be set to NULL, if no output required)
 * @param[in] til_prop			: bit-field managing functionalities within slicing or tiling process
 * @return						: time measurement for tiling process
 */
double tiling_partitioning(const int m_rows,
                           const int tn,
                           const int n_z,
                           const int* const row_ptr,
                           const int* const col_ptr,
                           const int* const layers_,
                           const int* const indices_,
                           const int* const slices_,
                           int** const ndc_dptr,
                           int** const tesserae_dptr,
                           int** const tiles_dptr,
                           int** const offsets_dptr,
                           char** const stats_dptr,
                           int* const n_z_ptr,
                           const uint8_t til_prop) {
	assert(m_rows > 0); // matrix consist of at least one row
	assert(tn > 0); // number of threads is at least one
	assert(n_z > 0); // number of slices has to be at least one
	assert(NULL != row_ptr); // pointer to row indices must be set
	assert(NULL != col_ptr); // pointer to column indices must be set
	assert(NULL != layers_); // array with layer number for each node in domain
	assert(NULL != indices_); // array with accumulated slice sizes
	assert(NULL != slices_); // array with slice nodes must be set
	int* write_ptr = (NULL == tesserae_dptr) ? NULL
	                 : (slices_ == *tesserae_dptr)
	                     ? * tesserae_dptr
	                     : * tesserae_dptr = new int[m_rows];
	int* const __restrict__ tiles_ = new int
	    [m_rows]; // allocate sufficient space for tiles array and extract pointer out of double pointer

	const int init_var = (1 == tn) ? 0 : -1;
#pragma omp parallel for
	for (int n_idx = 0; n_idx < m_rows; n_idx++) // go over the nodes in a slice
		tiles_[n_idx] = init_var; // initialize tiles_ entries
	int max_sz, max_idx = n_z - 1;
	{
		int slc_curr = indices_[n_z - 1],
		    slc_prev = indices_[n_z]; // read in last two indices
		max_sz = slc_prev - slc_curr; // calculate the size of last slice
		for (int n_idx = n_z - 1; n_idx > 0;
		     n_idx--) // go over the indices downwards
		{
			slc_prev =
			    slc_curr; // overwrite the biggest index yet with smaller one
			slc_curr = indices_[n_idx - 1]; // read in lower index
			const int slice_sz =
			    slc_prev - slc_curr; // calculate the size of one slice
			if (max_sz <
			    slice_sz) // if the size of slice is bigger than maximal one
			{
				max_sz = slice_sz; // overwrite the value for maximal slice
				max_idx = n_idx - 1; // set proper index for maximal slice
			}
		}
	}
	std::chrono::time_point<std::chrono::high_resolution_clock> start_,
	    end_; // time points for measurements
	start_ = Clock::now(); // start time measurements

	if (1 < tn) {
		int tile_amo =
		    tn *
		    n_z; // overall number of tiles ( can be changed during coarsening )
		int* const ndc_ =
		    new int[tile_amo + 1]; // allocate sufficient space for ndc array
		ndc_[tile_amo] = m_rows; // insert the last index
		int* prt_r = new int
		    [tn]; // contains amount of nodes already assigned to each tile/thread
		const int slice_start = indices_
		    [max_idx]; // the index of the first node in the largest slice
		const int slice_end =
		    indices_[max_idx +
		             1]; // the index of the node behind the largest slice
		idx_t* part_r = new int[max_sz]; // array with partition information
		int* const __restrict__ slice_spot = new int
		    [m_rows]; // array with distance for each node within a slice the node belongs to
		for (int ind_idx = 0, slc_end = indices_[0]; ind_idx < n_z;
		     ind_idx++) // go over slices
		{
			const int slc_start = slc_end; // set the begin for current slice
			slc_end =
			    indices_[ind_idx + 1]; // read in the begin of following slice
			for (int slc_idx = slc_start; slc_idx < slc_end;
			     slc_idx++) // go over slice nodes
				slice_spot[slices_[slc_idx]] =
				    slc_idx -
				    slc_start; // store for each node its position within current slice
		}
		{
			std::vector<idx_t> xadj(
			    max_sz +
			    1); // In this format the adjacency structure of a graph with n vertices and m edges is represented using two arrays xadj and adjncy
			std::vector<idx_t>
			    adjncy; // The xadj array is of size n + 1 whereas the adjncy array is of size 2m
			xadj[0] = 0; // first index is always zero
			for (int n_idx = slice_start, m_idx = 0; n_idx < slice_end;
			     n_idx++) // go over the nodes in the largest slice
			{
				const int node_ = slices_[n_idx]; // extract one node number
				const int node_layer =
				    layers_[node_]; // read in layer number the node is in
				const int row_end = row_ptr
				    [node_ +
				     1]; // index to the beginning of the next row in the matrix
				for (int row = row_ptr[node_]; row < row_end;
				     row++) // go over row
				{
					const int nei_ =
					    col_ptr[row]; // read in the node number of the neighbor
					const int nei_layer =
					    layers_[nei_]; // layer number the neighbour is in
					if (node_layer == nei_layer &&
					    node_ !=
					        nei_) // if neighbour is in the same slice and not diagonal element
						adjncy.push_back(
						    slice_spot
						        [nei_]); // store its index in the adjncy vector
				}
				xadj[++m_idx] =
				    adjncy
				        .size(); // conclude the count of neighbors by inserting the next entry to the xadj vector
			}

			metis_call(tn, max_sz, 1, xadj.data(), adjncy.data(), NULL, part_r);
			metis_tiles(
			    tn, max_sz, slices_ + slice_start, part_r, tiles_, prt_r);
		}
		int ndc_max = prt_r
		    [0]; // the number of nodes in tile zero is taken as initial value for searching of maximal tile
		for (int n_idx = 1; n_idx < tn; n_idx++) // go over the tiles
			if (prt_r[n_idx] >
			    ndc_max) // if tile size is bigger than the maximal one
				ndc_max = prt_r[n_idx]; // overwrite the maximal value
		const int report_sz =
		    get_report_size(tn, n_z, max_sz, ndc_max, til_prop);
		char* reports_r =
		    (0 < report_sz)
		        ? new char[report_sz]
		        : NULL; // array for printing of additional information
		if (NULL !=
		    reports_r) // if optional argument in order to print messages about tiles is set to true
		{
			sprintf(reports_r, "   tiling summary:\n");
			stage_handout(tn, 0, prt_r, strchr(reports_r, '\0'));
		}
		char* stats_ptr = alloc_stats(
		    tn,
		    n_z,
		    max_sz,
		    ndc_max,
		    report_sz,
		    stats_dptr,
		    til_prop); // allocate sufficient space for printing information if printing flags are set in til_prop
		form_tiles(max_idx,
		           tn,
		           slice_start,
		           tiles_,
		           slices_,
		           reports_r,
		           prt_r,
		           &stats_ptr,
		           ndc_,
		           write_ptr,
		           til_prop & tiling_properties::TILING::PRINT_SINF);
		int* borders_r = new int[tn + 1];
		if (til_prop & tiling_properties::TILING::METIS) // metising
		{
			if (max_idx > 0) {
				int slice_idx[2] = {slice_start, 0};
				for (int s_id = max_idx - 1; s_id >= 0;
				     s_id--) // left-side slices partitioning
				{
					slice_idx[1] = slice_idx[0];
					slice_idx[0] = indices_[s_id];
					metising(s_id,
					         tn,
					         slice_idx[0],
					         slice_idx[1],
					         -1,
					         borders_r,
					         row_ptr,
					         col_ptr,
					         layers_,
					         slice_spot,
					         slices_,
					         tiles_,
					         prt_r,
					         reports_r,
					         part_r);
					form_tiles(
					    s_id,
					    tn,
					    slice_idx[0],
					    tiles_,
					    slices_,
					    reports_r,
					    prt_r,
					    &stats_ptr,
					    ndc_,
					    write_ptr,
					    til_prop & tiling_properties::TILING::PRINT_SINF);
				}
			}
			if (max_idx < (n_z - 1)) {
				int slice_idx[2] = {0, slice_end};
				for (int s_id = max_idx + 1; s_id < n_z;
				     s_id++) // right-side slices partitioning
				{
					slice_idx[0] = slice_idx[1];
					slice_idx[1] = indices_[s_id + 1];
					metising(s_id,
					         tn,
					         slice_idx[0],
					         slice_idx[1],
					         1,
					         borders_r,
					         row_ptr,
					         col_ptr,
					         layers_,
					         slice_spot,
					         slices_,
					         tiles_,
					         prt_r,
					         reports_r,
					         part_r);
					form_tiles(
					    s_id,
					    tn,
					    slice_idx[0],
					    tiles_,
					    slices_,
					    reports_r,
					    prt_r,
					    &stats_ptr,
					    ndc_,
					    write_ptr,
					    til_prop & tiling_properties::TILING::PRINT_SINF);
				}
			}
		}
		else // sectioning
		{
			int* prt_f = new int
			    [tn]; // array containing amount of nodes already assigned to each tile/thread
			char* reports_f =
			    (0 < report_sz)
			        ? new char[report_sz]
			        : NULL; // array for printing of additional information
			int* part_f = new int
			    [max_sz]; // array which stores the index of a tile with maximal connections the node should be registered to
			int* conn_ = new int
			    [max_sz]; // array which stores maximal number of connections to tile specified in part_
			int* borders_f = new int
			    [tn +
			     1]; // array of start and end borders for each thread in csr format
			int* const direct_ = new int
			    [tn *
			     tn]; // array which stores the number of connections to the nodes in tiles in the neighbouring slice
			int* const indirect_ = new int
			    [tn *
			     tn]; // array which stores the number of connections to the tiles in the neighbouring slice from the adjacent nodes in the same layer
			if (max_idx > 0) // if maximal slice has not the smallest index
			{
				int slice_idx[3] = {
				    (int) indices_[max_idx - 1], slice_start, 0};
				get_borders(
				    slice_idx[1] - slice_idx[0],
				    tn,
				    borders_r); // determine for all participating threads areas to proceed
				bool regs[2] = {init_sectioning(max_idx - 1,
				                                tn,
				                                slice_idx[0],
				                                slice_idx[1],
				                                ndc_max,
				                                -1,
				                                borders_r,
				                                row_ptr,
				                                col_ptr,
				                                layers_,
				                                slices_,
				                                tiles_,
				                                prt_r,
				                                reports_r,
				                                part_r,
				                                conn_,
				                                direct_,
				                                indirect_),
				                false};
				for (int s_id = max_idx - 1; s_id > 0;
				     s_id--) // left-side slices partitioning
				{
					slice_idx[2] = slice_idx[1];
					slice_idx[1] = slice_idx[0];
					slice_idx[0] = indices_[s_id - 1];
					get_borders(
					    slice_idx[1] - slice_idx[0],
					    tn,
					    borders_f); // determine for all participating threads areas to proceed
					regs[1] = regs[0];
					regs[0] = init_sectioning(s_id - 1,
					                          tn,
					                          slice_idx[0],
					                          slice_idx[1],
					                          ndc_max,
					                          -1,
					                          borders_f,
					                          row_ptr,
					                          col_ptr,
					                          layers_,
					                          slices_,
					                          tiles_,
					                          prt_f,
					                          reports_f,
					                          part_f,
					                          conn_,
					                          direct_,
					                          indirect_);
					if (regs[1])
						final_sectioning(s_id,
						                 tn,
						                 slice_idx[1],
						                 slice_idx[2],
						                 ndc_max,
						                 borders_r,
						                 row_ptr,
						                 col_ptr,
						                 layers_,
						                 slices_,
						                 tiles_,
						                 prt_r,
						                 reports_r,
						                 part_r,
						                 conn_,
						                 direct_,
						                 indirect_);
					form_tiles(
					    s_id,
					    tn,
					    slice_idx[1],
					    tiles_,
					    slices_,
					    reports_r,
					    prt_r,
					    &stats_ptr,
					    ndc_,
					    write_ptr,
					    til_prop & tiling_properties::TILING::PRINT_SINF);
					std::swap(prt_r, prt_f);
					std::swap(reports_r, reports_f);
					std::swap(part_r, part_f);
					std::swap(borders_r, borders_f);
				}
				if (regs[0])
					final_sectioning(0,
					                 tn,
					                 slice_idx[0],
					                 slice_idx[1],
					                 ndc_max,
					                 borders_r,
					                 row_ptr,
					                 col_ptr,
					                 layers_,
					                 slices_,
					                 tiles_,
					                 prt_r,
					                 reports_r,
					                 part_r,
					                 conn_,
					                 direct_,
					                 indirect_);
				form_tiles(0,
				           tn,
				           slice_idx[0],
				           tiles_,
				           slices_,
				           reports_r,
				           prt_r,
				           &stats_ptr,
				           ndc_,
				           write_ptr,
				           til_prop & tiling_properties::TILING::PRINT_SINF);
			}
			const int s_end = n_z - 1;
			if (max_idx < s_end) // if maximal slice has not the biggest index
			{
				int slice_idx[3] = {0, slice_end, (int) indices_[max_idx + 2]};
				get_borders(
				    slice_idx[2] - slice_idx[1],
				    tn,
				    borders_r); // determine for all participating threads areas to proceed
				bool regs[2] = {false,
				                init_sectioning(max_idx + 1,
				                                tn,
				                                slice_idx[1],
				                                slice_idx[2],
				                                ndc_max,
				                                1,
				                                borders_r,
				                                row_ptr,
				                                col_ptr,
				                                layers_,
				                                slices_,
				                                tiles_,
				                                prt_r,
				                                reports_r,
				                                part_r,
				                                conn_,
				                                direct_,
				                                indirect_)};
				for (int s_id = max_idx + 1; s_id < s_end;
				     s_id++) // right-side slices partitioning
				{
					slice_idx[0] = slice_idx[1];
					slice_idx[1] = slice_idx[2];
					slice_idx[2] = indices_[s_id + 2];
					get_borders(
					    slice_idx[2] - slice_idx[1],
					    tn,
					    borders_f); // determine for all participating threads areas to proceed
					regs[0] = regs[1];
					regs[1] = init_sectioning(s_id + 1,
					                          tn,
					                          slice_idx[1],
					                          slice_idx[2],
					                          ndc_max,
					                          1,
					                          borders_f,
					                          row_ptr,
					                          col_ptr,
					                          layers_,
					                          slices_,
					                          tiles_,
					                          prt_f,
					                          reports_f,
					                          part_f,
					                          conn_,
					                          direct_,
					                          indirect_);
					if (regs[0])
						final_sectioning(s_id,
						                 tn,
						                 slice_idx[0],
						                 slice_idx[1],
						                 ndc_max,
						                 borders_r,
						                 row_ptr,
						                 col_ptr,
						                 layers_,
						                 slices_,
						                 tiles_,
						                 prt_r,
						                 reports_r,
						                 part_r,
						                 conn_,
						                 direct_,
						                 indirect_);
					form_tiles(
					    s_id,
					    tn,
					    slice_idx[0],
					    tiles_,
					    slices_,
					    reports_r,
					    prt_r,
					    &stats_ptr,
					    ndc_,
					    write_ptr,
					    til_prop & tiling_properties::TILING::PRINT_SINF);
					std::swap(prt_r, prt_f);
					std::swap(reports_r, reports_f);
					std::swap(part_r, part_f);
					std::swap(borders_r, borders_f);
				}
				if (regs[1])
					final_sectioning(s_end,
					                 tn,
					                 slice_idx[1],
					                 slice_idx[2],
					                 ndc_max,
					                 borders_r,
					                 row_ptr,
					                 col_ptr,
					                 layers_,
					                 slices_,
					                 tiles_,
					                 prt_r,
					                 reports_r,
					                 part_r,
					                 conn_,
					                 direct_,
					                 indirect_);
				form_tiles(s_end,
				           tn,
				           slice_idx[1],
				           tiles_,
				           slices_,
				           reports_r,
				           prt_r,
				           &stats_ptr,
				           ndc_,
				           write_ptr,
				           til_prop & tiling_properties::TILING::PRINT_SINF);
			}
			if (NULL != reports_f)
				delete[] reports_f;
			delete[] prt_f;
			delete[] conn_;
			delete[] part_f;
			delete[] borders_f;
			delete[] direct_;
			delete[] indirect_;
		}
		delete[] part_r;
		delete[] borders_r;
		delete[] prt_r;
		delete[] slice_spot;
		if (NULL != reports_r)
			delete[] reports_r;
		if (NULL !=
		    write_ptr) // if pointer for output writing is NULL no output is desired
		{
			for (int til_idx = tile_amo - 1,
			         til_prev,
			         til_curr = ndc_[tile_amo];
			     til_idx > -1;
			     til_idx--) // go over the indices
			{
				til_prev = til_curr; // set the begin of following tile
				til_curr = ndc_[til_idx]; // read in the begin of current tile
				const int til_sz =
				    til_prev - til_curr; // calculate the size of one tile
				if (ndc_max <
				    til_sz) // if the size of tile is bigger than maximal one
					ndc_max = til_sz; // overwrite the value for maximal tile
			}
			int* til_sum;
			int n_z_new = n_z;
			if (!(til_prop & tiling_properties::TILING::
			                     FINE)) // if layer merging is desired
			{
				n_z_new = coarsening_multi(tn,
				                           ndc_max,
				                           n_z,
				                           ndc_,
				                           write_ptr,
				                           ndc_,
				                           write_ptr,
				                           &til_sum);
				tile_amo = n_z_new * tn; // recalculate overall number of tiles
			}
			else
				create_til_sum(tn, n_z, ndc_, &til_sum);
			if (til_prop &
			    tiling_properties::TILING::
			        PRINT_LINF) // if output of number of layers and their sizes is desired
				stats_ptr = print_linf(tn, n_z_new, ndc_, stats_ptr);
			int* const ndc_new = new int[tile_amo + 1];
			int* const slices_new = new int[m_rows];
			ndc_new[0] = 0; // write first index of first tile
#pragma omp parallel num_threads(tn)
			{
				const int t_id = omp_get_thread_num(); // thread own gang id
				int t_ndc =
				    (t_id + 1) *
				    n_z_new; // begin of the area for the following thread
				int* wrt_ptr =
				    &slices_new[til_sum[t_id]]; // thread own write pointer
				for (int idx = tn * (n_z_new - 1) + t_id,
				         ndc_sz = til_sum[t_id];
				     idx > -1;
				     idx -= tn) // go over the layers
				{
					ndc_new[t_ndc--] = ndc_sz; // write end index of a tile
					const int ind_end =
					    ndc_[idx] -
					    1; // index to the end of a tile in the vector slices
					const int ind_begin =
					    ndc_[idx + 1] -
					    1; // index to the begin of a tile in the vector slices
					for (int ind = ind_begin; ind > ind_end;
					     ind--) // go over the tile nodes
						*(--wrt_ptr) =
						    write_ptr[ind]; // write nodes in permuted order
					ndc_sz -=
					    ind_begin -
					    ind_end; // decrement the overall size by current tile size
				}
			}
			delete[] write_ptr;
			delete[] ndc_;
			delete[] til_sum;
			*tesserae_dptr =
			    slices_new; // redirect output pointer to output memory location
			if (til_prop &
			    tiling_properties::TILING::
			        PRINT_RINF) // if statististics about redundant connections are desired
				redcal_multi(m_rows,
				             tn,
				             n_z_new,
				             row_ptr,
				             col_ptr,
				             ndc_new,
				             slices_new,
				             tiles_,
				             stats_ptr);
			if (NULL !=
			    offsets_dptr) // if separation line between nodes with and without redundant calculations for each tile is desired
				*offsets_dptr = offset_multi(tn,
				                             n_z_new,
				                             ndc_max,
				                             row_ptr,
				                             col_ptr,
				                             ndc_new,
				                             slices_new,
				                             tiles_,
				                             slices_new);
			if (NULL !=
			    ndc_dptr) // if structure containing the start of every tile and the end of the last tile is desired
				*ndc_dptr =
				    ndc_new; // redirect output pointer to structure containing the start of every tile and the end of the last tile
			else
				delete[] ndc_new; // delete allocated memory space
			if (NULL != n_z_ptr) // if overall number of layers is desired
				*n_z_ptr = n_z_new; // write out new overall number of layers
		} // end write desired
		else {
			if (NULL !=
			    ndc_dptr) // if structure containing the start of every tile and the end of the last tile is desired
				*ndc_dptr =
				    ndc_; // redirect output pointer to structure containing the start of every tile and the end of the last tile
			else
				delete[] ndc_; // delete allocated memory space
			if (NULL != n_z_ptr) // if overall number of layers is desired
				*n_z_ptr = n_z; // write out old overall number of layers
		}
	} // end 1<tn
	else // 1=tn
	{
		char* stats_ptr =
		    (*stats_dptr = (til_prop & tiling_properties::TILING::PRINT_RINF &&
		                    NULL != stats_dptr)
		                       ? new char[86]
		                       : NULL);
		int* ndc_ = (int*) indices_;
		int n_z_new = n_z;
		if (NULL !=
		    write_ptr) // if pointer for output writing is NULL no output is desired
		{
			if (!(til_prop & tiling_properties::TILING::
			                     FINE)) // if layer merging is desired
				n_z_new = coarsening_single(max_sz, n_z, indices_, ndc_);
			if (til_prop &
			    tiling_properties::TILING::
			        PRINT_LINF) // if output of number of layers and their sizes is desired
				stats_ptr = print_linf(1, n_z_new, ndc_, stats_ptr);
			if (til_prop &
			    tiling_properties::TILING::
			        PRINT_RINF) // if statististics about redundant connections are desired
				redcal_single(stats_ptr);
			if (NULL !=
			    offsets_dptr) // if separation line between nodes with and without redundant calculations for each tile is desired
				*offsets_dptr = offset_single(n_z_new, ndc_);
		} // end write desired
		if (NULL !=
		    ndc_dptr) // if structure containing the start of every tile and the end of the last tile is desired
			*ndc_dptr =
			    ndc_; // redirect output pointer to structure containing the start of every tile and the end of the last tile
		if (NULL != n_z_ptr) // if overall number of layers is desired
			*n_z_ptr = n_z_new; // write out new overall number of layers
	} // end 1=tn
	if (NULL !=
	    tiles_dptr) // if structure with tile number for each node is desired
		*tiles_dptr =
		    tiles_; // redirect output pointer to structure with tile number for each node
	else
		delete[] tiles_; // delete allocated memory space
	end_ = Clock::now(); // stop of the time measurement
	return DSeconds(end_ - start_).count(); // return duration
}

// utility functions
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <memory>


/// @brief 
/// @param [in] m_rows 
/// @param [in] number_of_tiles 
/// @param [in] row_ptr 
/// @param [in] col_ptr 
/// @param [out] slices_dptr 
/// @param [out] ndc_dptr 
/// @param [out] offsets_dptr 
void simple_tiling(const int m_rows, const int number_of_tiles,
                   const int* const row_ptr, const int* const col_ptr,
                   int** const slices_dptr, int** const ndc_dptr,
                   int** const offsets_dptr) {
  int indices_[] = {0, m_rows};
  int number_of_slices = 1;
  int* slices_ = new int[m_rows];
  int* layers_ = new int[m_rows];
  for (int i = 0; i < m_rows; i++) {
    slices_[i] = i;
    layers_[i] = 0;
  }
  *slices_dptr = slices_;
  tiling_partitioning(m_rows, number_of_tiles, number_of_slices, row_ptr,
                      col_ptr, layers_, indices_, slices_, ndc_dptr,
                      slices_dptr, nullptr, offsets_dptr, nullptr,
                      &number_of_slices, 0);
  delete[] layers_;
}


/// @brief 
/// @param [in] number_of_tiles 		:
/// @param [in] ndc_ 					: indices separating the matrix in tiles
/// @param [in] row_ptr 				: where ndc_ points to. Points to col_indices
/// @param [out] biggest_tileNodes  	: nr nodes in biggest tile
/// @param [out] biggest_tileEdges  	: nr edges in biggest tile
/// @param [out] maxTileSize 			: maximum nr of nodes in any tile
/// @param [out] maxEdges 				: max length of col_ptr array in any tile
void get_MaxTileSize(const int number_of_tiles,
                     const int* const ndc_,
                     const int* const row_ptr,
					 int* biggest_tileNodes,
					 int* biggest_tileEdges,
                     int* maxTileSize,
                     int* maxEdges) {
  int biggest_tile_nodes = 0;
  int biggest_tile_edges = 0;
  int max_nodes = 0;
  int max_edges = 0;
  // go over tiles
  for (int tile_nr = number_of_tiles; tile_nr > 0; --tile_nr) {
    // calculate tile size
    int tile_sz = ndc_[tile_nr] - ndc_[tile_nr - 1];
    int tile_edges = row_ptr[ndc_[tile_nr]] - row_ptr[ndc_[tile_nr - 1]];

    if (tile_edges + tile_sz > biggest_tile_nodes + biggest_tile_edges) {
      biggest_tile_nodes = tile_sz;
      biggest_tile_edges = tile_edges;
	}
	if (tile_sz > max_nodes) {
      max_nodes = tile_sz;
	}
	if (tile_edges > max_edges) {
      max_edges = tile_edges;
	}
  }
  // Return:
  *maxTileSize = max_nodes;
  *maxEdges = max_edges;
  *biggest_tileNodes = biggest_tile_nodes;
  *biggest_tileEdges = biggest_tile_edges;

}

/// @brief 
/// @param [in] row_ptr 
/// @param [in] m_rows 
/// @param [in] max_tile_size_byte 
/// @param [out] tile_boundaries 
/// @param [out] n_tiles 
/// @param [out] max_node_degree 
void very_simple_tiling(int* row_ptr,
                        int m_rows,
                        int max_tile_size_byte,
                        std::unique_ptr<int[]>* tile_boundaries,
                        int* n_tiles,
                        int* max_node_degree) {
  int len_row_ptr = m_rows + 1;
  std::vector<int> n_neighbors(len_row_ptr);
  auto start1 = row_ptr;
  auto end1 = row_ptr + len_row_ptr;

  std::adjacent_difference(start1, end1, n_neighbors.begin());
  // first element of n_neighbors unmodified. Differences starts at first elem

  auto max_idx = std::max_element(n_neighbors.begin() + 1, n_neighbors.end());
  *max_node_degree = *max_idx;


  std::vector<int> boundaries(len_row_ptr); //allocate for worst case
  boundaries[0] = 0;

  // Iterate over row_ptr and try to fit as many consecutive rows (including
  // adjacent columns) in one partition so that each partition doesn't exceed
  // the given memory requirement
  int current_size = 0;
  int tile_rows = 0;
  int tile_cols = 0;
  int tile_num = 1;
  for (int i = 1; i < m_rows+1; ++i) {
    tile_rows = i - boundaries[tile_num - 1];
    tile_cols += n_neighbors[i];
    current_size = tile_cols + tile_rows + 1;
    if (sizeof(int) * current_size > max_tile_size_byte) {
      if (tile_rows == 1) {
        std::printf("Error:");
        throw std::runtime_error(
            "Cannot fit big row in one tile. Increase tile size for this matrix");
      }
      // here we are already one row to big
      boundaries[tile_num] = i - 1;
      i -= 1;
      tile_cols = 0;
      tile_num += 1;
    }
  }
  // last tile not filled to the maximum
  if (tile_cols != 0 ) {
    boundaries[tile_num] = m_rows;
  }
  boundaries.resize(tile_num + 1);

  *tile_boundaries = std::make_unique<int[]>(tile_num + 1);
  std::copy(boundaries.begin(), boundaries.end(), tile_boundaries->get());
  *n_tiles = tile_num;

}
/** 
 * \file equipartitioning.cpp
 * @brief Main file with implementation of uniform tiling
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Division of workload among participating threads
 * @param[in] m_rows			: number of rows in the matrix
 * @param[in] tn				: number of started threads
 * @param[out] borders			: array of start borders for each thread and the end border for the last thread
 */
inline void get_borders(const int m_rows, const int tn, int* const borders_) {
	const int work_min = m_rows / tn; // minimal work area for each thread
	const int up_worker =
	    m_rows %
	    tn; // amount of threads that handle one more element as minimal work area
	const int work_max = work_min + 1; // maximal workload for a thread
#pragma omp parallel num_threads(tn)
	{
		const int t_id =
		    omp_get_thread_num(); // each thread gets its number in the gang
		if (t_id <=
		    up_worker) // if thread belongs to those with maximal workload
			borders_[t_id] =
			    t_id *
			    work_max; // write out the begin of workload area in matrix for each thread
		else // if thread belongs to those with minimal workload
			borders_[t_id] =
			    up_worker * work_max +
			    (t_id - up_worker) *
			        work_min; // write out the begin of workload area in matrix for each thread
		if (tn - 1 == t_id) // the last thread in the gang
			borders_[t_id + 1] =
			    m_rows; // write out the end of workload area in matrix for the last thread
	}
}

/**
 * @brief Equal distribution of adjacent nodes among tiles in the single layer
 * @param[in] m_rows			: number of nodes in layer
 * @param[in] tn				: number of tiles in layer
 * @param[out] ndc_dptr			: pointer to integer array of tn*n_z+1 elements that contains the start of every tile and the end of the last tile
 * @return						: time measurements for equipartitioning
 */
double tiling_equipartitioning(const int m_rows, const int tn, int** ndc_dptr) {
	assert(m_rows > 0); // matrix consist of at least one row
	assert(tn > 0); // number of threads is at least one
	if (NULL == ndc_dptr)
		return 0.0;
	int* const ndc_ptr = new int[tn + 1]; // allocate sufficient space for ndc
	std::chrono::time_point<std::chrono::high_resolution_clock> start_,
	    end_; // time points for measurements
	start_ = Clock::now(); // start of the time measurement
	get_borders(m_rows,
	            tn,
	            ndc_ptr); // division of workload among participating threads
	end_ = Clock::now(); // stop of the time measurement
	*ndc_dptr = ndc_ptr; // redirect output pointer to output memory location
	return DSeconds(end_ - start_)
	    .count(); // return elapsed time spend for equipartitioning
}

/** 
 * \file cpumultiply_printing.cpp
 * @brief File with interfaces for printing function
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

/**
 * @brief Print of comparison summary
 * @param[in] tn				: number of tiles in each layer
 * @param[in] ndc_				: array of indices for slicing array
 * @param[in] ref_ptr			: reference array
 * @param[in] sol_ptr			: calculated solution array
 */
template <class prcsn>
void compare_summary(const int tn,
                     const int* const ndc_,
                     const prcsn* const ref_ptr,
                     const prcsn* const sol_ptr) {
	const int m_rows = ndc_[tn]; // number of rows in matrix
	const int max_id =
	    omp_get_max_threads(); // maximum number of available threads
	double diffs
	    [3][max_id]; // an array to hold intermediate err values of each thread
	int zeroes
	    [2]
	    [max_id]; // an array to hold number of zero nor undefined values of each thread
#pragma omp parallel num_threads(max_id)
	{
		const int t_id = omp_get_thread_num(); // thread specific identification
		double absolute_diff,
		    absolute_err =
		        0.0f; // help buffers for storage of absolute difference(absolute_diff holds in each loop the intermediate value and absolute_err the global value over made loops)
		double relative_diff,
		    relative_err =
		        0.0f; // help buffers for storage of relative difference(relative_diff holds in each loop the intermediate value and relative_err the global value over made loops)
		double euclidean_dist =
		    0.0f; // stores the euclidean distance between two points in euclidean_dist
		int zero_thread = 0,
		    inf_thread =
		        0; // the output value cannot hold neither zero nor undefined values(error management)
		for (int i = t_id; i < m_rows; i += max_id) {
			if (sol_ptr[i] == ref_ptr[i])
				continue;
			else {
				if (fabs(sol_ptr[i]) == INFINITY)
					inf_thread++;
				if (fabs(sol_ptr[i]) == 0)
					zero_thread++;

				if (fabs(ref_ptr[i]) > 0 && fabs(sol_ptr[i]) != INFINITY) {
					absolute_diff = fabs(ref_ptr[i] - sol_ptr[i]);
					relative_diff = absolute_diff / fabs(ref_ptr[i]);
					euclidean_dist += absolute_diff * absolute_diff;
				}

				if ((absolute_diff - absolute_err) > 0)
					absolute_err = absolute_diff;
				if ((relative_diff - relative_err) > 0)
					relative_err = relative_diff;
			}
		}
		/* write local values into the heap */
		diffs[0][t_id] = absolute_err;
		diffs[1][t_id] = relative_err;
		diffs[2][t_id] = euclidean_dist;
		zeroes[0][t_id] = zero_thread;
		zeroes[1][t_id] = inf_thread;
	}
	/*	Initialize the global variables */
	double absolute_err = diffs[0][0], relative_err = diffs[1][0],
	       euclidean_dist = diffs[2][0]; // global error buffers
	int zero_entries = zeroes[0][0],
	    inf_entries = zeroes[1][0]; // global zeroes buffers
	for (int i = 1; i < max_id; i++) {
		if (diffs[0][i] > absolute_err)
			absolute_err = diffs[0][i];
		if (diffs[1][i] > relative_err)
			relative_err = diffs[1][i];
		euclidean_dist += diffs[2][i];
		zero_entries += zeroes[0][i];
		inf_entries += zeroes[1][i];
	}
	int til_min = INT_MAX,
	    til_max =
	        INT_MIN; // variables to collect the information about number of tiles and their sizes
	for (int til_idx = tn; til_idx > 0; til_idx--) // go over tiles of one slice
	{
		const int til_sz =
		    ndc_[til_idx] - ndc_[til_idx - 1]; // calculate tile size
		if (til_sz > til_max) // if tile size is bigger than the maximal one
			til_max = til_sz; // overwrite the maximal value
		if (til_sz < til_min) // if tile size is smaller than the minimal one
			til_min = til_sz; // overwrite the minimal value
	}
	printf(
	    "tiling:\n"
	    "\tnumber of tiles:\t%d\n"
	    "\tmin size:\t%d\n"
	    "\tavg size:\t%.2f\n"
	    "\tmax size:\t%d\n"
	    "Result difference:\n"
	    "\tabsolute: \t%.2e\n"
	    "\trelative: \t%.2e\n"
	    "\teuclidean:\t%.2e\n",
	    tn,
	    til_min,
	    ((double) m_rows) / tn,
	    til_max,
	    absolute_err,
	    relative_err,
	    sqrt(euclidean_dist / m_rows));
	if (zero_entries > 0)
		printf("solutions with zeros = %d\n", zero_entries);
	if (inf_entries > 0)
		printf("solutions with inf = %d\n", inf_entries);
}

/**
 * @brief Print of comparison summary
 * @param[in] tn				: number of tiles in each layer
 * @param[in] ndc_				: array of indices for slicing array
 * @param[in] ref_ptr			: reference array
 * @param[in] sol_ptr			: calculated solution array
 */
void cpumultiplyDsummary(const int tn,
                         const int* const ndc_,
                         const double* const ref_ptr,
                         const double* const sol_ptr) {
	compare_summary<double>(tn, ndc_, ref_ptr, sol_ptr);
}

/**
 * @brief Print of comparison summary
 * @param[in] tn				: number of tiles in each layer
 * @param[in] ndc_				: array of indices for slicing array
 * @param[in] ref_ptr			: reference array
 * @param[in] sol_ptr			: calculated solution array
 */
void cpumultiplySsummary(const int tn,
                         const int* const ndc_,
                         const float* const ref_ptr,
                         const float* const sol_ptr) {
	compare_summary<float>(tn, ndc_, ref_ptr, sol_ptr);
}

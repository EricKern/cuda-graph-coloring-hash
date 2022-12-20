/** 
 * \file cpumultiply_loadMTX.cpp
 * @brief File with matrix loading functions
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <sstream> //! stringstream
#include <vector>

/**
 * @brief Read in Matrix in mtx format to internal data structures
 * @param[in] inputMat	: path to the file with matrix information(<name>.mtx)
 * @param[out] row_dptr	: pointer to matrix row array (can be set to NULL, if no output required)
 * @param[out] col_dptr	: pointer to matrix column array (can be set to NULL, if no output required)
 * @param[out] val_dptr	: pointer to matrix value array (can be set to NULL, if no output required)
 * @return				: number of rows in read-in matrix
 */
template <class prcsn>
int load_mtx(const char* const inputMat,
             int** const row_dptr,
             int** const col_dptr,
             prcsn** const val_dptr) {
	assert(
	    NULL !=
	    inputMat); // path to the file with matrix information(<name>.mtx) or matrix generation dimensions(XxYxZ) must be set
	int len = strlen(inputMat);
	const std::string file_(inputMat);
	if (len < 5)
		throw std::runtime_error("invalid input file '" + file_ + "'");
	if (0 != std::strcmp(inputMat + len - 4, ".mtx"))
		throw std::runtime_error("invalid input file ending in path '" + file_ +
		                         "'");
	std::vector<std::vector<int>> col_inter;
	std::vector<std::vector<prcsn>> val_inter;
	bool
	    m_sizes = true,
	    m_header = true,
	    m_symm =
	        true; // flag which signals if matrix interiors were found in the file
	int row_sz = -1, col_sz = -1,
	    val_sz = -1; // initial values for matrix interior variables
	std::ifstream ifs_(file_); // create input file stream
	std::stringstream sstr; // string stream for interpretation of read-in data
	std::string line_; // for reading linewise from a file
	if (!ifs_.is_open()) // open stream
		throw std::runtime_error("cannot open file '" + file_ +
		                         "' for reading");
	while (std::getline(ifs_, line_)) // read in content of file line by line
	{
		sstr.clear(); // clear the stream from previous characters
		sstr.str(line_); // push the string into the stream
		if (!sstr) // take the integer with the partition number out from the stream
		{
			ifs_.close(); // close input file stream
			throw std::runtime_error(file_ + " : cannot read from the stream");
		}
		if (sstr.peek() ==
		    '%') // if the first character is percentage sign, it is comment line and should be omitted
		{
			if (m_header) {
				m_header = false;
				std::string banner, structure, coordinate, type, storage;
				const std::string general("general");
				sstr >> banner >> structure >> coordinate >> type >> storage;
				if (general == storage)
					m_symm = false;
			}
			continue; // go to the next line
		}

		if (m_sizes) // at the beginning of the file is a line with information about number of row and columns as well as non-zero entries in the matrix
		{
			sstr >> row_sz >> col_sz >>
			    val_sz; // read in the information about matrix interiors
			if (row_sz > 0 && col_sz > 0 &&
			    val_sz > 0) // only if all of them have plausible values
			{
				m_sizes =
				    false; // set the flag that the matrix information was read in
				col_inter.resize(row_sz);
				val_inter.resize(row_sz); // resize the vectors appropriately
			}
			else {
				ifs_.close(); // close input file stream
				throw std::runtime_error(
				    "invalid dimesions of matrix from file '" + file_ + "'");
			}
		}
		else // non zero entry of matrix is described by three values => row, column and value
		{
			int row = -1,
			    col =
			        -1; // initialize indices with values which allow later check up
			prcsn val;
			sstr >> row >> col >>
			    val; // read in the next matrix entry from the stream

			if ((row--) > 0 && (col--) > 0 && row < row_sz &&
			    col <
			        col_sz) // only if both indices have plausible values convert them to zero based indices
			{
				col_inter.at(row).push_back(col); // write column
				val_inter.at(row).push_back(val); // write value
				if (m_symm && row != col) // mirroring of lower triangular
				{
					col_inter.at(col).push_back(row); // write column
					val_inter.at(col).push_back(val); // write value
				}
			}
		}
	}
	ifs_.close(); // close input file stream
	if (m_symm)
		val_sz = val_sz * 2 - row_sz;

	int* const __restrict__ out_row_ptr = new int[row_sz + 1];
	int* const __restrict__ out_col_ptr = new int[val_sz];
	prcsn* const __restrict__ out_val_ptr = new prcsn[val_sz];
	int row_ind = 0; // first index in row is always zero
	for (size_t idx_ = 0; idx_ < row_sz; idx_++) // go over the rows
	{
		out_row_ptr[idx_] = row_ind; // write row index
		std::copy(
		    col_inter.at(idx_).begin(),
		    col_inter.at(idx_).end(),
		    out_col_ptr +
		        row_ind); // copy the entries of column indices of that row
		std::copy(
		    val_inter.at(idx_).begin(),
		    val_inter.at(idx_).end(),
		    out_val_ptr + row_ind); // copy the entries of values of that row
		row_ind +=
		    col_inter.at(idx_)
		        .size(); // increment row index by the number of written non zero entries
	}
	out_row_ptr[row_sz] =
	    row_ind; // last entry contains the number of all non zero entries in matrix
	if (NULL == row_dptr)
		delete out_row_ptr; // delete created row data
	else
		*row_dptr =
		    out_row_ptr; // redirect output row pointer to output memory location
	if (NULL == col_dptr)
		delete out_col_ptr; // delete created column data
	else
		*col_dptr =
		    out_col_ptr; // redirect output column pointer to output memory location
	if (NULL == val_dptr)
		delete out_val_ptr; // delete created value data
	else
		*val_dptr =
		    out_val_ptr; // redirect output value pointer to output memory location
	return row_sz;
}

/**
 * @brief Read in Matrix in mtx format to internal data structures
 * @param[in] inputMat	: path to the file with matrix information(<name>.mtx) or cuboid dimensions(XxYxZ) for matrix generation
 * @param[out] row_dptr	: pointer to matrix row array
 * @param[out] col_dptr	: pointer to matrix column array
 * @param[out] val_dptr	: pointer to matrix value array
 * @return				: number of rows in read-in matrix
 */
int cpumultiplyDloadMTX(const char* const inputMat,
                        int** const row_dptr,
                        int** const col_dptr,
                        double** const val_dptr) {
	return load_mtx<double>(inputMat, row_dptr, col_dptr, val_dptr);
}

/**
 * @brief Read in Matrix in mtx format to internal data structures
 * @param[in] inputMat	: path to the file with matrix information(<name>.mtx) or cuboid dimensions(XxYxZ) for matrix generation
 * @param[out] row_dptr	: pointer to matrix row array
 * @param[out] col_dptr	: pointer to matrix column array
 * @param[out] val_dptr	: pointer to matrix value array
 * @return				: number of rows in read-in matrix
 */
int cpumultiplySloadMTX(const char* const inputMat,
                        int** const row_dptr,
                        int** const col_dptr,
                        float** const val_dptr) {
	return load_mtx<float>(inputMat, row_dptr, col_dptr, val_dptr);
}

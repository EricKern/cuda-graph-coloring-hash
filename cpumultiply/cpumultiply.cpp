/** 
 * \file cpumultiply.cpp
 * @brief Library main file
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <omp.h>	//! include OpenMP pragmas
#include <climits>	//! INT_MAX, INT_MIN
#include <cassert> //! assert
#include <string> //! needed for string
#include <fstream> //! ifstream
#include <cstring>	//! memset, NULL
#include <cmath>	//! fabs, sqrt, INFINITY
#include <chrono>	//! needed for std::chrono
using Clock = std::chrono::high_resolution_clock;
using DSeconds = std::chrono::duration<double, std::ratio<1>>;

#include "cpumultiply_permutation.cpp" //! File with interfaces for permutation functions
#include "cpumultiply_loadMTX.cpp" //! File with matrix loading functions
#include "cpumultiply_printing.cpp" //! File with interfaces for printing function
#include "cpumultiply_spmv.cpp" //! File with interfaces for multiplication function

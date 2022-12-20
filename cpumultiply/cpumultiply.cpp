/** 
 * \file cpumultiply.cpp
 * @brief Library main file
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

#include <omp.h> //! include OpenMP pragmas
#include <cassert> //! assert
#include <chrono> //! needed for std::chrono
#include <climits> //! INT_MAX, INT_MIN
#include <cmath> //! fabs, sqrt, INFINITY
#include <cstring> //! memset, NULL
#include <fstream> //! ifstream
#include <string> //! needed for string
using Clock = std::chrono::high_resolution_clock;
using DSeconds = std::chrono::duration<double, std::ratio<1>>;

#include "cpumultiply_loadMTX.cpp" //! File with matrix loading functions
#include "cpumultiply_permutation.cpp" //! File with interfaces for permutation functions
#include "cpumultiply_printing.cpp" //! File with interfaces for printing function
#include "cpumultiply_spmv.cpp" //! File with interfaces for multiplication function

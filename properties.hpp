/** 
 * \file properties.hpp
 * @brief Header file with definitions of input argument properties
 * 
 * Created by Robert Gutmann for student research project.
 * In case of problems, contact me: r.gutmann@stud.uni-heidelberg.de.
 **/

struct tiling_properties
{
	uint8_t analyse : 1;	// kind of slicing to be performed: 0=recursive , 1=analyse
	uint8_t compress : 1;	// reduction method for number of slices : 0=merging , 1=shifting
	uint8_t rinf : 1;	// additional output of redundant connections in percent over all bars (avg, min, max) : 0=diactivated , 1=printing
	uint8_t tinf : 1;	// extentional information about tiling process for each slice is printed out : 0=diactivated , 1=printing
	uint8_t sinf : 1;	// information about tiles within each slice : 0=diactivated , 1=(printing of slice number, size of slice, average, minimal and maximal tile in slice)
	uint8_t linf : 1;	// information about layers after tiling process : 0=diactivated , 1=(printing of number of layers, sizes of layers)
	uint8_t fine : 1;	// merging of consecutive slices : 0=coarsening , 1=fine
	uint8_t metis : 1;	// method of tiling procedure : 0=sectioning , 1=metising
    enum TILING {
		METIS = 128,
		FINE = 64,
		PRINT_LINF = 32,
		PRINT_SINF = 16,
		PRINT_TINF = 8,
		PRINT_RINF = 4,
		SHIFTING = 2,
		ANALYSE = 1,
		DEFAULT = 0
    };
};

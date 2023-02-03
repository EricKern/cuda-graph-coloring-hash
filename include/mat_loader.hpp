#pragma once
#include "cpumultiply.hpp"  //! header file for tiling

namespace apa22_coloring {
// Singleton class so we load matrix only once
// consequentially we can only have only one matrix loaded at a time
class MatLoader {
 public:
  static MatLoader& getInstance(const char* path = "/mnt/matrix_store/MM/Bodendiek/CurlCurl_4/CurlCurl_4.mtx") {
    static MatLoader Instance(path);

    return Instance;
  }

  MatLoader(const MatLoader&) = delete;
  MatLoader& operator=(const MatLoader&) = delete;
  
  const char* path;
  int* row_ptr;
  int* col_ptr;
  double* val_ptr;
  int m_rows;

 private:
  MatLoader(const char* path) : path(path) {
    m_rows = cpumultiplyDloadMTX(path, &row_ptr, &col_ptr, &val_ptr);
  }
  ~MatLoader() {
    delete[] row_ptr;
    delete[] col_ptr;
    delete[] val_ptr;
  }
};

} // end apa22_coloring
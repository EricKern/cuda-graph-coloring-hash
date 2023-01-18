#pragma once

namespace def {
static constexpr const char* Mat0 =
    "/home/eric/Documents/graph-coloring/cage3.mtx";
static constexpr const char* Mat1 =
    "/home/eric/Documents/graph-coloring/cage4.mtx";
static constexpr const char* Mat2 =
    "/home/eric/Documents/graph-coloring/CurlCurl_0.mtx";
static constexpr const char* Mat3 =
    "/home/eric/Documents/graph-coloring/CurlCurl_4.mtx";

const char* choseMat(int mat_nr) {
  const char* return_mat;
  switch (mat_nr) {
    case 0:
      return_mat = def::Mat0;
      break;
    case 1:
      return_mat = def::Mat1;
      break;
    case 2:
      return_mat = def::Mat2;
      break;
    case 3:
      return_mat = def::Mat3;
      break;

    default:
      return_mat = def::Mat2;
  }
  return return_mat;
}
}  // namespace def
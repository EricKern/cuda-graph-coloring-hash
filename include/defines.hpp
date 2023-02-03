#pragma once
#include <unordered_map>
namespace def {
static constexpr const char* Mat0 =
    "/home/eric/Documents/graph-coloring/cage3.mtx";
static constexpr const char* Mat1 =
    "/home/eric/Documents/graph-coloring/cage4.mtx";
static constexpr const char* Mat2 =
    "/home/eric/Documents/graph-coloring/CurlCurl_0.mtx";
static constexpr const char* Mat3 =
    "/home/eric/Documents/graph-coloring/CurlCurl_4.mtx";

static constexpr const char* CurlCurl_0 =
    "/mnt/matrix_store/MM/Bodendiek/CurlCurl_0/CurlCurl_0.mtx";

static constexpr const char* CurlCurl_4 =
    "/mnt/matrix_store/MM/Bodendiek/CurlCurl_4/CurlCurl_4.mtx";

static constexpr const char* atmosmodd =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodd/admosmodd.mtx";

static constexpr const char* atmosmodj =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodj/admosmodj.mtx";

static constexpr const char* atmosmodl =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodl/admosmodl.mtx";

static constexpr const char* atmosmodm =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodm/admosmodm.mtx";

static constexpr const char* af_shell3 =
    "/mnt/matrix_store/MM/Schenk_AFE/af_shell3/af_shell3.mtx";

static constexpr const char* af_shell4 =
    "/mnt/matrix_store/MM/Schenk_AFE/af_shell4/af_shell4.mtx";

static constexpr const char* af_shell7 =
    "/mnt/matrix_store/MM/Schenk_AFE/af_shell7/af_shell7.mtx";

static constexpr const char* af_shell8 =
    "/mnt/matrix_store/MM/Schenk_AFE/af_shell8/af_shell8.mtx";

static constexpr const char* Cube_Coup_dt0 =
    "/mnt/matrix_store/MM/Janna/Cube_Coup_dt0/Cube_Coup_dt0.mtx";

static constexpr const char* ML_Geer =
    "/mnt/matrix_store/MM/Janna/ML_Geer/ML_Geer.mtx";


std::unordered_map<std::string, const char*> map = {
    {"CurlCurl_0", CurlCurl_0},
    {"CurlCurl_4", CurlCurl_4},
    {"atmosmodd", atmosmodd},
    {"atmosmodd", atmosmodj},
    {"atmosmodl", atmosmodl},
    {"atmosmodm", atmosmodm},
    {"af_shell3", af_shell3},
    {"af_shell4", af_shell4},
    {"af_shell7", af_shell7},
    {"af_shell8", af_shell8},
    {"Cube_Coup_dt0", Cube_Coup_dt0},
    {"ML_Geer", ML_Geer}
};
// const char* choseMat(const char* mat_nr) {
//   const char* return_mat;
//   switch (mat_nr) {
//     case strcmp(:
//       return_mat = def::Mat0;
//       break;
//     case 1:
//       return_mat = def::Mat1;
//       break;
//     case 2:
//       return_mat = def::Mat2;
//       break;
//     case 3:
//       return_mat = def::Mat3;
//       break;

//     default:
//       return_mat = def::Mat2;
//   }
//   return return_mat;
// }
}  // namespace def
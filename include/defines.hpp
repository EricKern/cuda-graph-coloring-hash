#pragma once
#include <unordered_map>
namespace def {

static constexpr const char* CurlCurl_0 =
    "/mnt/matrix_store/MM/Bodendiek/CurlCurl_0/CurlCurl_0.mtx";

static constexpr const char* CurlCurl_4 =
    "/mnt/matrix_store/MM/Bodendiek/CurlCurl_4/CurlCurl_4.mtx";

static constexpr const char* atmosmodd =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodd/atmosmodd.mtx";

static constexpr const char* atmosmodj =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodj/atmosmodj.mtx";

static constexpr const char* atmosmodl =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodl/atmosmodl.mtx";

static constexpr const char* atmosmodm =
    "/mnt/matrix_store/MM/Bourchtein/atmosmodm/atmosmodm.mtx";

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
    {"atmosmodj", atmosmodj},
    {"atmosmodl", atmosmodl},
    {"atmosmodm", atmosmodm},
    {"af_shell3", af_shell3},
    {"af_shell4", af_shell4},
    {"af_shell7", af_shell7},
    {"af_shell8", af_shell8},
    {"Cube_Coup_dt0", Cube_Coup_dt0},
    {"ML_Geer", ML_Geer}
};

}  // namespace def
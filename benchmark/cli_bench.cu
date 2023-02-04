#include <vector>
#include <nvbench/nvbench.cuh>

#include "cli_parser.hpp"
#include "setup.cuh"
#include "mat_loader.hpp"
#include "defines.hpp"


void remove_custom_clp(int argc, char** argv, std::vector<char*>& args) {
  // remove the two elements "-mat", "/path/to/mat.mtx" from argv
  int j = 0;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "-mat") == 0) {
      i++;
      continue;
    }
    args[j] = argv[i];
    j++;
  }
}

int main(int argc, char** argv) {
  char const *mat_path = NULL;                     //Default value
  chCommandLineGet<char const *>(&mat_path, "mat", argc, argv);

  const char* Mat;
  if(mat_path != NULL){
    std::string mat_path_str(mat_path);
    if (auto search = def::map.find(mat_path_str); search != def::map.end()){
      std::cout << "Found " << search->first << " " << search->second << '\n';
      Mat = search->second;
    }
    else {
        std::cout << "Using unknown user provided matrix path\n";
        Mat = mat_path; // assume user input is unkown valid path
    }
  }
  else{
    NVBENCH_MAIN_BODY(argc, argv);
    return 0;
  }

  // Create a new argument array without matrix path to pass to NVBench.
  std::vector<char*> args(argc - 2);
  remove_custom_clp(argc, argv, args);

  // build singleton from cl parameter
  apa22_coloring::MatLoader::getInstance(Mat);

  NVBENCH_MAIN_BODY(argc - 2, args.data());
  return 0;
}
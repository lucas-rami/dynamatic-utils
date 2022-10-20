set(CMAKE_BUILD_TYPE Release)

llvm_map_components_to_libnames(llvm_libs core)
message(STATUS "using libraries ${llvm_libs}")

add_llvm_library(LLVMStatisticsPass MODULE LLVMStatisticsPass.cpp ../IRStats.cpp)

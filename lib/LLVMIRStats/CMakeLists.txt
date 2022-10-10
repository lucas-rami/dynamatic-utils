set(CMAKE_BUILD_TYPE Release)

llvm_map_components_to_libnames(llvm_libs core)
message(STATUS "using libraries ${llvm_libs}")

add_llvm_library(LLVMIrStats MODULE LLVMIRStats.cpp ../IRStats.cpp)
# target_link_libraries(LLVMIrStats)
# target_compile_features(LLVMIrStats PRIVATE cxx_range_for cxx_auto_type)
# SET(CMAKE_CXX_FLAGS "-fno-rtti")

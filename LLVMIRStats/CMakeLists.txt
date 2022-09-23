set(CMAKE_BUILD_TYPE Release)

add_library(LLVMIrStats MODULE LLVMIRStats.cpp)

target_compile_features(LLVMIrStats PRIVATE cxx_range_for cxx_auto_type)
SET(CMAKE_CXX_FLAGS "-fno-rtti")

set(CMAKE_BUILD_TYPE Release)

add_mlir_library(MLIRStatisticsPass
  MLIRStatisticsPass.cpp
  ../IRStats.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
target_compile_options(MLIRStatisticsPass PRIVATE --std=c++20)

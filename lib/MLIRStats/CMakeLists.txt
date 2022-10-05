set(LLVM_OPTIONAL_SOURCES
  MLIRStatsAnalysis.cpp
  MLIRStatsPass.cpp
)

add_mlir_library(MLIRStatsAnalysis
  MLIRStatsAnalysis.cpp
  ../IRStats.cpp
  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(MLIRStatsPass
  MLIRStatsPass.cpp
  ../IRStats.cpp
  LINK_LIBS PUBLIC
  MLIRPass
  MLIRStatsAnalysis
)

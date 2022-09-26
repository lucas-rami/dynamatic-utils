add_mlir_library(MLIRStatsPass
  MLIRStatsPass.cpp
  LINK_LIBS PUBLIC
  MLIRPass
  )

add_subdirectory(tools-opt)

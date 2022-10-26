add_mlir_library(MLIRStatisticsPass
  MLIRStatisticsPass.cpp
  ../IRStats.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)

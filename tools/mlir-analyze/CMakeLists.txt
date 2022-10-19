get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRStatsPass
  MLIRStatsAnalysis
  MLIRGlobalsPass
)

add_llvm_executable(mlir-analyze mlir-analyze.cpp)

llvm_update_compile_flags(mlir-analyze)
target_link_libraries(mlir-analyze PRIVATE ${LIBS})

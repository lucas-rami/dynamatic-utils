get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRStatsPass
  MLIRStatsAnalysis
) 

add_llvm_executable(tools-opt tools-opt.cpp)
 
llvm_update_compile_flags(tools-opt)
target_link_libraries(tools-opt PRIVATE ${LIBS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRStatsPass
) 

add_llvm_executable(tools-opt tools-opt.cpp)

# install(TARGETS polygeist-opt
# EXPORT PolygeistTargets
# RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
# COMPONENT polygeist-opt)

llvm_update_compile_flags(tools-opt)
target_link_libraries(tools-opt PRIVATE ${LIBS})

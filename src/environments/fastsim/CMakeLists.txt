add_library(env_fastsim INTERFACE)
target_include_directories(env_fastsim INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(env_fastsim
  INTERFACE libfastsim
)

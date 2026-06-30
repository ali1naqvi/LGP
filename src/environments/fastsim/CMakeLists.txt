add_library(env_fastsim
  FastSimEnvironment.h
)
target_include_directories(env_fastsim PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(env_fastsim
  PUBLIC libfastsim
)
# Add cpp subdirectories
add_subdirectory(engine)
if(TPG_BUILD_EXPERIMENTS)
    add_subdirectory(experiments)
endif()
add_subdirectory(environments/classic_control)
add_subdirectory(environments/fastsim)
add_subdirectory(environments/xpredprey)
add_subdirectory(logging)

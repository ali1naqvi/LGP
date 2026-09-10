cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(xpredprey VERSION 1.0.0 LANGUAGES CXX)
    include(CTest)
endif()

add_library(xpredprey_simulator
    XPredPrey.cpp
    XPredPreyTask.cpp
    RobotEnv.cpp
    Utilities.cpp
)
target_include_directories(xpredprey_simulator PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_compile_features(xpredprey_simulator PUBLIC cxx_std_17)
target_compile_definitions(xpredprey_simulator
    PUBLIC XPREDPREY_DEFAULT_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)

if(MSVC)
    target_compile_options(xpredprey_simulator PRIVATE /W4)
else()
    target_compile_options(xpredprey_simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(xpredprey main.cpp)
target_link_libraries(xpredprey PRIVATE xpredprey_simulator)

set(XPREDPREY_ASSETS
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/marxbot-cylinder.sample"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/marxbot-scylinder.sample"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/marxbot-wall.sample"
)

add_custom_command(
    TARGET xpredprey POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:xpredprey>/assets"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${XPREDPREY_ASSETS}
            "$<TARGET_FILE_DIR:xpredprey>/assets"
    VERBATIM
)

if(BUILD_TESTING)
    add_executable(xpredprey_schedule_test XPredPreyScheduleTest.cpp)
    target_compile_features(xpredprey_schedule_test PRIVATE cxx_std_17)
    add_test(NAME xpredprey_schedule COMMAND xpredprey_schedule_test)
    add_test(NAME xpredprey_smoke COMMAND xpredprey --steps 5)
    add_executable(xpredprey_task_test XPredPreyTaskTest.cpp)
    target_link_libraries(xpredprey_task_test PRIVATE xpredprey_simulator)
    add_test(NAME xpredprey_task_rewards COMMAND xpredprey_task_test)
    set_tests_properties(xpredprey_smoke xpredprey_task_rewards xpredprey_schedule
                         PROPERTIES TIMEOUT 30)
endif()

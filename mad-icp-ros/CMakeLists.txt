cmake_minimum_required(VERSION 3.8)
project(mad-icp-ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)


find_package(mad_icp QUIET)

if (NOT mad_icp_FOUND)
    message(STATUS "mad_icp not found. Fetching from Git...")

    include(FetchContent)
    FetchContent_Declare(
        mad_icp
        GIT_REPOSITORY https://github.com/rvp-group/mad-icp.git
        GIT_TAG        main
    )
    FetchContent_MakeAvailable(mad_icp)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    add_subdirectory(${mad_icp_SOURCE_DIR}/mad_icp ${mad_icp_BINARY_DIR}/mad_icp_build)

    get_property(madicp_targets DIRECTORY ${mad_icp_SOURCE_DIR}/mad_icp PROPERTY BUILDSYSTEM_TARGETS)
    message(STATUS "Targets in mad-icp/mad_icp:")
    foreach(t ${madicp_targets})
      message(STATUS "  - ${t}")
    endforeach()
endif()

add_executable(main src/main.cpp)

message("AAAAAAAAAAAAAAAAAA")
message("${mad_icp_SOURCE_DIR}")

target_include_directories(main PUBLIC
  ${mad_icp_SOURCE_DIR}/mad_icp/src
)

target_link_libraries(main
  PUBLIC
  odometry
)

ament_package()

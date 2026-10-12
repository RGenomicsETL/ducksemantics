cmake_minimum_required(VERSION 3.16)
project(ducksemantics VERSION 0.0.1 LANGUAGES C)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The distribution currently verifies Linux POSIX hosts only")
endif()

# Native Make and distribution CMake compile the same canonical translation units.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/sources.list" DUCKSEMANTICS_SOURCES)
add_library(ducksemantics SHARED ${DUCKSEMANTICS_SOURCES})
set_target_properties(ducksemantics PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS NO
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES)
target_include_directories(ducksemantics PRIVATE duckdb_capi)
target_compile_definitions(ducksemantics PRIVATE
  _POSIX_C_SOURCE=200809L
  DUCKDB_EXTENSION_NAME=ducksemantics
  DUCKDB_EXTENSION_API_VERSION_MAJOR=1
  DUCKDB_EXTENSION_API_VERSION_MINOR=2
  DUCKDB_EXTENSION_API_VERSION_PATCH=0)
target_compile_options(ducksemantics PRIVATE
  -ffp-contract=off -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
find_package(Threads REQUIRED)
target_link_libraries(ducksemantics PRIVATE m Threads::Threads)

# extension-ci-tools appends the stable-C-ABI footer to this shared library.

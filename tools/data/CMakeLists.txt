cmake_minimum_required(VERSION 3.24)
project(kanachan_data_tools LANGUAGES CXX)

# Standalone CPU-only annotation build, including Windows/MSVC. The training
# simulator and its Python/Boost.Python dependencies are unnecessary here.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  set(Protobuf_USE_STATIC_LIBS ON)
endif()
find_package(Protobuf 3.20 REQUIRED)
find_path(KANACHAN_BOOST_INCLUDE_DIR boost/exception/exception.hpp REQUIRED)
get_filename_component(KANACHAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(annotate
  "${KANACHAN_ROOT}/src/annotation/main.cpp"
  "${KANACHAN_ROOT}/src/annotation/annotation.cpp"
  "${KANACHAN_ROOT}/src/annotation/round_progress.cpp"
  "${KANACHAN_ROOT}/src/annotation/player_state.cpp"
  "${KANACHAN_ROOT}/src/annotation/utility.cpp"
  "${KANACHAN_ROOT}/src/common/mahjongsoul.pb.cc"
  "${KANACHAN_ROOT}/src/common/assert.cpp"
  "${KANACHAN_ROOT}/src/common/throw.cpp"
  "${KANACHAN_ROOT}/src/common/type_name.cpp")
target_include_directories(annotate PRIVATE
  "${KANACHAN_ROOT}/src" "${KANACHAN_BOOST_INCLUDE_DIR}")
target_link_libraries(annotate PRIVATE protobuf::libprotobuf)
if(MSVC)
  target_compile_options(annotate PRIVATE /utf-8 /bigobj)
  target_compile_definitions(annotate PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  # The checked-in protoc-3.20 source predates MSVC's C++20 constinit
  # implementation. Compile only that generated translation unit as C++17;
  # annotation code still needs C++20 ranges.
  set_source_files_properties("${KANACHAN_ROOT}/src/common/mahjongsoul.pb.cc"
    PROPERTIES COMPILE_OPTIONS /std:c++17)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(annotate PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

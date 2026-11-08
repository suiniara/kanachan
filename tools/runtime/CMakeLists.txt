cmake_minimum_required(VERSION 3.24)
project(kanachan_native_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# An isolated Python extension build, including MSVC, without Linux-only
# system libraries or a second shared-library boundary around simulation.
get_filename_component(KANACHAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_path(KANACHAN_BOOST_ROOT boost/python.hpp REQUIRED)
find_path(KANACHAN_NYANTEN_ROOT nyanten/replacement_number.hpp REQUIRED)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../features" "candidate-features")
if(NOT KANACHAN_PYTHON_INCLUDE_DIR OR NOT KANACHAN_PYTHON_LIBRARY)
  message(FATAL_ERROR "Set KANACHAN_PYTHON_INCLUDE_DIR and KANACHAN_PYTHON_LIBRARY for the target interpreter")
endif()
file(GLOB_RECURSE BOOST_PYTHON_SOURCES CONFIGURE_DEPENDS "${KANACHAN_BOOST_ROOT}/libs/python/src/*.cpp")
list(FILTER BOOST_PYTHON_SOURCES EXCLUDE REGEX "/numpy/")
if(NOT BOOST_PYTHON_SOURCES)
  message(FATAL_ERROR "Extract Boost.Python sources as well as Boost headers")
endif()
add_library(boost_python_static STATIC ${BOOST_PYTHON_SOURCES})
target_compile_definitions(boost_python_static PUBLIC BOOST_PYTHON_STATIC_LIB BOOST_ALL_NO_LIB PRIVATE BOOST_PYTHON_SOURCE)
target_include_directories(boost_python_static PUBLIC "${KANACHAN_BOOST_ROOT}" "${KANACHAN_PYTHON_INCLUDE_DIR}")
set_target_properties(boost_python_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB SIMULATION_SOURCES CONFIGURE_DEPENDS "${KANACHAN_ROOT}/src/simulation/*.cpp")
add_library(_simulation MODULE ${SIMULATION_SOURCES}
  "${KANACHAN_ROOT}/src/common/thread.cpp"
  "${KANACHAN_ROOT}/src/common/assert.cpp"
  "${KANACHAN_ROOT}/src/common/throw.cpp"
  "${KANACHAN_ROOT}/src/common/type_name.cpp")
target_include_directories(_simulation PRIVATE "${KANACHAN_ROOT}/src" "${KANACHAN_NYANTEN_ROOT}")
target_link_libraries(_simulation PRIVATE boost_python_static "${KANACHAN_PYTHON_LIBRARY}")
target_compile_definitions(_simulation PRIVATE KANACHAN_ENABLE_ASSERT)
set_target_properties(_simulation PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${KANACHAN_ROOT}/kanachan/simulation" RUNTIME_OUTPUT_DIRECTORY "${KANACHAN_ROOT}/kanachan/simulation")
if(MSVC)
  foreach(target boost_python_static _simulation)
    target_compile_options(${target} PRIVATE /utf-8 /bigobj)
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endforeach()
  set_target_properties(_simulation PROPERTIES SUFFIX ".pyd")
else()
  find_package(Threads REQUIRED)
  target_link_libraries(_simulation PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

option(KANACHAN_BUILD_ANNOTATION "Build the Linux record preparation tools" OFF)
if(KANACHAN_BUILD_ANNOTATION)
  if(WIN32)
    message(FATAL_ERROR "Use the existing Windows annotation build on Windows")
  endif()
  find_package(Protobuf REQUIRED)
  set(GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
  file(MAKE_DIRECTORY "${GENERATED_ROOT}/common")
  add_custom_command(
    OUTPUT "${GENERATED_ROOT}/common/mahjongsoul.pb.cc" "${GENERATED_ROOT}/common/mahjongsoul.pb.h"
    COMMAND protobuf::protoc
      "--proto_path=${KANACHAN_ROOT}/src" "--cpp_out=${GENERATED_ROOT}"
      "${KANACHAN_ROOT}/src/common/mahjongsoul.proto"
    DEPENDS "${KANACHAN_ROOT}/src/common/mahjongsoul.proto"
    VERBATIM)
  add_library(annotation_common STATIC
    "${GENERATED_ROOT}/common/mahjongsoul.pb.cc"
    "${KANACHAN_ROOT}/src/common/thread.cpp"
    "${KANACHAN_ROOT}/src/common/assert.cpp"
    "${KANACHAN_ROOT}/src/common/throw.cpp"
    "${KANACHAN_ROOT}/src/common/type_name.cpp")
  target_include_directories(annotation_common PUBLIC
    "${GENERATED_ROOT}" "${KANACHAN_ROOT}/src" "${KANACHAN_BOOST_ROOT}")
  target_compile_definitions(annotation_common PUBLIC KANACHAN_ENABLE_ASSERT)
  target_link_libraries(annotation_common PUBLIC
    protobuf::libprotobuf Threads::Threads ${CMAKE_DL_LIBS})
  file(GLOB ANNOTATION_SOURCES CONFIGURE_DEPENDS "${KANACHAN_ROOT}/src/annotation/*.cpp")
  add_executable(annotate ${ANNOTATION_SOURCES})
  target_link_libraries(annotate PRIVATE annotation_common)
  add_executable(paishan "${KANACHAN_ROOT}/src/paishan/main.cpp")
  target_link_libraries(paishan PRIVATE annotation_common)
  set_target_properties(annotate paishan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${KANACHAN_ROOT}/bin")
endif()

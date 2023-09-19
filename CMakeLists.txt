cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(mpm)

find_package(CUDA REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if (CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "-ccbin /usr/bin/gcc;")
  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${ARCH_FLAGS})

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
  endif (UNIX)

  # add debugging to CUDA NVCC flags. For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
  
  include_directories(${CMAKE_SOURCE_DIR}/src)
  add_subdirectory(src)

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()

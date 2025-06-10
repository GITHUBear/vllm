cmake_minimum_required(VERSION 3.26)
project(dca_rotary_embedding_test)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "python" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    /data/shanhaikang.shk/vllm/csrc/
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS " build ${CU_FILE} in DEBUG MODE")
    # target_compile_options(${FILE_NAME} PRIVATE
    #     $<$<COMPILE_LANGUAGE:CUDA>:-G -O0>
    #     $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
    # )
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS " build ${CU_FILE} in RELEASE MODE")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    # target_compile_options(${FILE_NAME} PRIVATE
    #     $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    #     $<$<COMPILE_LANGUAGE:CXX>:-O3>
    # )
endif()

add_executable(
  test_pos_encoding_kernels
  ./test_dca_rotary_embedding.cu
  ../pos_encoding_kernels.cu
)

target_include_directories(test_pos_encoding_kernels PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(test_pos_encoding_kernels
    ${TORCH_LIBRARIES}
)


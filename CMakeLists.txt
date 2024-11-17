cmake_minimum_required(VERSION 3.11)
project(Lite_Mono LANGUAGES CXX CUDA)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(${TensorRT_SAMPLE_DIR}/common)

include_directories(${SRC_DIR}/engine)
include_directories(${SRC_DIR}/model)
include_directories(${SRC_DIR}/process)
include_directories(${SRC_DIR}/server)
include_directories(${SRC_DIR}/threadpool)
include_directories(${SRC_DIR}/utils)

link_directories(/usr/local/cuda/lib64/)


# 添加每个模块的源文件
set(SOURCES_MAIN
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/utils/utils.cpp
    ${SRC_DIR}/engine/engine.cu
    ${SRC_DIR}/server/server.cpp
    ${SRC_DIR}/model/depth_net.cpp
    ${SRC_DIR}/process/preprocess.cu
    ${SRC_DIR}/threadpool/threadpool.cpp
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)

# 添加每个模块的源文件
set(SOURCES_BUILD
    ${SRC_DIR}/build.cu
    ${SRC_DIR}/process/preprocess.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)


# 添加可执行文件
add_executable(main 
    ${SOURCES_MAIN}
)
target_link_libraries(main 
    -lpthread
    cuda 
    cudart 
    gflags
    opencv_dnn
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES} 
)
set_target_properties(main PROPERTIES 
    CUDA_ARCHITECTURES "61"
)


add_executable(build
    ${SOURCES_BUILD}
)
target_include_directories(build 
    PRIVATE 
    ${TensorRT_INCLUDE_DIRS} 
    ${TensorRT_SAMPLE_DIR}/common
)
target_link_libraries(build
    PRIVATE
    -Wl,--no-as-needed,--gc-sections,--unresolved-symbols=ignore-in-shared-libs
    gflags
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    )
set_target_properties(build PROPERTIES 
    CUDA_ARCHITECTURES "61"
)
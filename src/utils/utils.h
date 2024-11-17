#pragma once

#include <map>
#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include <string>
#include <memory>
#include <fstream>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <buffers.h>
#include <NvInfer.h>

#include <initializer_list>
#include <future>

#include <stdarg.h>
#include <numeric>
#include <sstream>
#include <cstdio>

#include <logger.h>
#include <common.h>
#include <safeCommon.h>
#include <gflags/gflags.h>



class video_tool{
public:
    video_tool()=default;
    ~video_tool(){
        cap.release();
        writer.release();
    }

    void setUp(const char *video_file);
    int width, height, fps;
    cv::VideoCapture cap;
    cv::VideoWriter writer;
};

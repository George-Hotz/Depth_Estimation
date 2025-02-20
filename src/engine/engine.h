#pragma once
#include "utils.h"

class TrtEngine {
public:
    TrtEngine(const std::string &model_path);
    ~TrtEngine() = default;
    void doInference(bool cpu_pre_flag);
    void* getHostBuffer(const char* tensorName);
    void* getDeviceBuffer(const char* tensorName);

private:

    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::shared_ptr<samplesCommon::BufferManager> buffers_;
};
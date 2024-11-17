#pragma once
#include "utils.h"
#include "preprocess.h"
#include "engine.h"


#define High 518
#define Width 518

class DepthNet {
public:
    DepthNet(const std::string &model_path);
    ~DepthNet() = default;
    cv::Mat run(const cv::Mat& img);
    void doInference();
    void preprocess(const cv::Mat& img);
    cv::Mat postprocess(const cv::Mat &img);

private:
    std::shared_ptr<TrtEngine> engine_;
    cv::Rect roi_;
    bool cpu_pre_flag = false;
    const int kInputH = High;
    const int kInputW = Width;
    const int kInputC = 3;
    const int kOutputH = High;
    const int kOutputW = Width;
    const char* kInputTensorName = "input";
    const char* kOutputTensorName = "output";
};
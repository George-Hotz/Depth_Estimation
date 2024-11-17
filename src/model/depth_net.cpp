#include "depth_net.h"

DepthNet::DepthNet(const std::string &model_path)
{
    engine_.reset(new TrtEngine(model_path));
}

cv::Mat DepthNet::run(const cv::Mat &img)
{
    preprocess(img);
    doInference();
    return postprocess(img);
}

void DepthNet::doInference()
{
    engine_->doInference(cpu_pre_flag);
}

void DepthNet::preprocess(const cv::Mat &img)
{
    if(cpu_pre_flag){
        Preprocess_cpu(img, kInputW, kInputH, (void *)engine_->getHostBuffer(kInputTensorName)); 
    }else{
        Preprocess_gpu(img, kInputW, kInputH, (float *)engine_->getDeviceBuffer(kInputTensorName));
    }
}

cv::Mat DepthNet::postprocess(const cv::Mat &img)
{
    int output_size = kOutputH * kOutputW;
    static std::vector<float> output(output_size);

    memcpy(output.data(), engine_->getHostBuffer(kOutputTensorName), output_size * sizeof(float));  // 将engine_中的outputTensorName的数据拷贝到matte中
    cv::Mat mat(kOutputH, kOutputW, CV_32FC1, output.data());                                       // 将matte的数据拷贝到mat中

    float wh_ratio = (float)img.cols / (float)img.rows;
    float WH_ratio = (float)kOutputW / (float)kOutputH;

    int error_limit = 20;    //误差范围

    if(wh_ratio > WH_ratio){
        roi_ = cv::Rect(0, (kOutputH - kOutputW/wh_ratio)/2 + error_limit, kOutputW, kOutputW/wh_ratio - 2*error_limit);
    }else{
        roi_ = cv::Rect((kOutputW - kOutputH*wh_ratio)/2 + error_limit, 0, kOutputH*wh_ratio - 2*error_limit, kOutputH);
    }

    cv::Mat out = mat(roi_);

    double minVal, maxVal;
    cv::minMaxLoc(out, &minVal, &maxVal);
    cv::convertScaleAbs(out, out, 255.0 / maxVal, 0);
    cv::applyColorMap(out, out, cv::COLORMAP_JET);

    cv::resize(out, out, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_LINEAR);

    return out;
}



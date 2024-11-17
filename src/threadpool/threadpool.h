#pragma once
#include "utils.h"
#include "depth_net.h"

class Thread_Pool{
private:
    std::queue<std::pair<int, cv::Mat>> tasks;             // <id, img>     用来存放任务
    std::vector<std::shared_ptr<DepthNet>> models;         // 深度估计模型

    std::map<int, cv::Mat> img_results;                    // <id, img>     用来存放结果（整张图片）
    std::vector<std::thread> threads;                      // 线程池

    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;

    bool if_draw;
    bool stop;

    void worker(int id);

public:
    Thread_Pool(bool draw=true);  // 构造函数
    ~Thread_Pool();               // 析构函数
                
    bool setUp(const char *depth_net_path,                 // 初始化
                int num_threads = 4);     
    bool submitTask(const cv::Mat &img, int id);                 // 提交任务
    bool getImgResult(cv::Mat &img, int id);               // 获取结果（图片）
    void stopAll();                                        // 停止所有线程
};

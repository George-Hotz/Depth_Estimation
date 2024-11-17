#include "server.h"


Depth_Server::Depth_Server()
{
    m_video_tool = new video_tool();
    m_thread_Pool = new Thread_Pool(true);
}


Depth_Server::~Depth_Server()
{
    delete m_video_tool;
    delete m_thread_Pool;
    cuda_preprocess_destroy();
}

void Depth_Server::init(const char *depth_net_path, const char *video_path, int thread_num)
{
    m_thread_num = thread_num;
    m_depth_net_path = depth_net_path;
    m_video_path = video_path;

    m_video_tool->setUp(m_video_path);
    m_thread_Pool->setUp(m_depth_net_path, m_thread_num);
    cuda_preprocess_init(m_video_tool->width * m_video_tool->height);
}


bool stop = false;

// 读取视频帧，提交任务
void Depth_Server::read_stream()
{
    int frame_id = 0;
    cv::Mat img;
    while (true)
    {
        // 读取视频帧
        m_video_tool->cap >> img;
        if (img.empty())
        {
            printf("Video end. \n");
            stop = true;
            break;
        }

        m_thread_Pool->submitTask(img.clone(), frame_id++);
    }
    // 释放资源
    m_video_tool->cap.release();
}


void Depth_Server::get_results()
{
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int frame_id = 0;
    while (true)
    {
        cv::Mat img;
        auto ret = m_thread_Pool->getImgResult(img, frame_id++);

        if (stop && !ret)
        {
            m_thread_Pool->stopAll();
            m_video_tool->writer.release();
            break;
        }

        // 写入视频帧
        m_video_tool->writer << img;

        // 算法2：计算超过 1s 一共处理了多少张图片
        frame_count++;

        auto end = std::chrono::high_resolution_clock::now();
        auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapse > 1000)
        {
            printf("Time:%fms, FPS:%f, Frame Count:%d  \n", elapse, frame_count / (elapse / 1000.0f), frame_count);
            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }
    }
    // 结束所有线程
    m_thread_Pool->stopAll();
    printf("Get results end. \n");
}









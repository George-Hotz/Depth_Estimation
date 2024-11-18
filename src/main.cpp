#include "server.h"

/*
    Author：雪花
    github: https://github.com/George-Hotz
    License: MIT License 
    Copyright (c) 2024 樂
*/

DEFINE_int32(thread_num, 4, "thread_num");
DEFINE_string(video_path, "", "video path");
DEFINE_string(model_path, "", "model path");

int main(int argc, char **argv)
{
    if (argc < 3){
        std::cout <<"[Error]: argc less than 3" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    int thread_num = FLAGS_thread_num;
    std::string video_path = FLAGS_video_path;
    std::string model_path = FLAGS_model_path;

    Depth_Server depth_server;
    depth_server.init(model_path.c_str(), video_path.c_str(), thread_num);
    
    // 读取视频
    std::thread input_thread([&depth_server]() { depth_server.read_stream(); });
    // 启动结果线程
    std::thread output_thread([&depth_server]() { depth_server.get_results(); });

    // 等待线程结束
    input_thread.join();
    output_thread.join();

    return 0;
}



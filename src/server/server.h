#pragma once
 
#include "utils.h"
#include "depth_net.h"
#include "threadpool.h"


class Depth_Server{
public:
    Depth_Server();
    ~Depth_Server();

    void init(const char *depth_net_path, const char *video_path, int thread_num);
    void get_results();
    void read_stream();

private:
    int m_thread_num;
    const char *m_video_path;
    const char * m_depth_net_path;
    video_tool *m_video_tool;
    Thread_Pool *m_thread_Pool;
};


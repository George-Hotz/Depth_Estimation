#include "threadpool.h"


Thread_Pool::Thread_Pool(bool draw)
{ 
    stop = false;
    if_draw = draw;
}


Thread_Pool::~Thread_Pool()
{
    stop = true;
    cv_task.notify_all();
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}


bool Thread_Pool::setUp(const char *depth_net_path, int num_threads)
{
    
    for (size_t i = 0; i < num_threads; ++i)
    {
        std::shared_ptr<DepthNet> depth_net = std::make_shared<DepthNet>(depth_net_path);
        models.push_back(depth_net);
    }

    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&Thread_Pool::worker, this, i);
    }
    return true;
}



void Thread_Pool::worker(int id)
{
    while (!stop)
    {
        std::pair<int, cv::Mat> task;
        std::shared_ptr<DepthNet> depth_net = models[id]; 

        {   
            std::unique_lock<std::mutex> lock(mtx1);

            cv_task.wait(lock, [&]{ 
                return (!tasks.empty() || stop); 
            });
            
            if(stop)
            {
                return;
            }

            task = tasks.front();
            tasks.pop();
        }
        
        auto idx = task.first;
        auto image = task.second;
        auto depth = depth_net->run(image);
        
        std::lock_guard<std::mutex> lock(mtx2);
        // img_results.insert({idx, depth});
        img_results[idx] = depth;

    }
}


// 提交任务，参数: 图片，id(号)
bool Thread_Pool::submitTask(const cv::Mat &img, int id)
{
    while (tasks.size() > 10)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {   
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});
        cv_task.notify_one();
    }
    return true;
}


bool Thread_Pool::getImgResult(cv::Mat &img, int id)
{

    int loop_cnt = 0;
    while (img_results.find(id) == img_results.end())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        loop_cnt++;
        if(loop_cnt > 1000)
        {
            std::cout << "getImgResult timeout" << std::endl;
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx2);
        img = img_results[id];
        img_results.erase(id);
    }
    return true;
}


// 停止所有线程
void Thread_Pool::stopAll()
{
    stop = true;
    cv_task.notify_all();
}




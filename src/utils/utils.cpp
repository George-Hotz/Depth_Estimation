#include "utils.h"

void video_tool::setUp(const char *video_file){
    cap = cv::VideoCapture(video_file);
    fps = cap.get(cv::CAP_PROP_FPS);
    width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    writer = cv::VideoWriter("output/videos/result.mp4", 
                                cv::VideoWriter::fourcc('H', '2', '6', '4'), 
                                fps, 
                                cv::Size(width, height));
    assert(cap.isOpened());
}
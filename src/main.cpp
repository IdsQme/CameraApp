#include "assert.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>  

int main() {

    // without this CMake crys since I don't have all the dependencies 
    _putenv("OPENCV_VIDEOIO_PRIORITY_LIST=MSMF");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    cv::VideoCapture cap(0);  
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame, prevFrame, diff, thresh;

    while (true) {
        cap >> frame; 
        if (frame.empty()) break;

        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Calc difference between frames for movement
        if (!prevFrame.empty()) {
            cv::absdiff(prevFrame, gray, diff); 
            cv::threshold(diff, thresh, 20, 255, cv::THRESH_BINARY);

            // Find contours (basically what moves)
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::drawContours(frame, contours, -1, cv::Scalar(0,255,0),2);
 

            cv::imshow("Motion", frame);
        }

        // updates frame
        prevFrame = gray; 

        
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
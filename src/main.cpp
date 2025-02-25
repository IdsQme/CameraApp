#include "assert.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <cstdlib>  
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

int main() {
    // Set environment variable for OpenCV (if needed)
    _putenv("OPENCV_VIDEOIO_PRIORITY_LIST=MSMF");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Paths to YOLO files
    std::string modelConfig = "../YOLO/yolov3.cfg"; // Update this path if needed
    std::string modelWeights = "../YOLO/yolov3.weights"; // Update this path if needed
    std::string classFile = "../YOLO/coco.names"; // Update this path if needed

    // Load YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    if (net.empty()) {
        std::cerr << "Error: Could not load YOLO network." << std::endl;
        return -1;
    }


    // Load class names
    std::vector<std::string> classNames;
    std::ifstream classNamesFile(classFile);
    if (!classNamesFile.is_open()) {
        std::cerr << "Error: Could not open class names file." << std::endl;
        return -1;
    }
    std::string className;
    while (std::getline(classNamesFile, className)) {
        classNames.push_back(className);
    }

    // Open the camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // User input for vision option
    int choice;
    std::cout << "Select vision option: \n";
    std::cout << "1 for camera view + motion.\n2 for motion only.\n";
    std::cin >> choice;

    if (choice == 1) {
        // Camera view + motion detection
        cv::Mat frame, prevFrame, diff, thresh;
        while (true) {
           
            cap >> frame;
            if (frame.empty()) break;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Calculate difference between frames for movement
            if (!prevFrame.empty()) {
                cv::absdiff(prevFrame, gray, diff);
                cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);

                // Find contours (motion regions)
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 2);

                cv::imshow("Motion", frame);
            }

            // Update previous frame
            prevFrame = gray;

            if (cv::waitKey(1) == 27) break; // Exit on 'Esc'
        }
    } else if (choice == 2) {
        // Motion detection only
        cv::Mat prevFrame, currentFrame, diffFrame, gray, thresh;
        bool firstFrame = true;

        while (true) {
            cap >> currentFrame; // Capture the current frame
            if (currentFrame.empty()) break;

            // Convert to grayscale
            cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
            if (firstFrame) {
                prevFrame = gray.clone(); // Initialize previous frame
                firstFrame = false;
                continue;
            }

            // Compute absolute difference between current and previous frame
            cv::absdiff(prevFrame, gray, diffFrame);

            // Apply threshold to detect motion
            cv::threshold(diffFrame, thresh, 25, 255, cv::THRESH_BINARY);

            cv::imshow("Motion", thresh); // Show motion detection

            prevFrame = gray.clone(); // Update previous frame

            if (cv::waitKey(30) >= 0) break; // Exit condition
        }
    } else {
        std::cerr << "Invalid choice. Exiting." << std::endl;
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


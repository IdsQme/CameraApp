#include "assert.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <cstdlib>  
#include <fstream>
int main() {
    // without this CMake crys since I don't have all the dependencies 
    _putenv("OPENCV_VIDEOIO_PRIORITY_LIST=MSMF");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

        
    // YOLO model
    std::string modelConfig = "../YOLO/yolov3.cfg";
    std::string modelWeights = "../YOLO/yolov3.weights";
    std::string classFile = "../YOLO/coco.names";
    
    std::ifstream classNamesFile(classFile);
    std::vector<std::string> classNames;
    std::string className;
    while(std::getline(classNamesFile, className))
    {
        classNames.push_back(className);
    }

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig,modelWeights);
    
    if (net.empty()) {
        std::cerr << "Error: Could not load YOLO network." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);  
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame, prevFrame, diff, thresh;
    int choice;
    std::cout << "Select vision option: \n";
    std::cout << "1 for camera view + motion.\n 2 for motion only.";

    std::cin >> choice;

    if(choice== 1){
        while (true) {
            cap >> frame; 
            if (frame.empty()) break;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Calc difference between frames for movement
            if (!prevFrame.empty()) {
                cv::absdiff(prevFrame, gray, diff); 
                cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);

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
    }   
    else if(choice == 2){
        cv::Mat prevFrame, currentFrame, diffFrame, gray, thresh;
        bool firstFrame = true;

        while (true) {
            cap >> currentFrame;  // Capture the current frame
            if (currentFrame.empty()) break;

            // to grayscale 
            cv::cvtColor(currentFrame,gray,cv::COLOR_BGR2GRAY);
            if (firstFrame) {
                prevFrame = gray.clone();  // Initialize previous frame
                firstFrame = false;
                continue;
            }

            // Compute absolute difference between current and previous frame
            cv::absdiff(prevFrame, gray, diffFrame);

            // Convert to grayscale and apply threshold to detect motion
            cv::threshold(diffFrame, thresh, 25, 255, cv::THRESH_BINARY);


 
            cv::imshow("Motion", thresh);  // Show only when motion is detected
      
            prevFrame = gray.clone();  // Update previous frame

            if (cv::waitKey(30) >= 0) break;  // Exit condition
        }

    } else {
        while (cap.read(frame)) {
            // Prepare the image for YOLO detection
            cv::Mat blob = cv::dnn::blobFromImage(frame, 0.00392, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
            net.setInput(blob);
    
            // Get YOLO layer names
            std::vector<cv::String> layerNames = net.getLayerNames();
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            std::vector<cv::String> outLayerNames;
            for (size_t i = 0; i < outLayers.size(); i++) {
                outLayerNames.push_back(layerNames[outLayers[i] - 1]);
            }
    
            // Run forward pass to get predictions
            std::vector<cv::Mat> detections;
            net.forward(detections, outLayerNames);
    
            std::vector<int> indices;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
    
            // Process detections
            for (auto& detection : detections) {
                for (int i = 0; i < detection.rows; i++) {
                    float confidence = detection.at<float>(i, 4); // Confidence score
                    if (confidence > 0.5) { // Adjust confidence threshold as needed
                        int centerX = static_cast<int>(detection.at<float>(i, 0) * frame.cols);
                        int centerY = static_cast<int>(detection.at<float>(i, 1) * frame.rows);
                        int width = static_cast<int>(detection.at<float>(i, 2) * frame.cols);
                        int height = static_cast<int>(detection.at<float>(i, 3) * frame.rows);
    
                        // Add bounding box and confidence score
                        boxes.push_back(cv::Rect(centerX - width / 2, centerY - height / 2, width, height));
                        confidences.push_back(confidence);
    
                        // Get class ID and label
                        int classID = static_cast<int>(detection.at<float>(i, 5)); // Class ID
                        std::string label = classNames[classID];
    
                        // Draw bounding box and label
                        cv::rectangle(frame, cv::Point(centerX - width / 2, centerY - height / 2),
                            cv::Point(centerX + width / 2, centerY + height / 2), cv::Scalar(0, 255, 0), 2);
                        cv::putText(frame, label, cv::Point(centerX, centerY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
    
            // Apply Non-Maximum Suppression to remove redundant boxes
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, nmsIndices); // 0.5 is confidence threshold, 0.4 is IoU threshold
    
            // Draw final boxes after NMS
            for (size_t i = 0; i < nmsIndices.size(); i++) {
                int idx = nmsIndices[i];
                cv::rectangle(frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, classNames[std::distance(boxes.begin(), boxes.begin() + idx)], 
                            boxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
    
            // Show the resulting frame
            cv::imshow("YOLO Object Detection", frame);
            if (cv::waitKey(1) == 27) break; // Press 'Esc' to exit
        }
    }
    

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-18
 * @author Anonymous
 */

#include <chrono>
#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"

int demo_object_counter(int argc, char* argv[])
{
    // cv::VideoCapture cap("TestVideo.mp4");
    cv::VideoCapture cap("D:/Документы/11\ semester/u131045/ComputerVision/lab1/cvclasses18/Video/TestVideo.mp4");

    if (!cap.isOpened())
        return -1;

    auto amcn = new cvlib::object_counter();
    const auto main_wnd = "orig";
    //const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    //cv::namedWindow(demo_wnd);

	int numCars = 0;
    cv::Mat frame;
    cv::Mat frame_amcn;
    utils::fps_counter fps;
    const int area_width = 320; // width of inspecting area (pix)

    std::ofstream out("result.txt");
    if (!out.is_open()) return -1;

    // auto time_start = std::chrono::system_clock::now();
    cv::Mat tempAreaInFrame;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
		if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(640, 480));
        //cv::imshow(main_wnd, frame);

        // Область подсчета
        std::vector<cv::Rect> movedAreas, checkedAreas;
        const cv::Rect tempArea = cv::Rect(frame.cols / 2 - area_width, frame.rows / 3, 2 * area_width, frame.rows / 3);
        //amcn->setCountingArea(cv::Rect(frame.cols / 2 - area_width, frame.rows / 3, 2 * area_width, frame.rows / 3));
        amcn->setCountingArea(tempArea);

        tempAreaInFrame = frame(tempArea);
        amcn->apply(tempAreaInFrame, /*frame_amcn,*/ movedAreas, checkedAreas);
        if (!tempAreaInFrame.empty())
        {
            if (!movedAreas.empty())
            {
                for (auto& it : movedAreas)
                {
                    const cv::Point point0 = {it.x + frame.cols / 2 - area_width, it.y + frame.rows / 3};
                    const cv::Point point1 = {point0.x + it.width, point0.y + it.height};
                    cv::Rect roiIt = cv::Rect(point0, point1);
                    //cv::rectangle(frame_amcn, roiIt, cv::Scalar(0, 0, 255), 2);
                    cv::rectangle(frame, roiIt, cv::Scalar(0, 0, 255), 2);
                }
                for (auto& it : checkedAreas)
                {
                    const cv::Point point0 = {it.x + frame.cols / 2 - area_width, it.y + frame.rows / 3};
                    const cv::Point point1 = {point0.x + it.width, point0.y + it.height};
                    cv::Rect roiIt = cv::Rect(point0, point1);
                    cv::rectangle(frame, roiIt, cv::Scalar(0, 255, 0), 2);
                    //cv::rectangle(frame_amcn, roiIt, cv::Scalar(0, 255, 0), 2);
                }
            }

            //cv::line(frame_amcn, cv::Point(frame_amcn.cols / 2, frame.rows / 3), cv::Point(frame_amcn.cols / 2, 2 * frame_amcn.rows / 3),
            //         cv::Scalar(0, 0, 255), 2, 8);
            cv::line(frame, cv::Point(tempAreaInFrame.cols / 2, frame.rows / 3), cv::Point(tempAreaInFrame.cols / 2, 2 * frame.rows / 3),
                cv::Scalar(0, 0, 255), 2, 8);

            //utils::put_fps_text(frame_amcn, fps);
            utils::put_fps_text(frame, fps);
			while (numCars < amcn->getNumCars())
			{
				numCars++;
				out << cap.get(cv::CAP_PROP_POS_MSEC) << "\n";
			}
            //utils::put_car_count_text(frame_amcn, numCars);
            utils::put_car_count_text(frame, numCars);

            // auto time_now = std::chrono::system_clock::now();
            // std::chrono::duration<double> time_seconds = time_now - time_start;
            //utils::put_time_text(frame_amcn, cap.get(cv::CAP_PROP_POS_MSEC)/1000);//time_seconds.count());
            utils::put_time_text(frame, cap.get(cv::CAP_PROP_POS_MSEC) / 1000);
            cv::imshow(main_wnd, frame);
            //cv::imshow(demo_wnd, frame_amcn);
        }
    }
	
	out.close();

    cv::destroyWindow(main_wnd);
    //cv::destroyWindow(demo_wnd);

    return 0;
}

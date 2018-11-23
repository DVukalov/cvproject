/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-18
 * @author Anonymous
 */

#include <chrono>
#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_object_counter(int argc, char* argv[])
{
    // cv::VideoCapture cap("TestVideo.mp4");
    cv::VideoCapture cap("D:/Документы/11\ semester/u131045/ComputerVision/lab1/cvclasses18/Video/TestVideo.mp4");

    if (!cap.isOpened())
        return -1;

    auto amcn = new cvlib::object_counter();
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    cv::Mat frame_amcn;
    utils::fps_counter fps;
    const int area_width = 100; // width of inspecting area (pix)

    auto time_start = std::chrono::system_clock::now();
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::resize(frame, frame, cv::Size(640, 480));
        cv::imshow(main_wnd, frame);

        // Область подсчета
        std::vector<cv::Rect> movedAreas;
        amcn->setCountingArea(cv::Rect(frame.cols / 2 - area_width, frame.rows / 3, 2 * area_width, frame.rows / 3));
        amcn->apply(frame, frame_amcn, movedAreas);
        if (!frame_amcn.empty())
        {
            if (!movedAreas.empty())
            {
                for (auto& it : movedAreas)
                {
                    const cv::Point point0 = {it.x + frame.cols / 2 - area_width, it.y + frame.rows / 3};
                    const cv::Point point1 = {point0.x + it.width, point0.y + it.height};
                    cv::Rect roiIt = cv::Rect(point0, point1);
                    cv::rectangle(frame_amcn, roiIt, cv::Scalar(0, 255, 0), 2);
                }
            }

            cv::line(frame_amcn, cv::Point(frame_amcn.cols / 2, frame.rows / 3), cv::Point(frame_amcn.cols / 2, 2 * frame_amcn.rows / 3),
                     cv::Scalar(0, 0, 255), 2, 8);
            utils::put_fps_text(frame_amcn, fps);
            utils::put_car_count_text(frame_amcn, 0);

            auto time_now = std::chrono::system_clock::now();
            std::chrono::duration<double> time_seconds = time_now - time_start;
            utils::put_time_text(frame_amcn, time_seconds.count());
            cv::imshow(demo_wnd, frame_amcn);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}

/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-18
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_object_counter(int argc, char* argv[])
{
    cv::VideoCapture cap("TestVideo.mp4");
    if (!cap.isOpened()) return -1;

    auto amcn = new cvlib::object_counter();
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    cv::Mat frame_amcn;
    utils::fps_counter fps;

    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
		cv::resize(frame, frame, cv::Size(640, 480));
        cv::imshow(main_wnd, frame);

		// Область подсчета
		amcn->setCountingArea(cv::Rect(frame.cols / 2 - 30, frame.rows/3, 60, frame.rows/3));
        amcn->apply(frame, frame_amcn);
        if (!frame_amcn.empty())
		{
			cv::line(frame_amcn, cv::Point(frame_amcn.cols / 2, frame.rows/3),
					 cv::Point(frame_amcn.cols / 2, 2*frame_amcn.rows/3),
					 cv::Scalar(0, 0, 255), 2, 8);
			utils::put_fps_text(frame_amcn, fps);
			utils::put_car_count_text(frame_amcn, 0);
			cv::imshow(demo_wnd, frame_amcn);
		}
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}

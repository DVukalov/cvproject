/* Utilites implemetation for demo applications of Computer Vision Library.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include <opencv2/opencv.hpp>

#include <vector>

#include "utils.hpp"

namespace
{
const auto txtFont = CV_FONT_HERSHEY_PLAIN;
const auto fontScale = 1;
const auto thickness = 1;
const cv::Size fpsTextSize = cv::getTextSize("FPS: 19.1275", txtFont, fontScale, thickness, nullptr);
const cv::Size carsTextSize = cv::getTextSize("CARS: 10000", txtFont, fontScale, thickness, nullptr);
const cv::Size timeTextSize = cv::getTextSize("TIME: 00:00", txtFont, fontScale, thickness, nullptr);
const int maxTextWidth = std::max(fpsTextSize.width, std::max(carsTextSize.width, timeTextSize.width));
const cv::Size maxTextSize = cv::Size(maxTextWidth, fpsTextSize.height);
} // namespace

namespace utils
{
moving_average::moving_average(int length /*= 30*/) : history_length_(length)
{
    history_.resize(history_length_, 0.0);
}

double moving_average::add_new_value(double val)
{
    if (!is_initialized_)
    {
        if (current_index_ < history_length_)
        {
            history_[current_index_] = val;
            accumulator_ += val;
            return accumulator_ / current_index_++;
        }
        else
        {
            is_initialized_ = true;
        }
    }

    if (current_index_ == history_length_)
    {
        current_index_ = 0;
    }

    accumulator_ -= history_[current_index_];
    history_[current_index_++] = val;
    accumulator_ += val;

    return accumulator_ / history_length_;
}

fps_counter::fps_counter(int length /*= 30*/) : last_tick_count_(cv::getTickCount()), accum_(length)
{
}

double fps_counter::process_new_frame()
{
    const int64 new_tick_count = cv::getTickCount();
    const double diff = cv::getTickFrequency() / (new_tick_count - last_tick_count_);
    last_tick_count_ = new_tick_count;
    return accum_.add_new_value(diff);
}

void put_fps_text(cv::Mat& image, fps_counter& fps, cv::Scalar color /*= (255, 0, 0)*/)
{
    static const cv::Point textOrgPoint = {0, maxTextSize.height * 2 + 4};
    static const cv::Point rectPoint1 = {0, maxTextSize.height};
    static const cv::Point rectPoint2 = {maxTextSize.width + 2, (maxTextSize.height + 3) * 2};

    std::stringstream ss;
    ss.precision(4);

    ss << "FPS: " << std::fixed << fps.process_new_frame();

    cv::Rect fpsRect = cv::Rect(rectPoint1, rectPoint2);
    cv::rectangle(image, fpsRect, cv::Scalar(0, 0, 0), -1);

    cv::putText(image, ss.str(), textOrgPoint, txtFont, fontScale, color, thickness, 8, false);
}

void put_car_count_text(cv::Mat& image, size_t car_count, cv::Scalar color /*= (255, 255, 255)*/)
{
    static const cv::Point textOrgPoint = {0, maxTextSize.height + 1};
    static const cv::Point rectPoint1 = {0, 0};
    static const cv::Point rectPoint2 = {maxTextSize.width + 2, maxTextSize.height + 3};

    std::stringstream ss;
    ss << "CARS: " << car_count;

    cv::Rect fpsRect = cv::Rect(rectPoint1, rectPoint2);
    cv::rectangle(image, fpsRect, cv::Scalar(0, 0, 0), -1);

    cv::putText(image, ss.str(), textOrgPoint, txtFont, fontScale, color, thickness, 8, false);
}

void put_time_text(cv::Mat& image, double time_now, cv::Scalar color /*= (255, 255, 255)*/)
{
    static const cv::Point textOrgPoint = {0, maxTextSize.height * 4 - 2};
    static const cv::Point rectPoint1 = {0, maxTextSize.height * 2 + 4};
    static const cv::Point rectPoint2 = {maxTextSize.width + 2, maxTextSize.height * 4 + 1};

    std::stringstream ss;
    ss.precision(5);
    ss << "TIME: " << time_now;

    cv::Rect fpsRect = cv::Rect(rectPoint1, rectPoint2);
    cv::rectangle(image, fpsRect, cv::Scalar(0, 0, 0), -1);

    cv::putText(image, ss.str(), textOrgPoint, txtFont, fontScale, color, thickness, 8, false);
}

} // namespace utils

/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{

object_counter::object_counter()
{
	pSubstractor = cv::createBackgroundSubtractorMOG2();
}

void object_counter::setCountingArea(const cv::Rect& area)
{
	if (mCountingArea != area)
		mCountingArea = area;
}

void object_counter::apply(const cv::Mat& image, cv::Mat& result)
{
	if (image.empty()) return;

	result.release();
	if (image.channels() > 1) cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
	else result = image.clone();

	cv::Mat roi = result(mCountingArea);
	pSubstractor->apply(roi, roi);

	cv::threshold(roi, roi, 250, 255, cv::THRESH_BINARY);
	
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
	// cv::morphologyEx(roi, roi, cv::MORPH_OPEN, kernel);
	// cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, kernel);
	cv::erode(roi, roi, kernel, cv::Point2f(-1,-1), 1);
	cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

} // namespace cvlib

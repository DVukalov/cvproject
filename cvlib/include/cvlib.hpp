/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \return segmented image
cv::Mat split_and_merge(const cv::Mat& image, double stddev);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    /// \brief ctor
    motion_segmentation();

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = -1) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override
    {
        backgroundImage.assign(bg_model_);
    }

    private:
    cv::Mat bg_model_;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    /// \see Feature2d::compute
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;

    /// \see Feature2d::getDefaultName
    virtual cv::String getDefaultName() const override
    {
        return "FAST_Binary";
    }
};

struct TrackHistory;

/// \brief Moving objects counter class
class object_counter
{
    public:
    /// \brief ctor
    object_counter();

    void setCountingArea(const cv::Rect& area);

    void apply(const cv::Mat& image, /*cv::Mat& result,*/ std::vector<cv::Rect>& movedAreas, std::vector<cv::Rect>& checkedAreas);

    void thresholdImage(cv::Mat& src);
    std::vector<int> regionQuery(std::vector<cv::Point>* points, cv::Point* point, float eps);
    std::vector<std::vector<cv::Point>> dbScan_points(std::vector<cv::Point>* points, float eps, int minPts);

    bool isRectsOverlap(const cv::Rect& rect1, const cv::Rect& rect2);
    void compareArrays(std::vector<cv::Rect>& firstArray, std::vector<cv::Rect>& secondArray, std::vector<cv::Rect>& resultArray);
    cv::Rect merge(const cv::Rect& rect1, const cv::Rect& rect2);
    void mergeIntersectingRectangles(std::vector<cv::Rect>& sourceArray, std::vector<cv::Rect>& resultArray);
    void filterRectangles(const std::vector<cv::Rect> &sourceArray, std::vector<cv::Rect> &moved, std::vector<cv::Rect> &checked);
    void incrNumCars();
    size_t getNumCars();
	
	bool isRectsNear(const cv::Rect &rect1, const cv::Rect &rect2, int threshold);
	void mergeNearestRectangles(std::vector<cv::Rect> &sourceArray, std::vector<cv::Rect> &resultArray, int threshold);

    private:
    cv::Rect mCountingArea; // Inside this area we will count objects
    cv::Ptr<cv::BackgroundSubtractor> pSubstractor;
    std::vector<TrackHistory> mTrackedRects;

    std::vector<cv::Rect> mBoundRect;
    std::vector<cv::Rect> mRectFirstArray;
    std::vector<cv::Rect> mRectSecondArray;
    size_t mNumCars;
};

} // namespace cvlib

#endif // __CVLIB_HPP__

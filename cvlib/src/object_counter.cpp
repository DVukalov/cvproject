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
    mNumCars = 0;
    pSubstractor = cv::createBackgroundSubtractorMOG2(); // пока идеальные параметры
    // pSubstractor = cv::createBackgroundSubtractorMOG2(100, 50.0, true);

    // выделяет стабильнее, но очень тормозит все
    // pSubstractor = cv::createBackgroundSubtractorKNN(500, 300.0, false);
}

void object_counter::setCountingArea(const cv::Rect& area)
{
    if (mCountingArea != area)
        mCountingArea = area;
}

void object_counter::apply(const cv::Mat& image, cv::Mat& result, std::vector<cv::Rect>& movedAreas)
{
    if (image.empty())
        return;

    result.release();
    if (image.channels() > 1)
        cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
    else
        result = image.clone();

    cv::Mat roi = result(mCountingArea);
    pSubstractor->apply(roi, roi);

    // supa-pupa-lupa algorithm
    this->thresholdImage(roi);
    if (!mBoundRect.empty())
    {
        // минимальная площадь строба
        int minSize = 30;
        // фильтр маленьких областей
        mBoundRect.erase(std::remove_if(mBoundRect.begin(), mBoundRect.end(), [&](const cv::Rect& rect) { return rect.area() < minSize; }),
                         mBoundRect.end());
        mRectSecondArray = mBoundRect;
        if (mRectFirstArray.empty())
        {
            mRectFirstArray = mRectSecondArray;
        }
        std::vector<cv::Rect> tmpArray;
        compareArrays(mRectFirstArray, mRectSecondArray, tmpArray);
        std::vector<cv::Rect> mergeArray;
        mergeIntersectingRectangles(tmpArray, mergeArray);
        for (auto& it : mergeArray)
        {
            //проверка на выход стробов за границы фрейма
            if (it.x > 0 && it.y > 0 && (it.x + it.width) < roi.cols && (it.y + it.height) < roi.rows)
            {
                movedAreas.push_back(it);
            }
        }
        mBoundRect.clear();
    }

    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

void object_counter::thresholdImage(cv::Mat& src)
{
    if (src.empty())
        return;
    // морфологическое размыкание (крестом 4х4)
    const cv::Mat cross = (cv::Mat_<uchar>(4, 4) << 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0);
    cv::morphologyEx(src, src, cv::MORPH_OPEN, cross, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    // ищем замкнутые контуры, затем аппроксимируем их и затем заново находим
    // контуры кластеризацией, в итоге получаем меньше стробов на объект
    std::vector<std::vector<cv::Point>> foundContours;
    cv::findContours(src, foundContours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    std::vector<cv::Point> contourMesh;
    for (auto& it : foundContours)
    {
        std::vector<cv::Point> tmp;
        cv::approxPolyDP(it, tmp, 0.1 * cv::arcLength(it, true), true);
        contourMesh.insert(contourMesh.end(), tmp.begin(), tmp.end());
    }
    foundContours.clear();
    foundContours = dbScan_points(&contourMesh, 50, 3);
    if (!foundContours.empty())
    {
        for (auto& it : foundContours)
        {
            if (!it.empty())
                mBoundRect.push_back(cv::boundingRect(it));
        }
    }
}

bool object_counter::isRectsOverlap(const cv::Rect& rect1, const cv::Rect& rect2)
{
    int x11 = rect1.x;
    int y11 = rect1.y;
    int x12 = rect1.x + rect1.width;
    int y12 = rect1.y + rect1.height;

    int x21 = rect2.x;
    int y21 = rect2.y;
    int x22 = rect2.x + rect2.width;
    int y22 = rect2.y + rect2.height;

    int x_overlap = std::max(0, std::min(x12, x22) - std::max(x11, x21));
    int y_overlap = std::max(0, std::min(y12, y22) - std::max(y11, y21));
    double overlapArea = (double)x_overlap * (double)y_overlap;

    return (overlapArea > 0.0);
}

void object_counter::compareArrays(std::vector<cv::Rect>& firstArray, std::vector<cv::Rect>& secondArray, std::vector<cv::Rect>& resultArray)
{
    //для фильтрации ложных срабатываний сравниваем на 2х последовательных
    //кадрах прямоугольники. считаем истинными те, которые пересекаются.
    if (firstArray.empty() || secondArray.empty())
        return;
    for (std::vector<cv::Rect>::iterator firstArrayPtr = firstArray.begin(); firstArrayPtr != firstArray.end(); ++firstArrayPtr)
    {
        for (std::vector<cv::Rect>::iterator secArrayPtr = secondArray.begin(); secArrayPtr != secondArray.end(); ++secArrayPtr)
        {
            if (isRectsOverlap(*firstArrayPtr, *secArrayPtr))
            {
                resultArray.push_back(*secArrayPtr);
            }
        }
    }
    firstArray.clear();
    firstArray = secondArray;
}

cv::Rect object_counter::merge(const cv::Rect& rect1, const cv::Rect& rect2)
{
    return cv::Rect(std::min(rect1.x, rect2.x), std::min(rect1.y, rect2.y), std::max(rect1.width, rect2.width), std::max(rect1.height, rect2.height));
}

void object_counter::mergeIntersectingRectangles(std::vector<cv::Rect>& sourceArray, std::vector<cv::Rect>& resultArray)
{
    if (sourceArray.empty())
        return;
    //ищем для каждого прямоугольника пересечения, объединяем и удаляем пересекающиеся.
    //в итоге остаются только непересекающиеся прямоугольники
    std::vector<cv::Rect>::iterator srcPtr1 = sourceArray.begin();
    while (srcPtr1 != sourceArray.end() && sourceArray.size() != 0)
    {
        std::vector<cv::Rect>::iterator srcPtr2 = sourceArray.begin();
        while (srcPtr2 != sourceArray.end() && sourceArray.size() != 0)
        {
            if (srcPtr1 != srcPtr2)
            {
                if (isRectsOverlap(*srcPtr1, *srcPtr2))
                {
                    cv::Rect merged = merge(*srcPtr1, *srcPtr2);
                    std::swap(*srcPtr1, merged);
                    srcPtr2 = sourceArray.erase(srcPtr2);
                }
                else
                    srcPtr2++;
            }
            else
                srcPtr2++;
        }
        srcPtr1++;
    }
    resultArray = sourceArray;
}

void object_counter::incrNumCars()
{
    mNumCars++;
}

size_t object_counter::getNumCars()
{
    return mNumCars;
}

std::vector<int> object_counter::regionQuery(std::vector<cv::Point>* points, cv::Point* point, float eps)
{
    double dist;
    std::vector<int> retKeys;
    for (uint i = 0; i < points->size(); i++)
    {
        //евклидово расстояние между точками
        dist = cv::norm(*point - points->at(i));
        if (dist <= eps && dist != 0.0f)
        {
            retKeys.push_back(i);
        }
    }
    return retKeys;
}

std::vector<std::vector<cv::Point>> object_counter::dbScan_points(std::vector<cv::Point>* points, float eps, int minPts)
{
    std::vector<std::vector<cv::Point>> clusters;
    std::vector<bool> clustered;
    std::vector<bool> visited;
    std::vector<int> neighborPts;
    std::vector<int> neighborPts_;
    int c;

    size_t noKeys = points->size();

    for (size_t k = 0; k < noKeys; k++)
    {
        clustered.push_back(false);
        visited.push_back(false);
    }

    c = 0;
    clusters.push_back(std::vector<cv::Point>());

    for (size_t i = 0; i < noKeys; i++)
    {
        if (!visited[i])
        {
            visited[i] = true;
            neighborPts = regionQuery(points, &points->at(i), eps);
            if (neighborPts.size() >= (uint)minPts)
            {
                clusters.push_back(std::vector<cv::Point>());
                c++;
                clusters[c].push_back(points->at(i));
                for (uint j = 0; j < neighborPts.size(); j++)
                {
                    if (!visited[neighborPts[j]])
                    {
                        visited[neighborPts[j]] = true;
                        neighborPts_ = regionQuery(points, &points->at(neighborPts[j]), eps);
                        if (neighborPts_.size() >= (uint)minPts)
                        {
                            neighborPts.insert(neighborPts.end(), neighborPts_.begin(), neighborPts_.end());
                        }
                    }
                    if (!clustered[neighborPts[j]])
                    {
                        clustered[neighborPts[j]] = true;
                        clusters[c].push_back(points->at(neighborPts[j]));
                    }
                }
            }
        }
    }
    return clusters;
}

} // namespace cvlib

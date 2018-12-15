/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

const int gHistorySize = 10;

namespace cvlib
{

struct TrackHistory
{
    cv::Rect getRect() const
    {
        return storedRect;
    }

    void predictRect()
    {
        cv::Rect pred(storedRect.x + trajectory.x*missedFrames,
                      storedRect.y + trajectory.y*missedFrames, avgWidth, avgHeight);
		//cv::Rect pred;
        storedRect = pred;
    }

    void addRect(const cv::Rect& rect)
    {
        if (storedRect.area() > 0)
        {
            int dx = rect.x + rect.width/2 - storedRect.x - storedRect.width/2;
            int dy = rect.y + rect.height/2 - storedRect.y - storedRect.height/2;
            trajectory = cv::Point2f(dx, dy);
        }

        storedRect = rect;
        missedFrames = 0;
        trackedFrames = std::min(trackedFrames+1, gHistorySize);

        avgWidth += (rect.width - avgWidth)/trackedFrames;
        avgHeight += (rect.height - avgHeight)/trackedFrames;
    }

    void startSearchingNext()
    {
        missedFrames++;
    }

    bool willBeNext(const cv::Rect& rect, cv::Rect& newRect) const
    {
        if (this->isTracked()) return false;

        int dw = ((int)avgWidth - rect.width)/4;
        int dh = ((int)avgHeight - rect.height)/4;

        // большие стробы имеют больший вес
        if (rect.width > avgWidth) dw /= 2;
        if (rect.height > avgHeight) dh /= 2;

        // слишком маленькие стробы растягиваем до больших
        if (rect.width*3 < avgWidth) dw *= 2;
        if (rect.height*3 < avgHeight) dh *= 2;

        newRect = cv::Rect(rect.x - dw, rect.y - dh, rect.width + 2*dw, rect.height + 2*dh);

        return (newRect.area() > 0) && ((newRect & storedRect).area() > 0);
    }

    bool isTracked() const
    {
        return (trackedFrames >= gHistorySize) && (missedFrames == 0);
    }

    bool isValid() const
    {
        return missedFrames < gHistorySize;
    }

    bool isMissed() const
    {
        return (missedFrames < 5) && !isTracked() && (trackedFrames >= gHistorySize);
    }

	bool isChecked() const
	{
		return checked;
	}
	
	void setChecked(const bool& chk)
	{
		checked = chk;
	}
	
private:
    double avgWidth = 0;
    double avgHeight = 0;
    int missedFrames = 0;
    int trackedFrames = 0;
	bool checked = false;
    cv::Rect storedRect;
    cv::Point2f trajectory;
};
	
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

void object_counter::apply(const cv::Mat& image, cv::Mat& result, std::vector<cv::Rect>& movedAreas, std::vector<cv::Rect>& checkedAreas)
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
	cv::threshold(roi, roi, 200, 255, CV_THRESH_BINARY); // delete shadows

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
        if (mRectFirstArray.empty()) mRectFirstArray = mRectSecondArray;
        std::vector<cv::Rect> tmpArray;
        compareArrays(mRectFirstArray, mRectSecondArray, tmpArray);
        std::vector<cv::Rect> mergeArray;
        mergeIntersectingRectangles(tmpArray, mergeArray);
        std::vector<cv::Rect> movedArray, checkedArray;
        this->filterRectangles(mergeArray, movedArray, checkedArray);
        for (auto &it : movedArray)
        {
            //проверка на выход стробов за границы фрейма
            if (it.x > 0 && it.y > 0 && (it.x + it.width) < roi.cols && (it.y + it.height) < roi.rows)
            {
                movedAreas.push_back(it);
            }
        }
		for (auto &it : checkedArray)
        {
            //проверка на выход стробов за границы фрейма
            if (it.x > 0 && it.y > 0 && (it.x + it.width) < roi.cols && (it.y + it.height) < roi.rows)
            {
                checkedAreas.push_back(it);
            }
        }
        mBoundRect.clear();
    }

    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
}

void object_counter::thresholdImage(cv::Mat& src)
{
    if (src.empty()) return;
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
            if (!it.empty()) mBoundRect.push_back(cv::boundingRect(it));
        }
    }
}

bool object_counter::isRectsOverlap(const cv::Rect& rect1, const cv::Rect& rect2)
{
    // int x11 = rect1.x;
    // int y11 = rect1.y;
    // int x12 = rect1.x + rect1.width;
    // int y12 = rect1.y + rect1.height;

    // int x21 = rect2.x;
    // int y21 = rect2.y;
    // int x22 = rect2.x + rect2.width;
    // int y22 = rect2.y + rect2.height;

    // int x_overlap = std::max(0, std::min(x12, x22) - std::max(x11, x21));
    // int y_overlap = std::max(0, std::min(y12, y22) - std::max(y11, y21));
    // double overlapArea = (double)x_overlap * (double)y_overlap;
    // return (overlapArea > 0.0);
	
	return ((rect1 & rect2).area() > 0);
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
            if ((*firstArrayPtr & *secArrayPtr).area() > 0)//isRectsOverlap(*firstArrayPtr, *secArrayPtr))
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
	int x = std::min(rect1.x, rect2.x);
	int y = std::min(rect1.y, rect2.y);
	int width = std::max(rect1.x + rect1.width, rect2.x + rect2.width) - x;
	int height = std::max(rect1.y + rect1.height, rect2.y + rect2.height) - y;
    return cv::Rect(x, y, width, height);
}

void object_counter::mergeIntersectingRectangles(std::vector<cv::Rect>& sourceArray, std::vector<cv::Rect>& resultArray)
{
    if (sourceArray.empty()) return;
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
                if ((*srcPtr1 & *srcPtr2).area() > 0)//isRectsOverlap(*srcPtr1, *srcPtr2))
                {
                    cv::Rect merged = (*srcPtr1 | *srcPtr2);//merge(*srcPtr1, *srcPtr2);
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

bool object_counter::isRectsNear(const cv::Rect &rect1, const cv::Rect &rect2, int threshold)
{
	if (isRectsOverlap(rect1, rect2)) return true;

	int xc1 = rect1.x + rect1.width/2;
	int yc1 = rect1.y + rect1.height/2;
	int xc2 = rect2.x + rect2.width/2;
	int yc2 = rect2.y + rect2.height/2;
	
	int x_diff = xc1 - xc2;
	int y_diff = yc1 - yc2;
	int dist = x_diff*x_diff + y_diff*y_diff;

	if(dist <= threshold*threshold && std::abs(y_diff) < threshold*threshold)
	{
		return true;
	}
	else return false;
}

void object_counter::mergeNearestRectangles(std::vector<cv::Rect> &sourceArray, std::vector<cv::Rect> &resultArray, int threshold)
{
	/*if(sourceArray.empty()) return;
	//ищем для каждого прямоугольника ближайшие и объединяем
	std::vector<cv::Rect>::iterator srcPtr1 = sourceArray.begin();
	while(srcPtr1 != sourceArray.end() && sourceArray.size() != 0)
	{
		std::vector<cv::Rect>::iterator srcPtr2 = sourceArray.begin();
		while(srcPtr2 != sourceArray.end() && sourceArray.size() != 0)
		{
			if(srcPtr1 != srcPtr2)
			{
                if (isRectsNear(*srcPtr1, *srcPtr2, threshold))
				{
					cv::Rect merged = merge(*srcPtr1, *srcPtr2);
					std::swap(*srcPtr1, merged);
					srcPtr2 = sourceArray.erase(srcPtr2);
				}
				else srcPtr2++;
			}
			else srcPtr2++;
		}
		srcPtr1++;
	}
	resultArray = sourceArray;*/
	
	if(sourceArray.empty()) return;

	resultArray.clear();
	//ищем для каждого прямоугольника ближайшие и объединяем
	for (int i = 0; i < sourceArray.size() && sourceArray.size() > 0; i++)
	{
		int j = 0;
		cv::Rect rect1 = sourceArray.at(i);
		while (j < sourceArray.size() && sourceArray.size() > 0)
		{
			cv::Rect rect2 = sourceArray.at(j);
			if (rect1 != rect2)
			{
				if (isRectsNear(rect1, rect2, threshold))
				{
					rect1 = merge(rect1, rect2);
					sourceArray.erase(sourceArray.begin() + j);
					j = 0;
				}
				else j++;
			}
			else j++;
		}
		resultArray.push_back(rect1);
	}
	
	
	// std::vector<cv::Rect>::iterator srcPtr1 = sourceArray.begin();
	// while(srcPtr1 != sourceArray.end() && sourceArray.size() != 0)
	// {
		// std::vector<cv::Rect>::iterator srcPtr2 = sourceArray.begin();
		// while(srcPtr2 != sourceArray.end() && sourceArray.size() != 0)
		// {
			// if(srcPtr1 != srcPtr2)
			// {
                // if (isRectsNear(*srcPtr1, *srcPtr2, threshold))
				// {
					// cv::Rect merged = merge(*srcPtr1, *srcPtr2);
					// // std::swap(*srcPtr1, merged);
					// *srcPtr1 = merged;
					// // srcPtr2 = sourceArray.erase(srcPtr2);
					// sourceArray.erase(srcPtr2);
					// srcPtr2 = sourceArray.begin();
				// }
				// else srcPtr2++;
			// }
			// else srcPtr2++;
		// }
		// srcPtr1++;
	// }
	// resultArray = sourceArray;
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

void object_counter::filterRectangles(const std::vector<cv::Rect> &sourceArray, std::vector<cv::Rect> &moved, std::vector<cv::Rect> &checked)
{
    if (sourceArray.empty()) return;

    for (uint i = 0; i < mTrackedRects.size(); i++)
    {
        mTrackedRects[i].startSearchingNext();
    }

    for (auto& it : sourceArray)
    {
        bool added = false;
        for (uint i = 0; i < mTrackedRects.size() && !added; i++)
        {
            cv::Rect newRect;
            if (mTrackedRects[i].willBeNext(it, newRect))
            {
                mTrackedRects[i].addRect(newRect);
                added = true;
				
				if (!mTrackedRects[i].isChecked())
				{
					cv::Rect curr = mTrackedRects[i].getRect(); 
					bool passed = (mCountingArea.x + mCountingArea.width/2 - curr.x) < 6
						&& (curr.x - mCountingArea.x - mCountingArea.width/2) < 16;
					if (passed)
					{
						mTrackedRects[i].setChecked(true);
						incrNumCars();
						if (curr.height > mCountingArea.height/3) incrNumCars();
					}
				}
            }
        }

        if (!added)
        {
            TrackHistory h;
            h.addRect(it);
            mTrackedRects.push_back(h);
        }
    }

    mTrackedRects.erase(std::remove_if(mTrackedRects.begin(), mTrackedRects.end(),
                                    [&](const TrackHistory &history) {
        return !history.isValid();
    }), mTrackedRects.end());

    for (auto& it : mTrackedRects)
    {
        if (it.isTracked())
		{
			if (it.isChecked()) checked.push_back(it.getRect());
			else moved.push_back(it.getRect());
		}
    }

   for (uint i = 0; i < mTrackedRects.size(); i++)
   {
       if (mTrackedRects[i].isMissed()) mTrackedRects[i].predictRect();
   }
}

} // namespace cvlib

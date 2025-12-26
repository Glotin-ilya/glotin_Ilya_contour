#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

struct Pt {
    double x, y;
    Pt(double x_, double y_) : x(x_), y(y_) {}
    Pt(const cv::Point& p) : x(static_cast<double>(p.x)), y(static_cast<double>(p.y)) {}
};

std::vector<Pt> global_contour;
std::vector<bool> global_isKey;

void findKeyPoints(int start, int end) {
    const double EPS = 0.1;

    if (end <= start + 1) return;

    double dx = global_contour[start].x - global_contour[end].x;
    double dy = global_contour[start].y - global_contour[end].y;
    double baseLen = std::sqrt(dx * dx + dy * dy);
    if (baseLen < 1e-9) return;

    double maxRatio = 0.0;
    int bestIdx = -1;

    for (int i = start + 1; i < end; ++i) {
        double area = std::abs(
            (global_contour[end].x - global_contour[start].x) *
            (global_contour[start].y - global_contour[i].y) -
            (global_contour[start].x - global_contour[i].x) *
            (global_contour[end].y - global_contour[start].y)
        );
        double d = area / baseLen;
        double ratio = d / baseLen;

        if (ratio > maxRatio) {
            maxRatio = ratio;
            bestIdx = i;
        }
    }

    if (maxRatio > EPS && bestIdx != -1) {
        global_isKey[bestIdx] = true;
        findKeyPoints(start, bestIdx);
        findKeyPoints(bestIdx, end);
    }
}


std::string classifyContour(const std::vector<cv::Point>& contour_input) {
    if (contour_input.size() < 3) {
        return "unknown";
    }

    
    std::vector<Pt> contour;
    for (const auto& p : contour_input) {
        contour.push_back(Pt(p));
    }

    
    if (contour.size() >= 2) {
        double dx = contour.front().x - contour.back().x;
        double dy = contour.front().y - contour.back().y;
        if (dx * dx + dy * dy < 1e-6) {
            contour.pop_back();
        }
    }
    if (contour.size() < 3) {
        return "unknown";
    }

    
    global_contour = contour;
    global_isKey.assign(contour.size(), false);
    global_isKey[0] = true;
    global_isKey[contour.size() - 1] = true;

    
    findKeyPoints(0, static_cast<int>(contour.size()) - 1);

    
    std::vector<Pt> keyPts;
    for (size_t i = 0; i < contour.size(); ++i) {
        if (global_isKey[i]) {
            keyPts.push_back(contour[i]);
        }
    }

    if (keyPts.size() < 3) {
        return "unknown";
    }

    
    int numCorners = static_cast<int>(keyPts.size()) - 2;

    
    double totalLen = 0.0;
    for (size_t i = 1; i < keyPts.size(); ++i) {
        double dx = keyPts[i].x - keyPts[i - 1].x;
        double dy = keyPts[i].y - keyPts[i - 1].y;
        totalLen += std::sqrt(dx * dx + dy * dy);
    }

    double dx_span = keyPts.back().x - keyPts.front().x;
    double dy_span = keyPts.back().y - keyPts.front().y;
    double span = std::sqrt(dx_span * dx_span + dy_span * dy_span);
    double elongation = (span > 1e-9) ? totalLen / span : 1.0;

    
    if (numCorners >= 40) {
        return "tank";
    }
    else if (elongation > 10) {
        return "rocket";
    }
    else {
        return "bus";
    }
}
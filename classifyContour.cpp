#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <functional> // для std::function

std::string classifyContour(const std::vector<cv::Point>& contour_input) {
    if (contour_input.size() < 3) {
        return "unknown";
    }

    // Внутреннее представление точки с double-координатами
    struct Pt {
        double x, y;
        Pt(double x_, double y_) : x(x_), y(y_) {}
        Pt(const cv::Point& p) : x(static_cast<double>(p.x)), y(static_cast<double>(p.y)) {}
    };

    // Преобразуем входной контур
    std::vector<Pt> contour;
    contour.reserve(contour_input.size());
    for (const auto& p : contour_input) {
        contour.emplace_back(p);
    }

    // Убираем замыкающую точку, если есть
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

    // Порог для поиска характерных точек (по методу Протасова)
    const double EPS = 0.1;

    // Вспомогательные функции
    auto dist = [](const Pt& a, const Pt& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
        };

    auto distToLine = [&](const Pt& p, const Pt& a, const Pt& b) {
        double len = dist(a, b);
        if (len < 1e-9) return dist(p, a);
        double area = std::abs((b.x - a.x) * (a.y - p.y) - (a.x - p.x) * (b.y - a.y));
        return area / len;
        };

    // Центр масс (требуется по методу Протасова)
    Pt center(0, 0);
    for (const auto& p : contour) {
        center.x += p.x;
        center.y += p.y;
    }
    center.x /= contour.size();
    center.y /= contour.size();

    // Поиск характерных точек рекурсивно
    std::vector<bool> isKey(contour.size(), false);
    isKey[0] = true;
    isKey[contour.size() - 1] = true;

    std::function<void(int, int)> findKeyPoints = [&](int start, int end) {
        if (end <= start + 1) return;
        double baseLen = dist(contour[start], contour[end]);
        if (baseLen < 1e-9) return;

        double maxRatio = 0.0;
        int bestIdx = -1;
        for (int i = start + 1; i < end; ++i) {
            double d = distToLine(contour[i], contour[start], contour[end]);
            double ratio = d / baseLen;
            if (ratio > maxRatio) {
                maxRatio = ratio;
                bestIdx = i;
            }
        }

        if (maxRatio > EPS && bestIdx != -1) {
            isKey[bestIdx] = true;
            findKeyPoints(start, bestIdx);
            findKeyPoints(bestIdx, end);
        }
        };

    findKeyPoints(0, static_cast<int>(contour.size()) - 1);

    // Сбор ключевых точек
    std::vector<Pt> keyPts;
    for (size_t i = 0; i < contour.size(); ++i) {
        if (isKey[i]) {
            keyPts.push_back(contour[i]);
        }
    }

    if (keyPts.size() < 3) {
        return "unknown";
    }

    // Анализ формы
    int numCorners = static_cast<int>(keyPts.size()) - 2; // концевые точки не считаются углами

    // Вычисляем степень вытянутости
    double totalLen = 0.0;
    for (size_t i = 1; i < keyPts.size(); ++i) {
        totalLen += dist(keyPts[i - 1], keyPts[i]);
    }
    double span = dist(keyPts.front(), keyPts.back());
    double elongation = (span > 1e-9) ? totalLen / span : 1.0;

    // Классификация по методу, вдохновлённому Протасовым
    if (numCorners >= 40) {
        return "tank";      // много изломов — танк (башня + гусеницы)
    }
    if (numCorners <= 40) {
        if (elongation > 10) {
            return "rocket"; // сильно вытянут — ракета
        }
        else {
            return "car";    // компактный и широкий — машина
        }
    }
    return "tank"; // промежуточный случай — предполагаем танк
}
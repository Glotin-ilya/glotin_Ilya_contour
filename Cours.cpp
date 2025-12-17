#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

std::string classifyContour(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double peri = cv::arcLength(contour, true);
    double approxAccuracy = 0.02 * peri;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, approxAccuracy, true);
    int vertices = (int)approx.size();

    cv::Rect boundingRect = cv::boundingRect(contour);
    double rect_width = boundingRect.width;
    double rect_height = boundingRect.height;
    double aspect_ratio = rect_width / rect_height;

    if (vertices >= 8) {
        return "tank";
    }
    if (aspect_ratio < 0.4) {
        return "rocket";
    }
    if (1.0 / aspect_ratio < 0.4) { // т.е. aspect_ratio > 2.5
        return "car";
    }
    return "unknown";
}

int main(int argc, char* argv[]) {
    // Проверяем, передано ли имя файла
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению>" << std::endl;
        return -1;
    }

    std::string filename = argv[1]; // Имя файла — первый аргумент

    // Загружаем исходное изображение
    cv::Mat img_full = cv::imread(filename);
    if (img_full.empty()) {
        std::cout << "Не удалось загрузить изображение: " << filename << std::endl;
        return -1;
    }

    // Уменьшаем изображение (сохраняя пропорции)
    const int TARGET_WIDTH = 640;
    double scale = (double)TARGET_WIDTH / img_full.cols;
    cv::Mat img;
    cv::resize(img_full, img, cv::Size(), scale, scale, cv::INTER_AREA);

    // Преобразуем в оттенки серого и бинаризуем
    cv::Mat gray, bin;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin, 240, 255, cv::THRESH_BINARY_INV);

    // Находим внешние контуры
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cout << "Контуры не найдены" << std::endl;
        return 0;
    }

    // Находим контур с наибольшей площадью
    double maxArea = 0;
    int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) {
            maxArea = a;
            maxIdx = (int)i;
        }
    }

    // Распознаём и отображаем результат
    if (maxIdx != -1) {
        std::string name = classifyContour(contours[maxIdx]);
        cv::drawContours(img, contours, maxIdx, cv::Scalar(0, 255, 0), 2);

        // Надпись в левом верхнем углу
        cv::putText(img, name, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        std::cout << "Объект распознан как: " << name << std::endl;
    }

    cv::imshow("Результат", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
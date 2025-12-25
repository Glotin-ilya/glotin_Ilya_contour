#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <functional> // для std::function

std::string classifyContour(const std::vector<cv::Point>& contour_input);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat originalImage = cv::imread(imagePath);
    if (originalImage.empty()) {
        std::cout << "Не удалось загрузить изображение: " << imagePath << std::endl;
        return -1;
    }

    const int TARGET_WIDTH = 640;
    double scaleFactor = static_cast<double>(TARGET_WIDTH) / originalImage.cols;
    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);

    cv::Mat grayscaleImage;
    cv::cvtColor(resizedImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    cv::Mat binaryImage;
    cv::threshold(grayscaleImage, binaryImage, 240, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> foundContours;
    cv::findContours(binaryImage, foundContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (foundContours.empty()) {
        std::cout << "Контуры не найдены" << std::endl;
        return 0;
    }

    double largestArea = 0.0;
    int largestContourIndex = -1;
    for (size_t i = 0; i < foundContours.size(); ++i) {
        double currentArea = cv::contourArea(foundContours[i]);
        if (currentArea > largestArea) {
            largestArea = currentArea;
            largestContourIndex = static_cast<int>(i);
        }
    }

    if (largestContourIndex != -1) {
        std::string objectClass = classifyContour(foundContours[largestContourIndex]);
        cv::drawContours(resizedImage, foundContours, largestContourIndex, cv::Scalar(0, 255, 0), 2);
        cv::putText(resizedImage, objectClass, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        std::cout << "Объект распознан как: " << objectClass << std::endl;
    }

    cv::imshow("Результат", resizedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
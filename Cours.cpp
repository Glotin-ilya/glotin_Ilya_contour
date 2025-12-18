#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

std::string classifyContour(const std::vector<cv::Point>& contour) {
    double contourArea = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    double approximationAccuracy = 0.02 * perimeter;
    // Аппроксимируем контур ломаной линией
    std::vector<cv::Point> approximatedContour;
    cv::approxPolyDP(contour, approximatedContour, approximationAccuracy, true);
    // Считаем количество вершин у аппроксимированного контура
    int vertexCount = static_cast<int>(approximatedContour.size());
    cv::Rect boundingBox = cv::boundingRect(contour);
    double width = static_cast<double>(boundingBox.width);
    double height = static_cast<double>(boundingBox.height);
    double aspectRatio = width / height;

    // Если вершин много — скорее всего, это танк (округлая форма с деталями)
    if (vertexCount >= 8) {
        return "tank";
    }
    // Очень узкий и высокий объект — похож на ракету
    if (aspectRatio < 0.4) {
        return "rocket";
    }
    // Очень широкий и низкий объект — похож на машину
    if (aspectRatio > 2.5) {
        return "car";
    }
    // Если ни один признак не подошёл — неизвестный объект
    return "unknown";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению>" << std::endl;
        return -1;
    }

    
    std::string imagePath = argv[1];
    // Загружаю изображение с диска
    cv::Mat originalImage = cv::imread(imagePath);
    if (originalImage.empty()) {
        std::cout << "Не удалось загрузить изображение: " << imagePath << std::endl;
        return -1;
    }

    // Уменьшаем изображение до ширины (640 пикселей), сохраняя пропорции
    const int TARGET_WIDTH = 640;
    double scaleFactor = static_cast<double>(TARGET_WIDTH) / originalImage.cols;
    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);

    cv::Mat grayscaleImage;
    cv::cvtColor(resizedImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    // Бинаризуем: делаем чёрно-белое изображение (инвертированное — объекты белые на чёрном фоне)
    cv::Mat binaryImage;
    cv::threshold(grayscaleImage, binaryImage, 240, 255, cv::THRESH_BINARY_INV);

    // Находим все внешние контуры на бинарном изображении. Если их нет, то завершаем
    std::vector<std::vector<cv::Point>> foundContours;
    cv::findContours(binaryImage, foundContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (foundContours.empty()) {
        std::cout << "Контуры не найдены" << std::endl;
        return 0;
    }

    // Ищем самый большой контур (по площади)
    double largestArea = 0.0;
    int largestContourIndex = -1;
    for (size_t i = 0; i < foundContours.size(); ++i) {
        double currentArea = cv::contourArea(foundContours[i]);
        if (currentArea > largestArea) {
            largestArea = currentArea;
            largestContourIndex = static_cast<int>(i);
        }
    }

    // Если нашли подходящий контур — определяем его тип и рисуем на изображении
    if (largestContourIndex != -1) {
        std::string objectClass = classifyContour(foundContours[largestContourIndex]);
        cv::drawContours(resizedImage, foundContours, largestContourIndex, cv::Scalar(0, 255, 0), 2);

        // Выводим название объекта в левом верхнем углу изображения
        cv::putText(resizedImage, objectClass, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        std::cout << "Объект распознан как: " << objectClass << std::endl;
    }

    cv::imshow("Результат", resizedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
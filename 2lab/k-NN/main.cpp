#include "knn.h"
#include <iostream>

int main() {
    // Пример данных: [длина, ширина] -> метка класса (0 или 1)
    std::vector<std::vector<double>> X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 1.0}, {6.0, 5.0}, {7.0, 7.0}, {8.0, 6.0}};
    std::vector<int> y = {0, 0, 0, 1, 1, 1};

    KNN knn(3);
    knn.fit(X, y);

    // Тестовый образец
    std::vector<double> sample = {4.0, 3.0};
    int prediction = knn.predict(sample);

    std::cout << "Предсказываемый класс: " << prediction << std::endl;

    return 0;
}
#include "knn.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <limits>
#include <omp.h>

KNN::KNN(int k) : k(k) {}

void KNN::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    X_train = X;
    y_train = y;
}

double KNN::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    // Проверка размеров векторов
    if (a.size() != b.size()) {
        std::cerr << "Ошибка: размеры векторов не совпадают ("
                  << a.size() << " vs " << b.size() << ")" << std::endl;
        return std::numeric_limits<double>::max();
    }

    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}

int KNN::predict(const std::vector<double>& x) {
    size_t n = X_train.size();

    // Проверка: должно быть достаточно точек для классификации
    if (n == 0) {
        std::cerr << "Ошибка: нет обучающих данных" << std::endl;
        return -1;
    }

    // Проверка: k не должно превышать количество точек
    int k_actual = k;
    if (k > static_cast<int>(n)) {
        std::cerr << "Предупреждение: k=" << k << " больше количества точек n=" << n
                  << ", используем k=" << n << std::endl;
        k_actual = n;
    }

    std::vector<std::pair<double, int>> distances(n);

    // #pragma omp parallel for - распараллеливает вычисление расстояний
    // schedule(static) - статическое распределение итераций между потоками
    // каждый поток получает примерно равное количество итераций заранее
    // Эффективно когда время выполнения каждой итерации примерно одинаково
    // Здесь НЕ нужен reduction, т.к. каждый поток пишет в свой элемент distances[i]
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double dist = euclideanDistance(x, X_train[i]);
        distances[i] = std::make_pair(dist, y_train[i]);
    }

    // Сортируем по расстоянию (первые k_actual элементов)
    // partial_sort сложно распараллелить эффективно, поэтому не используем pragma omp здесь
    std::partial_sort(
        distances.begin(),
        distances.begin() + k_actual,
        distances.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );

    // Подсчитываем частоту меток среди k_actual ближайших соседей
    // k обычно маленькое (3-10), параллелить не имеет смысла
    std::map<int, int> freq;
    for (int i = 0; i < k_actual; ++i) {
        freq[distances[i].second]++;
    }

    // Находим метку с максимальной частотой
    int predicted_label = -1;
    int max_freq = 0;
    for (const auto& entry : freq) {
        if (entry.second > max_freq) {
            max_freq = entry.second;
            predicted_label = entry.first;
        }
    }

    return predicted_label;
}
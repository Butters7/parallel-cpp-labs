#include "knn.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <stdexcept>

KNN::KNN(int k) : k(k) {}

void KNN::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same size");
    }
    X_train = X;
    y_train = y;
}

int KNN::predict(const std::vector<double>& x) {
    const size_t n = X_train.size();
    if (n == 0) {
        throw std::runtime_error("Model not fitted");
    }

    const size_t dim = x.size();

    // Проверка размерности
    if (!X_train.empty() && X_train[0].size() != dim) {
        throw std::invalid_argument("Input dimension mismatch");
    }

    // k не может быть больше n
    const int k_actual = std::min(k, static_cast<int>(n));

    // Предвыделяем память для distances (каждый поток пишет в свой элемент)
    std::vector<std::pair<double, int>> distances(n);

    // Параллельное вычисление расстояний
    // schedule(dynamic) - т.к. euclideanDistance может занимать разное время
    // default(none) - явное указание всех переменных
    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(distances, X_train, y_train, x, n, dim)
    for (size_t i = 0; i < n; ++i) {
        const std::vector<double>& train_point = X_train[i];

        // Инлайн вычисление евклидова расстояния (избегаем вызова функции)
        double dist = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            double diff = x[j] - train_point[j];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        distances[i] = {dist, y_train[i]};
    }

    // Частичная сортировка - только первые k_actual элементов
    std::partial_sort(
        distances.begin(),
        distances.begin() + k_actual,
        distances.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        }
    );

    // Подсчёт частоты меток среди k ближайших соседей
    std::unordered_map<int, int> freq;
    for (int i = 0; i < k_actual; ++i) {
        freq[distances[i].second]++;
    }

    // Находим метку с максимальной частотой
    int predicted_label = distances[0].second;
    int max_freq = 0;
    for (const auto& entry : freq) {
        if (entry.second > max_freq) {
            max_freq = entry.second;
            predicted_label = entry.first;
        }
    }

    return predicted_label;
}

void KNN::setNumThreads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

int KNN::getNumThreads() const {
    return omp_get_max_threads();
}

#include "floyd_warshall.h"
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <thread>
#include <functional>
#include <exception>
#include <stdexcept>

using namespace std;

FloydWarshall::FloydWarshall(const vector<vector<int>>& graph) {
    n = static_cast<int>(graph.size());
    hasNegativeCycle = false;

    // Инициализируем плоские массивы
    dist.resize(n * n);
    next.resize(n * n, -1);

    // Параллельная инициализация матриц
    int numThreads = static_cast<int>(thread::hardware_concurrency());
    if (numThreads == 0) numThreads = 4;
    if (numThreads > n) numThreads = n;

    vector<thread> initThreads;
    initThreads.reserve(numThreads);

    for (int t = 0; t < numThreads; t++) {
        initThreads.emplace_back([&, t]() {
            // Равномерное распределение строк между потоками
            for (int i = t; i < n; i += numThreads) {
                for (int j = 0; j < n; j++) {
                    int idx = i * n + j;
                    dist[idx] = graph[i][j];
                    if (graph[i][j] != INT_MAX && i != j) {
                        next[idx] = j;
                    }
                }
            }
        });
    }

    for (auto& thr : initThreads) {
        thr.join();
    }
}

void FloydWarshall::run() {
    int numThreads = static_cast<int>(thread::hardware_concurrency());
    if (numThreads == 0) numThreads = 4;
    if (numThreads > n) numThreads = n;

    // Если граф слишком мал, используем один поток
    if (n <= 2) {
        numThreads = 1;
    }

    // Используем собственную реализацию барьера для совместимости
    Barrier sync_point(numThreads);

    // Для хранения исключений из потоков
    vector<exception_ptr> exceptions(numThreads);

    vector<thread> threads;
    threads.reserve(numThreads);

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            try {
                // Равномерное распределение строк (cyclic distribution)
                // Уменьшает дисбаланс нагрузки
                for (int k = 0; k < n; k++) {
                    // Кэшируем строку k для уменьшения доступа к памяти
                    // dist[k][j] нужен всем потокам, поэтому читаем его локально

                    // Каждый поток обрабатывает свои строки
                    for (int i = t; i < n; i += numThreads) {
                        int dist_ik = dist[i * n + k];

                        // Пропускаем если путь через k невозможен
                        if (dist_ik == INT_MAX) continue;

                        int base_i = i * n;
                        for (int j = 0; j < n; j++) {
                            int dist_kj = dist[k * n + j];

                            // Проверка на переполнение и бесконечность
                            if (dist_kj == INT_MAX) continue;

                            int new_dist = dist_ik + dist_kj;

                            // Проверка на переполнение при сложении
                            if (dist_ik > 0 && dist_kj > 0 && new_dist < 0) continue;

                            if (dist[base_i + j] > new_dist) {
                                dist[base_i + j] = new_dist;
                                next[base_i + j] = next[i * n + k];
                            }
                        }
                    }

                    // Синхронизация перед следующей итерацией k
                    sync_point.arrive_and_wait();
                }
            } catch (...) {
                exceptions[t] = current_exception();
            }
        });
    }

    // Ожидаем завершения всех потоков
    for (auto& thr : threads) {
        thr.join();
    }

    // Проверяем исключения
    for (const auto& ex : exceptions) {
        if (ex) {
            rethrow_exception(ex);
        }
    }

    // Параллельная проверка на отрицательные циклы
    atomic<bool> foundNegative{false};
    vector<thread> checkThreads;
    checkThreads.reserve(numThreads);

    for (int t = 0; t < numThreads; t++) {
        checkThreads.emplace_back([&, t]() {
            for (int i = t; i < n; i += numThreads) {
                if (dist[i * n + i] < 0) {
                    foundNegative = true;
                    return;  // Досрочный выход при обнаружении
                }
                // Проверяем флаг чтобы выйти раньше если другой поток нашёл
                if (foundNegative.load(memory_order_relaxed)) return;
            }
        });
    }

    for (auto& thr : checkThreads) {
        thr.join();
    }

    hasNegativeCycle = foundNegative.load();
}

int FloydWarshall::getDistance(int from, int to) const {
    if (from < 0 || from >= n || to < 0 || to >= n) {
        return INT_MAX;
    }
    return dist[from * n + to];
}

bool FloydWarshall::hasNegativeCycles() const {
    return hasNegativeCycle.load();
}

vector<int> FloydWarshall::getPath(int from, int to) const {
    vector<int> path;

    if (from < 0 || from >= n || to < 0 || to >= n) {
        return path;
    }

    if (dist[from * n + to] == INT_MAX) {
        return path;
    }

    int current = from;
    // Защита от бесконечного цикла (максимум n шагов)
    int steps = 0;
    while (current != to && steps < n) {
        path.push_back(current);
        int nextNode = next[current * n + to];
        if (nextNode == -1) {
            return vector<int>();  // Путь не найден
        }
        current = nextNode;
        steps++;
    }

    if (current == to) {
        path.push_back(to);
    } else {
        return vector<int>();  // Путь не найден (цикл?)
    }

    return path;
}

vector<vector<int>> FloydWarshall::getDistanceMatrix() const {
    // Конвертируем плоский массив обратно в 2D для совместимости
    vector<vector<int>> result(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = dist[i * n + j];
        }
    }
    return result;
}

void FloydWarshall::printMatrix() const {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = dist[i * n + j];
            if (val == INT_MAX) {
                cout << "INF\t";
            } else {
                cout << val << "\t";
            }
        }
        cout << endl;
    }
}

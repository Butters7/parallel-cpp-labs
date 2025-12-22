#include "floyd_warshall.h"
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <thread>
#include <barrier>
#include <functional>

using namespace std;

FloydWarshall::FloydWarshall(const vector<vector<int>>& graph) {
    n = graph.size();
    dist = graph;
    next.resize(n, vector<int>(n, -1));
    hasNegativeCycle = false;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i][j] != INT_MAX && i != j) {
                next[i][j] = j;
            }
        }
    }
}

void FloydWarshall::run() {
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > n) numThreads = n;

    // Используем барьер для синхронизации между итерациями
    // Барьер освобождается когда все потоки достигают его
    barrier sync_point(numThreads);

    // Создаём потоки один раз, а не на каждой итерации
    vector<thread> threads;

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            // Каждый поток знает свой диапазон строк
            int rowsPerThread = n / numThreads;
            int startRow = t * rowsPerThread;
            int endRow = (t == numThreads - 1) ? n : startRow + rowsPerThread;

            // Внешний цикл по k выполняется каждым потоком
            for (int k = 0; k < n; k++) {
                // Обрабатываем только свои строки
                // Используем блочное распределение вместо построчного
                // чтобы уменьшить false sharing
                for (int i = startRow; i < endRow; i++) {
                    // Кэшируем dist[i][k] чтобы избежать повторных обращений
                    int dist_ik = dist[i][k];
                    if (dist_ik != INT_MAX) {
                        for (int j = 0; j < n; j++) {
                            int dist_kj = dist[k][j];
                            if (dist_kj != INT_MAX) {
                                int new_dist = dist_ik + dist_kj;
                                if (dist[i][j] > new_dist) {
                                    dist[i][j] = new_dist;
                                    next[i][j] = next[i][k];
                                }
                            }
                        }
                    }
                }

                // Все потоки ждут здесь перед переходом к следующему k
                sync_point.arrive_and_wait();
            }
        });
    }

    // Ждём завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }

    // Проверка на отрицательные циклы
    for (int i = 0; i < n; i++) {
        if (dist[i][i] < 0) {
            hasNegativeCycle = true;
            break;
        }
    }
}

int FloydWarshall::getDistance(int from, int to) const {
    if (from < 0 || from >= n || to < 0 || to >= n) {
        return INT_MAX;
    }
    return dist[from][to];
}

bool FloydWarshall::hasNegativeCycles() const {
    return hasNegativeCycle;
}

vector<int> FloydWarshall::getPath(int from, int to) const {
    vector<int> path;

    if (from < 0 || from >= n || to < 0 || to >= n || dist[from][to] == INT_MAX) {
        return path;
    }

    int current = from;
    while (current != to) {
        path.push_back(current);
        current = next[current][to];
        if (current == -1) {
            return vector<int>();
        }
    }
    path.push_back(to);

    return path;
}

const vector<vector<int>>& FloydWarshall::getDistanceMatrix() const {
    return dist;
}

void FloydWarshall::printMatrix() const {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i][j] == INT_MAX) {
                cout << "INF\t";
            } else {
                cout << dist[i][j] << "\t";
            }
        }
        cout << endl;
    }
}

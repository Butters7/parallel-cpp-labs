#include "floyd_warshall.h"
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <thread>

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
    // Определяем количество потоков (по числу ядер процессора)
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > n) numThreads = n;

    // Внешний цикл по k - промежуточная вершина
    for (int k = 0; k < n; k++) {
        vector<thread> threads;

        // Создаём потоки для параллельной обработки строк матрицы
        // Каждый поток обрабатывает свой диапазон строк i
        for (int t = 0; t < numThreads; t++) {
            threads.emplace_back([&, t, k]() {
                // Вычисляем диапазон строк для этого потока
                int rowsPerThread = n / numThreads;
                int startRow = t * rowsPerThread;
                int endRow = (t == numThreads - 1) ? n : startRow + rowsPerThread;

                // Обрабатываем строки [startRow, endRow)
                // Параллелизм безопасен: разные потоки пишут в разные строки
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < n; j++) {
                        if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                            if (dist[i][j] > dist[i][k] + dist[k][j]) {
                                dist[i][j] = dist[i][k] + dist[k][j];
                                next[i][j] = next[i][k];
                            }
                        }
                    }
                }
            });
        }

        // Ждём завершения всех потоков перед следующей итерацией k
        for (auto& t : threads) {
            t.join();
        }
    }

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

#include "floyd_warshall.h"
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>

using namespace std;

// Барьер для синхронизации потоков между итерациями k
class Barrier {
    mutex mtx;
    condition_variable cv;
    int total;
    int waiting = 0;
    int generation = 0;
public:
    Barrier(int n) : total(n) {}

    void wait() {
        unique_lock<mutex> lock(mtx);
        int gen = generation;
        if (++waiting == total) {
            generation++;
            waiting = 0;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

FloydWarshall::FloydWarshall(const vector<vector<int>>& graph) {
    n = graph.size();
    dist = graph;
    next.resize(n, vector<int>(n, -1));
    hasNegativeCycle = false;

    // Инициализация next матрицы для восстановления пути
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
    if (numThreads < 1) numThreads = 1;

    Barrier barrier(numThreads);
    vector<thread> threads;
    vector<exception_ptr> exceptions(numThreads);

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([this, t, numThreads, &barrier, &exceptions]() {
            try {
                for (int k = 0; k < n; k++) {
                    for (int i = t; i < n; i += numThreads) {
                        if (dist[i][k] == INT_MAX) continue;

                        for (int j = 0; j < n; j++) {
                            if (dist[k][j] != INT_MAX) {
                                // Используем long long для предотвращения переполнения
                                long long newDist = (long long)dist[i][k] + dist[k][j];
                                if (newDist < dist[i][j]) {
                                    dist[i][j] = (int)newDist;
                                    next[i][j] = next[i][k];
                                }
                            }
                        }
                    }
                    barrier.wait();
                }
            } catch (...) {
                exceptions[t] = current_exception();
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Проверяем исключения из потоков
    for (auto& ex : exceptions) {
        if (ex) {
            rethrow_exception(ex);
        }
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
        return path; // Пустой путь
    }

    int current = from;
    while (current != to) {
        path.push_back(current);
        current = next[current][to];
        if (current == -1) {
            return vector<int>(); // Путь не существует
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

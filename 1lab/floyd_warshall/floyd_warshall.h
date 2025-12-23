#ifndef FLOYD_WARSHALL_H
#define FLOYD_WARSHALL_H

#include <vector>
#include <climits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

// Реализация барьера на основе condition_variable для совместимости
// с компиляторами без полной поддержки C++20 std::barrier
class Barrier {
private:
    std::mutex mtx;
    std::condition_variable cv;
    const int threshold;
    int count;
    int generation;

public:
    explicit Barrier(int numThreads)
        : threshold(numThreads), count(numThreads), generation(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        if (--count == 0) {
            generation++;
            count = threshold;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

class FloydWarshall {
private:
    // Используем плоский массив для лучшего использования кэша
    std::vector<int> dist;      // размер n*n
    std::vector<int> next;      // размер n*n
    int n;
    std::atomic<bool> hasNegativeCycle;

    // Вспомогательные функции для доступа к элементам плоского массива
    inline int& distAt(int i, int j) { return dist[i * n + j]; }
    inline int distAt(int i, int j) const { return dist[i * n + j]; }
    inline int& nextAt(int i, int j) { return next[i * n + j]; }
    inline int nextAt(int i, int j) const { return next[i * n + j]; }

public:
    FloydWarshall(const std::vector<std::vector<int>>& graph);

    void run();

    int getDistance(int from, int to) const;

    bool hasNegativeCycles() const;

    std::vector<int> getPath(int from, int to) const;

    std::vector<std::vector<int>> getDistanceMatrix() const;

    void printMatrix() const;
};

#endif

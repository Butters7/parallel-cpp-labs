#ifndef FLOYD_WARSHALL_H
#define FLOYD_WARSHALL_H

#include <vector>
#include <climits>
#include <thread>             // Для создания потоков

class FloydWarshall {
private:
    std::vector<std::vector<int>> dist;
    std::vector<std::vector<int>> next;
    int n;
    bool hasNegativeCycle;

public:
    FloydWarshall(const std::vector<std::vector<int>>& graph);

    void run();

    int getDistance(int from, int to) const;

    bool hasNegativeCycles() const;

    std::vector<int> getPath(int from, int to) const;

    const std::vector<std::vector<int>>& getDistanceMatrix() const;

    void printMatrix() const;
};

#endif

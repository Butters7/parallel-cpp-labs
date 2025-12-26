#ifndef KNN_H
#define KNN_H

#include <vector>

class KNN {
private:
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    int k;

public:
    KNN(int k = 3);

    // Обучение (запоминание данных)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    // Предсказание для одного образца
    int predict(const std::vector<double>& x);

    // Настройка количества потоков OpenMP
    void setNumThreads(int num_threads);
    int getNumThreads() const;
};

#endif // KNN_H

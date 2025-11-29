#ifndef KNN_H
#define KNN_H

#include <vector>

class KNN {
private:
    std::vector<std::vector<double>> X_train; // Обучающие признаки
    std::vector<int> y_train;                 // Метки классов
    int k;                                    // Количество соседей

    // Вычисление евклидова расстояния между двумя векторами
    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);

public:
    KNN(int k = 3);
    
    // Обучение (запоминание данных)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    
    // Предсказание для одного образца
    int predict(const std::vector<double>& x);
};

#endif // KNN_H
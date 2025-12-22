#ifndef SPAMFILTER_H
#define SPAMFILTER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>

class SpamFilter {
private:
    struct WordStats {
        int spamCount = 0;
        int hamCount = 0;
    };

    std::unordered_map<std::string, WordStats> vocabulary;
    std::atomic<int> totalSpam{0};
    std::atomic<int> totalHam{0};
    double alpha = 1.0;
    std::atomic<bool> is_trained{false};  // Флаг обученности модели

    std::vector<std::string> tokenize(const std::string& text);
    std::string toLower(const std::string& str);

public:
    SpamFilter(double smoothing = 1.0);

    // ВАЖНО: train() должен вызываться ОДИН РАЗ до любых вызовов classify()
    // Не потокобезопасен - не вызывайте параллельно с classify()
    void train(const std::string& spamFile, const std::string& hamFile);

    // classify() потокобезопасен после завершения train()
    // Можно вызывать параллельно из нескольких потоков
    bool classify(const std::string& email);

    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};

#endif
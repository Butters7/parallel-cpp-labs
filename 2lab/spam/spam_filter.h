#ifndef SPAMFILTER_H
#define SPAMFILTER_H

#include <string>
#include <vector>
#include <unordered_map>

class SpamFilter {
private:
    struct WordStats {
        int spamCount = 0;
        int hamCount = 0;
    };

    std::unordered_map<std::string, WordStats> vocabulary;
    int totalSpam = 0;
    int totalHam = 0;
    double alpha = 1.0;

    std::vector<std::string> tokenize(const std::string& text);
    std::string toLower(const std::string& str);

public:
    SpamFilter(double smoothing = 1.0);
    void train(const std::string& spamFile, const std::string& hamFile);
    bool classify(const std::string& email);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};

#endif
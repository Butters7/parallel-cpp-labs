#include "spam_filter.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <omp.h>  // OpenMP для параллельного программирования

SpamFilter::SpamFilter(double smoothing) : alpha(smoothing) {}

std::string SpamFilter::toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<std::string> SpamFilter::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (ss >> token) {
        token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
        token = toLower(token);
        if (!token.empty() && token.length() > 2) tokens.push_back(token);
    }
    
    return tokens;
}

void SpamFilter::train(const std::string& spamFile, const std::string& hamFile) {
    std::ifstream spamStream(spamFile);
    std::ifstream hamStream(hamFile);
    std::string line;
    
    while (std::getline(spamStream, line)) {
        auto tokens = tokenize(line);
        for (const auto& token : tokens) vocabulary[token].spamCount++;
        totalSpam++;
    }
    
    while (std::getline(hamStream, line)) {
        auto tokens = tokenize(line);
        for (const auto& token : tokens) vocabulary[token].hamCount++;
        totalHam++;
    }
}

bool SpamFilter::classify(const std::string& email) {
    auto tokens = tokenize(email);
    double spamLogProb = std::log(static_cast<double>(totalSpam) / (totalSpam + totalHam));
    double hamLogProb = std::log(static_cast<double>(totalHam) / (totalSpam + totalHam));
    int vocabSize = vocabulary.size();

    // Переменные для накопления логарифмов вероятностей
    double spamSum = 0.0;
    double hamSum = 0.0;

    // #pragma omp parallel for - распараллеливает цикл обработки токенов
    // reduction(+:spamSum, hamSum) - каждый поток накапливает свои частичные суммы,
    // в конце все суммы объединяются (избегаем гонки данных)
    // schedule(dynamic) - динамическое распределение итераций между потоками
    // например, поиск в vocabulary может занимать разное время)
    #pragma omp parallel for reduction(+:spamSum, hamSum) schedule(dynamic)
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        auto it = vocabulary.find(token);
        double spamProb = alpha;
        double hamProb = alpha;

        if (it != vocabulary.end()) {
            spamProb += it->second.spamCount;
            hamProb += it->second.hamCount;
        }

        // Накапливаем логарифмы вероятностей
        spamSum += std::log(spamProb / (totalSpam + alpha * vocabSize));
        hamSum += std::log(hamProb / (totalHam + alpha * vocabSize));
    }

    spamLogProb += spamSum;
    hamLogProb += hamSum;

    return spamLogProb > hamLogProb;
}

void SpamFilter::saveModel(const std::string& filename) {
    std::ofstream file(filename);
    file << totalSpam << " " << totalHam << " " << alpha << std::endl;
    for (const auto& [word, stats] : vocabulary) {
        file << word << " " << stats.spamCount << " " << stats.hamCount << std::endl;
    }
}

void SpamFilter::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    file >> totalSpam >> totalHam >> alpha;
    
    std::string word;
    int spamCount, hamCount;
    while (file >> word >> spamCount >> hamCount) {
        vocabulary[word] = {spamCount, hamCount};
    }
}
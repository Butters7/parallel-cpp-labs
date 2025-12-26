#include "spam_filter.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

SpamFilter::SpamFilter(double smoothing) : alpha(smoothing) {}

std::string SpamFilter::toLower(const std::string& str) const {
    std::string result = str;
    for (char& c : result) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

std::vector<std::string> SpamFilter::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream ss(text);
    std::string token;

    while (ss >> token) {
        // Удаляем пунктуацию
        token.erase(
            std::remove_if(token.begin(), token.end(),
                [](unsigned char c) { return std::ispunct(c); }),
            token.end()
        );
        token = toLower(token);
        if (!token.empty() && token.length() > 2) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

void SpamFilter::train(const std::string& spamFile, const std::string& hamFile) {
    // Читаем файлы последовательно
    std::ifstream spamStream(spamFile);
    std::ifstream hamStream(hamFile);

    if (!spamStream.is_open() || !hamStream.is_open()) {
        std::cerr << "Ошибка открытия файлов обучения" << std::endl;
        return;
    }

    std::vector<std::string> spamLines;
    std::vector<std::string> hamLines;
    std::string line;

    while (std::getline(spamStream, line)) {
        if (!line.empty()) spamLines.push_back(line);
    }
    while (std::getline(hamStream, line)) {
        if (!line.empty()) hamLines.push_back(line);
    }

    totalSpam = static_cast<int>(spamLines.size());
    totalHam = static_cast<int>(hamLines.size());

    // Очищаем словарь
    vocabulary.clear();

    // Локальные словари для каждого потока (избегаем race condition)
    const int maxThreads = omp_get_max_threads();
    std::vector<std::unordered_map<std::string, WordStats>> threadVocabs(maxThreads);

    // Параллельная обработка спам-сообщений
    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(spamLines, threadVocabs)
    for (size_t i = 0; i < spamLines.size(); ++i) {
        int tid = omp_get_thread_num();
        auto tokens = tokenize(spamLines[i]);
        for (const auto& token : tokens) {
            threadVocabs[tid][token].spamCount++;
        }
    }

    // Параллельная обработка ham-сообщений
    #pragma omp parallel for schedule(dynamic) default(none) \
        shared(hamLines, threadVocabs)
    for (size_t i = 0; i < hamLines.size(); ++i) {
        int tid = omp_get_thread_num();
        auto tokens = tokenize(hamLines[i]);
        for (const auto& token : tokens) {
            threadVocabs[tid][token].hamCount++;
        }
    }

    // Слияние локальных словарей (последовательно, безопасно)
    for (const auto& localVocab : threadVocabs) {
        for (const auto& entry : localVocab) {
            vocabulary[entry.first].spamCount += entry.second.spamCount;
            vocabulary[entry.first].hamCount += entry.second.hamCount;
        }
    }

    std::cout << "Обучение завершено: spam=" << totalSpam
              << ", ham=" << totalHam
              << ", словарь=" << vocabulary.size() << " слов" << std::endl;
}

bool SpamFilter::classify(const std::string& email) const {
    auto tokens = tokenize(email);

    if (tokens.empty() || totalSpam == 0 || totalHam == 0) {
        return false;
    }

    const double spamPrior = std::log(static_cast<double>(totalSpam) / (totalSpam + totalHam));
    const double hamPrior = std::log(static_cast<double>(totalHam) / (totalSpam + totalHam));
    const int vocabSize = static_cast<int>(vocabulary.size());
    const double smoothing = alpha;
    const int spamTotal = totalSpam;
    const int hamTotal = totalHam;

    double spamSum = 0.0;
    double hamSum = 0.0;

    // Параллельное вычисление логарифмов вероятностей
    // reduction - безопасное накопление сумм
    // schedule(static) - итерации примерно равны по времени
    #pragma omp parallel for reduction(+:spamSum, hamSum) schedule(static) \
        default(none) shared(tokens, vocabulary, vocabSize, smoothing, spamTotal, hamTotal)
    for (size_t i = 0; i < tokens.size(); ++i) {
        double spamProb = smoothing;
        double hamProb = smoothing;

        auto it = vocabulary.find(tokens[i]);
        if (it != vocabulary.end()) {
            spamProb += it->second.spamCount;
            hamProb += it->second.hamCount;
        }

        spamSum += std::log(spamProb / (spamTotal + smoothing * vocabSize));
        hamSum += std::log(hamProb / (hamTotal + smoothing * vocabSize));
    }

    return (spamPrior + spamSum) > (hamPrior + hamSum);
}

void SpamFilter::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка сохранения модели: " << filename << std::endl;
        return;
    }

    file << totalSpam << " " << totalHam << " " << alpha << "\n";
    for (const auto& entry : vocabulary) {
        file << entry.first << " "
             << entry.second.spamCount << " "
             << entry.second.hamCount << "\n";
    }

    std::cout << "Модель сохранена: " << filename << std::endl;
}

void SpamFilter::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка загрузки модели: " << filename << std::endl;
        return;
    }

    vocabulary.clear();
    file >> totalSpam >> totalHam >> alpha;

    std::string word;
    int spamCount, hamCount;
    while (file >> word >> spamCount >> hamCount) {
        vocabulary[word] = {spamCount, hamCount};
    }

    std::cout << "Модель загружена: spam=" << totalSpam
              << ", ham=" << totalHam
              << ", словарь=" << vocabulary.size() << " слов" << std::endl;
}

void SpamFilter::setNumThreads(int n) {
    if (n > 0) {
        omp_set_num_threads(n);
    }
}

int SpamFilter::getNumThreads() const {
    return omp_get_max_threads();
}

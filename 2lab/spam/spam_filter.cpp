#include "spam_filter.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
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

    // Очищаем предыдущую модель если была
    vocabulary.clear();
    totalSpam.store(0);
    totalHam.store(0);

    int local_spam_count = 0;
    int local_ham_count = 0;

    // Обучение на спаме
    while (std::getline(spamStream, line)) {
        auto tokens = tokenize(line);
        for (const auto& token : tokens) {
            vocabulary[token].spamCount++;
        }
        local_spam_count++;
    }

    // Обучение на хаме
    while (std::getline(hamStream, line)) {
        auto tokens = tokenize(line);
        for (const auto& token : tokens) {
            vocabulary[token].hamCount++;
        }
        local_ham_count++;
    }

    // Атомарное обновление счетчиков
    totalSpam.store(local_spam_count);
    totalHam.store(local_ham_count);

    // Помечаем модель как обученную
    is_trained.store(true);
}

bool SpamFilter::classify(const std::string& email) {
    // Проверка что модель обучена
    if (!is_trained.load()) {
        std::cerr << "Ошибка: модель не обучена. Вызовите train() перед classify()" << std::endl;
        return false;
    }

    auto tokens = tokenize(email);

    // Загружаем атомарные значения один раз для консистентности
    int spam_count = totalSpam.load();
    int ham_count = totalHam.load();

    if (spam_count == 0 && ham_count == 0) {
        std::cerr << "Ошибка: нет обучающих данных" << std::endl;
        return false;
    }

    double spamLogProb = std::log(static_cast<double>(spam_count) / (spam_count + ham_count));
    double hamLogProb = std::log(static_cast<double>(ham_count) / (spam_count + ham_count));
    int vocabSize = vocabulary.size();

    double spamSum = 0.0;
    double hamSum = 0.0;

    // #pragma omp parallel for - распараллеливает цикл обработки токенов
    // reduction(+:spamSum, hamSum) - каждый поток накапливает свои частичные суммы,
    // в конце все суммы объединяются (избегаем гонки данных)
    // schedule(dynamic) - динамическое распределение итераций между потоками
    // (поиск в vocabulary может занимать разное время)
    // ПОТОКОБЕЗОПАСНО: vocabulary используется только для чтения (после train())
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
        spamSum += std::log(spamProb / (spam_count + alpha * vocabSize));
        hamSum += std::log(hamProb / (ham_count + alpha * vocabSize));
    }

    spamLogProb += spamSum;
    hamLogProb += hamSum;

    return spamLogProb > hamLogProb;
}

void SpamFilter::saveModel(const std::string& filename) {
    std::ofstream file(filename);
    file << totalSpam.load() << " " << totalHam.load() << " " << alpha << std::endl;
    for (const auto& [word, stats] : vocabulary) {
        file << word << " " << stats.spamCount << " " << stats.hamCount << std::endl;
    }
}

void SpamFilter::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    int spam, ham;
    file >> spam >> ham >> alpha;

    totalSpam.store(spam);
    totalHam.store(ham);

    vocabulary.clear();
    std::string word;
    int spamCount, hamCount;
    while (file >> word >> spamCount >> hamCount) {
        vocabulary[word] = {spamCount, hamCount};
    }

    // Помечаем модель как обученную после загрузки
    is_trained.store(true);
}
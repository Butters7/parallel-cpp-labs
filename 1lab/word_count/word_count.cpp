#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <thread>
#include <exception>

// Функция для приведения строки к нижнему регистру
std::string toLower(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (unsigned char c : str) {
        result += static_cast<char>(std::tolower(c));
    }
    return result;
}

// Функция для очистки слова от знаков препинания
std::string cleanWord(const std::string& word) {
    if (word.empty()) return "";

    size_t start = 0;
    size_t end = word.size();

    while (start < end && std::ispunct(static_cast<unsigned char>(word[start]))) {
        start++;
    }
    while (end > start && std::ispunct(static_cast<unsigned char>(word[end - 1]))) {
        end--;
    }

    return word.substr(start, end - start);
}

// Парсинг CSV строки с учётом кавычек
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

// Извлекает поле prompt из CSV (4-й столбец, индекс 3)
std::string extractPrompt(const std::string& line) {
    auto fields = parseCSVLine(line);
    if (fields.size() >= 4) {
        return fields[3];
    }
    return "";
}

// Структура для хранения слова и его частоты
struct WordCount {
    std::string word;
    int count;
};

// Функция для сравнения (для сортировки по убыванию)
bool compareWordCount(const WordCount& a, const WordCount& b) {
    return a.count > b.count;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " <файл> [количество слов]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int topN = 20;  // TOP 20 по умолчанию согласно заданию

    if (argc > 2) {
        try {
            topN = std::stoi(argv[2]);
            if (topN <= 0) {
                std::cerr << "Количество слов должно быть положительным" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Ошибка парсинга числа: " << e.what() << std::endl;
            return 1;
        }
    }

    // Открываем файл
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return 1;
    }

    // Читаем все строки, извлекая поле prompt
    std::vector<std::string> lines;
    std::string line;

    // Пропускаем заголовок CSV
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::string prompt = extractPrompt(line);
        if (!prompt.empty()) {
            lines.push_back(prompt);
        }
    }
    file.close();

    if (lines.empty()) {
        std::cout << "Файл пуст или не содержит данных" << std::endl;
        return 0;
    }

    // Определяем количество потоков
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > static_cast<int>(lines.size())) {
        numThreads = static_cast<int>(lines.size());
    }

    // Локальные карты для каждого потока (избегаем race conditions)
    std::vector<std::unordered_map<std::string, int>> localMaps(numThreads);
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions(numThreads);

    // Блочное распределение строк (сбалансированная нагрузка)
    size_t linesPerThread = lines.size() / numThreads;
    size_t remainder = lines.size() % numThreads;

    for (int t = 0; t < numThreads; t++) {
        // Вычисляем диапазон для потока
        size_t startIdx = t * linesPerThread + std::min(static_cast<size_t>(t), remainder);
        size_t endIdx = startIdx + linesPerThread + (static_cast<size_t>(t) < remainder ? 1 : 0);

        threads.emplace_back([&lines, &localMaps, &exceptions, t, startIdx, endIdx]() {
            try {
                for (size_t i = startIdx; i < endIdx; i++) {
                    std::istringstream iss(lines[i]);
                    std::string word;
                    while (iss >> word) {
                        word = cleanWord(word);
                        word = toLower(word);
                        if (!word.empty()) {
                            localMaps[t][word]++;
                        }
                    }
                }
            } catch (...) {
                exceptions[t] = std::current_exception();
            }
        });
    }

    // Ждём завершения всех потоков
    for (auto& th : threads) {
        th.join();
    }

    // Проверяем исключения
    for (const auto& ex : exceptions) {
        if (ex) {
            std::rethrow_exception(ex);
        }
    }

    // Последовательное слияние карт (безопасно, нет race conditions)
    std::unordered_map<std::string, int> wordCountMap;
    for (const auto& localMap : localMaps) {
        for (const auto& pair : localMap) {
            wordCountMap[pair.first] += pair.second;
        }
    }

    // Преобразуем в вектор для сортировки
    std::vector<WordCount> wordCounts;
    wordCounts.reserve(wordCountMap.size());
    for (const auto& pair : wordCountMap) {
        wordCounts.push_back({pair.first, pair.second});
    }

    // Частичная сортировка (эффективнее для TOP-N)
    if (static_cast<size_t>(topN) < wordCounts.size()) {
        std::partial_sort(wordCounts.begin(), wordCounts.begin() + topN,
                          wordCounts.end(), compareWordCount);
    } else {
        std::sort(wordCounts.begin(), wordCounts.end(), compareWordCount);
    }

    // Выводим результаты
    std::cout << "Топ-" << topN << " самых частых слов в файле '" << filename << "':" << std::endl;
    std::cout << std::setw(20) << std::left << "Слово" << "Количество" << std::endl;
    std::cout << std::string(30, '-') << std::endl;

    int limit = std::min(topN, static_cast<int>(wordCounts.size()));
    for (int i = 0; i < limit; i++) {
        std::cout << std::setw(20) << std::left << wordCounts[i].word
                  << wordCounts[i].count << std::endl;
    }

    return 0;
}

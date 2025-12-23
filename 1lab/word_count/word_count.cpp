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
#include <mutex>
#include <atomic>
#include <stdexcept>
#ifdef HAS_TBB
#include <execution>
#endif

// Функция для приведения строки к нижнему регистру
// Оптимизировано: используем std::transform вместо конкатенации
std::string toLower(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    std::transform(str.begin(), str.end(), std::back_inserter(result),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Функция для очистки слова от знаков препинания
// Оптимизировано: используем find_if для поиска границ вместо циклов erase
std::string cleanWord(const std::string& word) {
    if (word.empty()) return word;

    // Находим первый не-пунктуационный символ
    auto start = std::find_if(word.begin(), word.end(),
                              [](unsigned char c) { return !std::ispunct(c); });

    // Находим последний не-пунктуационный символ
    auto end = std::find_if(word.rbegin(), word.rend(),
                            [](unsigned char c) { return !std::ispunct(c); }).base();

    if (start >= end) return "";

    return std::string(start, end);
}

// Функция для парсинга CSV строки с учётом кавычек
// Возвращает вектор полей
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            // Проверяем на экранированную кавычку ""
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

// Извлекает поле prompt из CSV строки (4-й столбец, индекс 3)
std::string extractPrompt(const std::string& line) {
    auto fields = parseCSVLine(line);
    if (fields.size() >= 4) {
        return fields[3];  // prompt - 4-й столбец
    }
    return "";
}

// Структура для хранения слова и его частоты
struct WordCount {
    std::string word;
    int count;
};

// Функция для сравнения двух WordCount объектов (для сортировки)
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
            std::cerr << "Ошибка парсинга количества слов: " << e.what() << std::endl;
            return 1;
        }
    }
    
    // Открываем файл
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return 1;
    }
    
    // Читаем все строки файла в вектор, извлекая только поле prompt
    std::vector<std::string> lines;
    std::string line;

    // Пропускаем заголовок CSV
    if (std::getline(file, line)) {
        // Заголовок пропущен
    }

    // Читаем данные и извлекаем поле prompt
    while (std::getline(file, line)) {
        std::string prompt = extractPrompt(line);
        if (!prompt.empty()) {
            lines.push_back(prompt);
        }
    }
    file.close();

    if (lines.empty()) {
        std::cout << "Файл пуст" << std::endl;
        return 0;
    }

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > lines.size()) numThreads = static_cast<unsigned int>(lines.size());

    // Используем пакетную обработку строк для уменьшения contention
    // Каждый поток обрабатывает пакет строк за раз вместо одной строки
    const size_t BATCH_SIZE = 64;
    std::atomic<size_t> nextBatch{0};
    std::vector<std::unordered_map<std::string, int>> localMaps(numThreads);

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (unsigned int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            // Каждый поток берёт пакет строк для обработки
            // Это уменьшает количество атомарных операций
            while (true) {
                size_t batchStart = nextBatch.fetch_add(BATCH_SIZE);
                if (batchStart >= lines.size()) break;

                size_t batchEnd = std::min(batchStart + BATCH_SIZE, lines.size());
                for (size_t lineIndex = batchStart; lineIndex < batchEnd; ++lineIndex) {
                    std::istringstream iss(lines[lineIndex]);
                    std::string word;
                    while (iss >> word) {
                        word = cleanWord(word);
                        word = toLower(word);
                        if (!word.empty()) {
                            localMaps[t][word]++;
                        }
                    }
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Параллельное объединение результатов
    // Используем иерархическое слияние с переиспользованием потоков
    std::unordered_map<std::string, int> wordCountMap;

    // Проверка на пустой результат
    if (numThreads == 0) {
        std::cout << "Нет данных для обработки" << std::endl;
        return 0;
    }

    // Сливаем пары карт параллельно с использованием существующих потоков
    size_t numMaps = numThreads;
    while (numMaps > 1) {
        size_t pairs = numMaps / 2;
        std::vector<std::thread> mergeThreads;
        mergeThreads.reserve(pairs);

        for (size_t i = 0; i < pairs; i++) {
            size_t leftIdx = 2 * i;
            size_t rightIdx = 2 * i + 1;

            // Проверяем, что правый индекс существует
            if (rightIdx < localMaps.size() && !localMaps[rightIdx].empty()) {
                mergeThreads.emplace_back([&localMaps, leftIdx, rightIdx]() {
                    // Резервируем место для слияния
                    localMaps[leftIdx].reserve(localMaps[leftIdx].size() + localMaps[rightIdx].size());
                    for (auto& pair : localMaps[rightIdx]) {
                        localMaps[leftIdx][pair.first] += pair.second;
                    }
                    // Освобождаем память через swap с пустым контейнером
                    std::unordered_map<std::string, int>().swap(localMaps[rightIdx]);
                });
            }
        }

        for (auto& t : mergeThreads) {
            t.join();
        }

        // Перемещаем объединённые карты в начало вектора
        size_t writeIdx = 0;
        for (size_t i = 0; i < numMaps; i += 2) {
            if (writeIdx != i) {
                localMaps[writeIdx] = std::move(localMaps[i]);
            }
            writeIdx++;
        }
        // Учитываем нечётную карту, если она есть
        if (numMaps % 2 == 1) {
            if (writeIdx != numMaps - 1) {
                localMaps[writeIdx] = std::move(localMaps[numMaps - 1]);
            }
            writeIdx++;
        }
        numMaps = writeIdx;
    }

    // Финальная карта
    if (!localMaps.empty()) {
        wordCountMap = std::move(localMaps[0]);
    }
    
    // Преобразуем map в вектор для сортировки
    std::vector<WordCount> wordCounts;
    wordCounts.reserve(wordCountMap.size());
    for (const auto& pair : wordCountMap) {
        wordCounts.push_back({pair.first, pair.second});
    }

    // Сортируем вектор по убыванию частоты
#ifdef HAS_TBB
    // Используем параллельную политику выполнения, если TBB доступен
    std::sort(std::execution::par, wordCounts.begin(), wordCounts.end(), compareWordCount);
#else
    std::sort(wordCounts.begin(), wordCounts.end(), compareWordCount);
#endif
    
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

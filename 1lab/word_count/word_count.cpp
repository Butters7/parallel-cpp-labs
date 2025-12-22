#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>

// Функция для приведения строки к нижнему регистру
std::string toLower(const std::string& str) {
    std::string result;
    for (char c : str) {
        result += std::tolower(c);
    }
    return result;
}

// Функция для очистки слова от знаков препинания
std::string cleanWord(std::string word) {
    // Удаляем знаки препинания в начале слова
    while (!word.empty() && std::ispunct(word.front())) {
        word.erase(0, 1);
    }
    
    // Удаляем знаки препинания в конце слова
    while (!word.empty() && std::ispunct(word.back())) {
        word.pop_back();
    }
    
    return word;
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
    int topN = 10;
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
    
    // Читаем все строки файла в вектор
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    if (lines.empty()) {
        std::cout << "Файл пуст" << std::endl;
        return 0;
    }

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    if (numThreads > static_cast<int>(lines.size())) numThreads = lines.size();

    // Используем динамическую балансировку нагрузки
    std::atomic<size_t> nextLine{0};
    std::vector<std::map<std::string, int>> localMaps(numThreads);

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            // Каждый поток берет следующую доступную строку
            // Это обеспечивает лучшую балансировку нагрузки
            while (true) {
                size_t lineIndex = nextLine.fetch_add(1);
                if (lineIndex >= lines.size()) break;

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
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Параллельное объединение результатов
    // Используем иерархическое слияние
    std::map<std::string, int> wordCountMap;

    // Сначала сливаем пары карт параллельно
    int numMaps = numThreads;
    while (numMaps > 1) {
        std::vector<std::thread> mergeThreads;
        int pairs = numMaps / 2;

        for (int i = 0; i < pairs; i++) {
            mergeThreads.emplace_back([&localMaps, i]() {
                for (const auto& pair : localMaps[2*i + 1]) {
                    localMaps[2*i][pair.first] += pair.second;
                }
                localMaps[2*i + 1].clear();
            });
        }

        for (auto& t : mergeThreads) {
            t.join();
        }

        numMaps = pairs + (numMaps % 2);
    }

    // Финальная карта
    wordCountMap = std::move(localMaps[0]);
    
    // Преобразуем map в вектор для сортировки
    std::vector<WordCount> wordCounts;
    for (const auto& pair : wordCountMap) {
        wordCounts.push_back({pair.first, pair.second});
    }
    
    // Сортируем вектор по убыванию частоты
    std::sort(wordCounts.begin(), wordCounts.end(), compareWordCount);
    
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <thread>  // Для создания потоков
#include <mutex>   // Для синхронизации доступа к общим данным

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
    // Проверяем аргументы командной строки
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " <файл> [количество слов]" << std::endl;
        return 1;
    }
    
    // Получаем имя файла и количество слов для вывода
    std::string filename = argv[1];
    int topN = 10;
    if (argc > 2) {
        topN = std::stoi(argv[2]);
    }
    
    // Открываем файл
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return 1;
    }
    
    // Читаем все строки файла в вектор для параллельной обработки
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Определяем количество потоков
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    // Так как size_t конвертим в int без неявных преобразований
    if (numThreads > static_cast<int>(lines.size())) numThreads = lines.size();

    // Вектор локальных карт — каждый поток заполняет свою карту
    std::vector<std::map<std::string, int>> localMaps(numThreads);

    // Создаём потоки для параллельного подсчёта слов
    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; t++) {
        // emplace_back создаёт поток прямо в векторе
        // Лямбда-функция [&, t]() — анонимная функция, выполняемая потоком
        // [&, t] — захват переменных: & — все по ссылке, t — по значению (копия)
        // t копируем, чтобы каждый поток имел свой номер (иначе все увидят последнее значение t)
        threads.emplace_back([&, t]() {
            // Вычисляем диапазон строк для этого потока
            int linesPerThread = lines.size() / numThreads;
            int startLine = t * linesPerThread;
            int endLine = (t == numThreads - 1) ? lines.size() : startLine + linesPerThread;

            // Обрабатываем свои строки, заполняем локальную карту
            for (int i = startLine; i < endLine; i++) {
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
        });
    }

    // Ждём завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }

    // Объединяем результаты из всех локальных карт в одну общую
    std::map<std::string, int> wordCountMap;
    for (const auto& localMap : localMaps) {
        for (const auto& pair : localMap) {
            wordCountMap[pair.first] += pair.second;
        }
    }
    
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

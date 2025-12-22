#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <filesystem>  // Для обхода файловой системы (C++17)
#include <omp.h>       // OpenMP для параллельной обработки

namespace fs = std::filesystem;

// Структура для хранения информации о найденной уязвимости
struct SecurityIssue {
    std::string type;
    std::string description;
    std::string filename;     // Имя файла
    int line_number;
    std::string code_snippet;
};

// Структура для представления токена кода
struct CodeToken {
    std::string content;
    std::string type;  // "keyword", "function", "variable", "string", "number"
    int line_number;
};

class SASTAnalyzer {
private:
    std::vector<CodeToken> tokens;
    std::vector<SecurityIssue> issues;
    std::string currentFile;  // Текущий анализируемый файл

    // Паттерны для поиска уязвимостей
    std::vector<std::string> dangerous_functions = {
        "strcpy", "strcat", "gets", "sprintf", "scanf",
        "system", "popen", "exec", "malloc", "free", "new", "delete"
    };

    std::vector<std::string> sensitive_keywords = {
        "password", "secret", "key", "token", "auth"
    };

    // Расширения файлов для анализа
    std::vector<std::string> source_extensions = {".c", ".cpp", ".h", ".hpp"};

public:
    // Упрощенный токенизатор кода
    void tokenize(const std::vector<std::string>& source_lines) {
        tokens.clear();
        
        for (int i = 0; i < source_lines.size(); ++i) {
            std::string line = source_lines[i];
            std::string current_token;
            
            for (char c : line) {
                if (std::isspace(c) || std::ispunct(c)) {
                    if (!current_token.empty()) {
                        CodeToken token;
                        token.content = current_token;
                        token.line_number = i + 1;
                        
                        // Простая классификация токенов
                        if (std::find(dangerous_functions.begin(), 
                                    dangerous_functions.end(), 
                                    current_token) != dangerous_functions.end()) {
                            token.type = "dangerous_function";
                        } else if (std::find(sensitive_keywords.begin(), 
                                          sensitive_keywords.end(), 
                                          current_token) != sensitive_keywords.end()) {
                            token.type = "sensitive_data";
                        } else if (current_token.find("pass") != std::string::npos) {
                            token.type = "potential_credential";
                        } else {
                            token.type = "identifier";
                        }
                        
                        tokens.push_back(token);
                        current_token.clear();
                    }
                } else {
                    current_token += c;
                }
            }
            
            // Добавляем последний токен в строке, если он есть
            if (!current_token.empty()) {
                CodeToken token;
                token.content = current_token;
                token.line_number = i + 1;
                
                if (std::find(dangerous_functions.begin(), 
                            dangerous_functions.end(), 
                            current_token) != dangerous_functions.end()) {
                    token.type = "dangerous_function";
                } else if (std::find(sensitive_keywords.begin(), 
                                  sensitive_keywords.end(), 
                                  current_token) != sensitive_keywords.end()) {
                    token.type = "sensitive_data";
                } else if (current_token.find("pass") != std::string::npos) {
                    token.type = "potential_credential";
                } else {
                    token.type = "identifier";
                }
                
                tokens.push_back(token);
            }
        }
    }

    // Анализ буферных переполнений
    void analyze_buffer_overflows() {
        for (int i = 0; i < tokens.size(); ++i) {
            if (tokens[i].content == "strcpy" || tokens[i].content == "strcat") {
                SecurityIssue issue;
                issue.type = "BUFFER_OVERFLOW";
                issue.description = "Использование небезопасной функции: " + tokens[i].content;
                issue.line_number = tokens[i].line_number;
                issue.code_snippet = "Обнаружена потенциально опасная операция копирования";
                
                issues.push_back(issue);
            }
        }
    }

    // Анализ утечек памяти
    void analyze_memory_leaks() {
        std::vector<int> malloc_lines;
        std::vector<int> free_lines;
        
        // Собираем вызовы malloc и free
        for (int i = 0; i < tokens.size(); ++i) {
            if (tokens[i].content == "malloc") {
                malloc_lines.push_back(tokens[i].line_number);
            } else if (tokens[i].content == "free") {
                free_lines.push_back(tokens[i].line_number);
            }
        }
        
        // Ищем несопоставленные вызовы malloc
        for (int i = 0; i < malloc_lines.size(); ++i) {
            bool has_corresponding_free = false;
            
            for (int free_line : free_lines) {
                if (free_line > malloc_lines[i]) {
                    has_corresponding_free = true;
                    break;
                }
            }
            
            if (!has_corresponding_free) {
                SecurityIssue issue;
                issue.type = "MEMORY_LEAK";
                issue.description = "Возможная утечка памяти: malloc без соответствующего free";
                issue.line_number = malloc_lines[i];
                issue.code_snippet = "Память выделена, но не освобождена";
                
                issues.push_back(issue);
            }
        }
    }

    // Анализ чувствительных данных
    void analyze_sensitive_data() {
        for (int i = 0; i < tokens.size(); ++i) {
            if (tokens[i].type == "sensitive_data" || 
                tokens[i].type == "potential_credential") {
                
                SecurityIssue issue;
                issue.type = "SENSITIVE_DATA_EXPOSURE";
                issue.description = "Обнаружены чувствительные данные: " + tokens[i].content;
                issue.line_number = tokens[i].line_number;
                issue.code_snippet = "Потенциальное раскрытие конфиденциальной информации";
                
                issues.push_back(issue);
            }
        }
    }

    // Анализ командной инъекции
    void analyze_command_injection() {
        for (int i = 0; i < tokens.size(); ++i) {
            if (tokens[i].content == "system" || tokens[i].content == "popen") {
                SecurityIssue issue;
                issue.type = "COMMAND_INJECTION";
                issue.description = "Использование системных команд: " + tokens[i].content;
                issue.line_number = tokens[i].line_number;
                issue.code_snippet = "Возможность инъекции команд через пользовательский ввод";
                
                issues.push_back(issue);
            }
        }
    }

    // Анализ форматных строк
    void analyze_format_strings() {
        for (int i = 0; i < tokens.size(); ++i) {
            if (tokens[i].content == "printf" || tokens[i].content == "sprintf") {
                SecurityIssue issue;
                issue.type = "FORMAT_STRING";
                issue.description = "Использование форматных строк: " + tokens[i].content;
                issue.line_number = tokens[i].line_number;
                issue.code_snippet = "Потенциальная уязвимость форматной строки";
                
                issues.push_back(issue);
            }
        }
    }

    // Основной метод анализа
    void analyze(const std::vector<std::string>& source_code) {
        std::cout << "Начало SAST анализа..." << std::endl;
        
        // Токенизация кода
        tokenize(source_code);
        std::cout << "Токенизация завершена. Найдено токенов: " << tokens.size() << std::endl;
        
        // Последовательный анализ различных типов уязвимостей
        analyze_buffer_overflows();
        analyze_memory_leaks();
        analyze_sensitive_data();
        analyze_command_injection();
        analyze_format_strings();
        
        std::cout << "Анализ завершен. Найдено проблем: " << issues.size() << std::endl;
    }

    // Вывод результатов
    void print_results() {
        std::cout << "\n=== РЕЗУЛЬТАТЫ SAST АНАЛИЗА ===" << std::endl;

        // Группируем проблемы по типу для лучшей читаемости
        std::unordered_map<std::string, std::vector<SecurityIssue>> grouped_issues;

        for (const auto& issue : issues) {
            grouped_issues[issue.type].push_back(issue);
        }

        for (const auto& group : grouped_issues) {
            std::cout << "\n--- " << group.first << " (" << group.second.size() << " найденных) ---" << std::endl;

            // Выводим только первые 10 для каждого типа
            int count = 0;
            for (const auto& issue : group.second) {
                if (count++ >= 10) {
                    std::cout << "   ... и ещё " << (group.second.size() - 10) << std::endl;
                    break;
                }
                std::cout << issue.filename << ":" << issue.line_number << ": " << issue.description << std::endl;
            }
        }

        // Сводная статистика
        std::cout << "\n=== СВОДНАЯ СТАТИСТИКА ===" << std::endl;
        std::cout << "Всего проблем безопасности: " << issues.size() << std::endl;

        for (const auto& group : grouped_issues) {
            std::cout << group.first << ": " << group.second.size() << std::endl;
        }
    }

    // Проверка расширения файла
    bool isSourceFile(const std::string& filename) {
        for (const auto& ext : source_extensions) {
            if (filename.size() >= ext.size() &&
                filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
                return true;
            }
        }
        return false;
    }

    // Сбор всех исходных файлов из директории
    // Вызывается ОДИН РАЗ до параллельного анализа - потокобезопасность не требуется
    std::vector<std::string> collectSourceFiles(const std::string& directory) {
        std::vector<std::string> files;

        try {
            // Обход директорий может выбросить исключение если нет прав доступа
            for (const auto& entry : fs::recursive_directory_iterator(
                directory,
                fs::directory_options::skip_permission_denied)) {

                try {
                    if (entry.is_regular_file() && isSourceFile(entry.path().string())) {
                        files.push_back(entry.path().string());
                    }
                } catch (const fs::filesystem_error& e) {
                    // Пропускаем файлы с ошибками доступа
                    std::cerr << "Предупреждение: " << e.what() << std::endl;
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Ошибка при обходе директории: " << e.what() << std::endl;
        }

        return files;
    }

    // Чтение файла в вектор строк
    // ПОТОКОБЕЗОПАСНО: каждый вызов создает свой локальный ifstream
    std::vector<std::string> readFile(const std::string& filename) {
        std::vector<std::string> lines;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Предупреждение: не удалось открыть файл " << filename << std::endl;
            return lines;
        }

        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        return lines;
    }

    // Анализ директории с исходным кодом (параллельная версия)
    void analyzeDirectory(const std::string& directory) {
        std::cout << "Сканирование директории: " << directory << std::endl;

        // Собираем все файлы для анализа (последовательно, один раз)
        std::vector<std::string> files = collectSourceFiles(directory);
        std::cout << "Найдено файлов для анализа: " << files.size() << std::endl;

        if (files.empty()) {
            std::cout << "Нет файлов для анализа" << std::endl;
            return;
        }

        // Вектор для сбора результатов из всех потоков
        // Предварительно выделяем память для максимального числа потоков
        std::vector<std::vector<SecurityIssue>> threadResults(omp_get_max_threads());

        // #pragma omp parallel for - распараллеливает обработку файлов
        // schedule(dynamic) - динамическое распределение файлов между потоками
        // (файлы разного размера - динамическое распределение эффективнее)
        // ПОТОКОБЕЗОПАСНОСТЬ: каждый поток пишет в свой вектор threadResults[threadId]
        // чтобы избежать синхронизации при записи результатов
        // Каждый файл обрабатывается ровно одним потоком (нет коллизий)
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < files.size(); ++i) {
            int threadId = omp_get_thread_num();
            const std::string& file = files[i];

            // Читаем файл (каждый поток создает свой локальный ifstream - потокобезопасно)
            std::vector<std::string> lines = readFile(file);

            if (lines.empty()) {
                continue;  // Пропускаем пустые или недоступные файлы
            }

            // Анализируем каждую строку на наличие опасных паттернов
            for (size_t lineNum = 0; lineNum < lines.size(); ++lineNum) {
                const std::string& line = lines[lineNum];

                // Проверяем опасные функции
                for (const auto& func : dangerous_functions) {
                    if (line.find(func) != std::string::npos) {
                        SecurityIssue issue;
                        issue.filename = file;
                        issue.line_number = lineNum + 1;

                        // Определяем тип уязвимости
                        if (func == "strcpy" || func == "strcat" || func == "gets" || func == "sprintf") {
                            issue.type = "BUFFER_OVERFLOW";
                            issue.description = "Небезопасная функция: " + func;
                        } else if (func == "malloc") {
                            issue.type = "MEMORY_LEAK";
                            issue.description = "malloc без проверки free в функции";
                        } else if (func == "new") {
                            issue.type = "MEMORY_LEAK";
                            issue.description = "new без проверки delete в функции";
                        } else if (func == "system" || func == "popen" || func == "exec") {
                            issue.type = "COMMAND_INJECTION";
                            issue.description = "Потенциальная командная инъекция: " + func;
                        } else if (func == "scanf") {
                            issue.type = "FORMAT_STRING";
                            issue.description = "Небезопасный ввод: " + func;
                        }

                        issue.code_snippet = line.substr(0, std::min(line.size(), (size_t)80));
                        threadResults[threadId].push_back(issue);
                    }
                }
            }
        }

        // Объединяем результаты из всех потоков (последовательно, но быстро)
        for (const auto& results : threadResults) {
            issues.insert(issues.end(), results.begin(), results.end());
        }

        std::cout << "Анализ завершён. Найдено проблем: " << issues.size() << std::endl;
    }

    // Получение всех найденных проблем
    const std::vector<SecurityIssue>& get_issues() const {
        return issues;
    }
};

int main(int argc, char* argv[]) {
    // Путь к директории wireshark (можно передать как аргумент)
    std::string directory = "wireshark-wireshark-4.4.9";

    if (argc > 1) {
        directory = argv[1];
    }

    SASTAnalyzer analyzer;

    // Запуск параллельного анализа директории
    analyzer.analyzeDirectory(directory);
    analyzer.print_results();

    return 0;
}
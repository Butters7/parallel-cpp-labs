#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

struct SecurityIssue {
    std::string file;
    int line;
    std::string type;
    std::string pattern;
    std::string snippet;
};

class SASTAnalyzer {
private:
    struct Pattern {
        std::string name;
        std::string type;
    };

    std::vector<Pattern> patterns = {
        {"strcpy",   "BUFFER_OVERFLOW"},
        {"strcat",   "BUFFER_OVERFLOW"},
        {"gets",     "BUFFER_OVERFLOW"},
        {"sprintf",  "FORMAT_STRING"},
        {"vsprintf", "FORMAT_STRING"},
        {"scanf",    "FORMAT_STRING"},
        {"sscanf",   "FORMAT_STRING"},
        {"system",   "COMMAND_INJECTION"},
        {"popen",    "COMMAND_INJECTION"},
        {"execl",    "COMMAND_INJECTION"},
        {"execlp",   "COMMAND_INJECTION"},
        {"execv",    "COMMAND_INJECTION"},
        {"execvp",   "COMMAND_INJECTION"},
        {"malloc",   "MEMORY_ALLOC"},
        {"calloc",   "MEMORY_ALLOC"},
        {"realloc",  "MEMORY_ALLOC"},
        {"free",     "MEMORY_FREE"},
    };

    std::vector<std::string> extensions = {".c", ".cpp", ".h", ".hpp"};

    bool hasExtension(const std::string& path) const {
        for (const auto& ext : extensions) {
            if (path.size() >= ext.size() &&
                path.compare(path.size() - ext.size(), ext.size(), ext) == 0) {
                return true;
            }
        }
        return false;
    }

    // Быстрый поиск паттерна (вместо regex)
    bool findPattern(const std::string& line, const std::string& pattern) const {
        size_t pos = line.find(pattern);
        if (pos == std::string::npos) return false;

        // Проверяем что это не часть другого слова
        if (pos > 0 && (std::isalnum(line[pos-1]) || line[pos-1] == '_')) {
            return false;
        }
        size_t endPos = pos + pattern.size();
        if (endPos < line.size() && (std::isalnum(line[endPos]) || line[endPos] == '_')) {
            return false;
        }
        return true;
    }

    std::vector<SecurityIssue> analyzeFile(const std::string& filepath) const {
        std::vector<SecurityIssue> issues;
        std::ifstream file(filepath);
        if (!file.is_open()) return issues;

        std::string line;
        int lineNum = 0;
        int mallocCount = 0, freeCount = 0;

        while (std::getline(file, line)) {
            lineNum++;

            for (const auto& p : patterns) {
                if (findPattern(line, p.name)) {
                    if (p.type == "MEMORY_ALLOC") {
                        mallocCount++;
                    } else if (p.type == "MEMORY_FREE") {
                        freeCount++;
                    } else {
                        SecurityIssue issue;
                        issue.file = filepath;
                        issue.line = lineNum;
                        issue.type = p.type;
                        issue.pattern = p.name;
                        issue.snippet = line.substr(0, std::min(line.size(), size_t(80)));
                        issues.push_back(issue);
                    }
                }
            }
        }

        // Проверка malloc/free
        if (mallocCount > freeCount) {
            SecurityIssue leak;
            leak.file = filepath;
            leak.line = 0;
            leak.type = "MEMORY_LEAK";
            leak.pattern = "malloc/free";
            leak.snippet = "malloc: " + std::to_string(mallocCount) +
                          ", free: " + std::to_string(freeCount);
            issues.push_back(leak);
        }

        return issues;
    }

public:
    std::vector<std::string> collectFiles(const std::string& directory) const {
        std::vector<std::string> files;
        try {
            for (const auto& entry : fs::recursive_directory_iterator(
                    directory, fs::directory_options::skip_permission_denied)) {
                if (entry.is_regular_file() && hasExtension(entry.path().string())) {
                    files.push_back(entry.path().string());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Ошибка: " << e.what() << std::endl;
        }
        return files;
    }

    std::vector<SecurityIssue> analyzeDirectory(const std::string& directory) {
        std::cout << "Сканирование: " << directory << std::endl;

        auto files = collectFiles(directory);
        std::cout << "Найдено файлов: " << files.size() << std::endl;

        if (files.empty()) return {};

        const int maxThreads = omp_get_max_threads();
        std::vector<std::vector<SecurityIssue>> threadResults(maxThreads);

        #pragma omp parallel for schedule(dynamic) default(none) \
            shared(files, threadResults)
        for (size_t i = 0; i < files.size(); ++i) {
            int tid = omp_get_thread_num();
            auto fileIssues = analyzeFile(files[i]);
            for (auto& issue : fileIssues) {
                threadResults[tid].push_back(std::move(issue));
            }
        }

        std::vector<SecurityIssue> allIssues;
        for (const auto& results : threadResults) {
            allIssues.insert(allIssues.end(), results.begin(), results.end());
        }

        return allIssues;
    }

    void printResults(const std::vector<SecurityIssue>& issues) const {
        std::cout << "\n=== РЕЗУЛЬТАТЫ SAST ===" << std::endl;
        std::cout << "Всего проблем: " << issues.size() << std::endl;

        std::unordered_map<std::string, int> stats;
        for (const auto& issue : issues) {
            stats[issue.type]++;
        }

        std::cout << "\n=== СТАТИСТИКА ===" << std::endl;
        for (const auto& s : stats) {
            std::cout << s.first << ": " << s.second << std::endl;
        }

        // Примеры по типам
        std::unordered_map<std::string, int> shown;
        std::cout << "\n=== ПРИМЕРЫ ===" << std::endl;
        for (const auto& issue : issues) {
            if (shown[issue.type]++ < 3) {
                std::cout << "[" << issue.type << "] " << issue.file
                          << ":" << issue.line << " - " << issue.pattern << std::endl;
            }
        }
    }

    int getNumThreads() const { return omp_get_max_threads(); }
};

int main(int argc, char* argv[]) {
    std::string targetDir = ".";
    if (argc > 1) targetDir = argv[1];

    SASTAnalyzer analyzer;
    std::cout << "=== SAST (OpenMP, " << analyzer.getNumThreads() << " потоков) ===" << std::endl;

    auto start = omp_get_wtime();
    auto issues = analyzer.analyzeDirectory(targetDir);
    auto end = omp_get_wtime();

    analyzer.printResults(issues);
    std::cout << "\nВремя: " << (end - start) << " сек" << std::endl;

    return 0;
}

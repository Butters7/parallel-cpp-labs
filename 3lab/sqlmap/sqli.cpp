#include <mpi.h>      // MPI - библиотека для распределённых вычислений
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <sstream>
#include <regex>
#include <cstring>
#include <algorithm>
#include "xml_parser.h"

// Теги для MPI сообщений
const int TAG_TASK = 1;         // Отправка задания worker'у
const int TAG_RESULT = 2;       // Получение результата от worker'а
const int TAG_TERMINATE = 3;    // Сигнал завершения работы
const int TAG_STATS = 4;        // Реал-тайм статистика
const int TAG_RETRY = 5;        // Повторный запуск при сбое

// Максимальные размеры для передачи данных
const int MAX_URL_LEN = 512;
const int MAX_DATA_LEN = 1024;
const int MAX_RESULT_LEN = 8192;
const int MAX_RETRIES = 3;      // Максимальное количество повторных попыток

// Профили сканирования
enum ScanProfile {
    PROFILE_QUICK = 1,      // Быстрое сканирование (level 1-2, risk 1)
    PROFILE_DEEP = 2,       // Глубокий аудит (level 3-5, risk 2)
    PROFILE_AGGRESSIVE = 3  // Агрессивное тестирование (level 5, risk 3)
};

// Структура задания для MPI передачи
struct ScanTask {
    char url[MAX_URL_LEN];
    char data[MAX_DATA_LEN];
    int task_id;
    int profile;            // Профиль сканирования
    int retry_count;        // Счётчик повторных попыток
};

// Структура результата сканирования
struct ScanResult {
    int task_id;
    int worker_rank;
    int vulnerable;         // 1 = найдена уязвимость, 0 = нет
    int success;            // 1 = успешно выполнено, 0 = ошибка
    double scan_time;
    char url[MAX_URL_LEN];
    char details[MAX_RESULT_LEN];
};

// Структура реал-тайм статистики worker'а
struct WorkerStats {
    int worker_rank;
    int tasks_completed;
    int tasks_in_progress;
    int vulnerabilities_found;
    double total_time;
    double current_task_time;
};

// Получение параметров профиля сканирования
std::pair<int, int> get_profile_params(int profile) {
    switch (profile) {
        case PROFILE_QUICK:
            return {1, 1};  // level 1, risk 1 - быстрое сканирование
        case PROFILE_DEEP:
            return {3, 2};  // level 3, risk 2 - глубокий аудит
        case PROFILE_AGGRESSIVE:
            return {5, 3};  // level 5, risk 3 - агрессивное тестирование
        default:
            return {1, 1};
    }
}

std::string get_profile_name(int profile) {
    switch (profile) {
        case PROFILE_QUICK: return "Quick (level 1-2)";
        case PROFILE_DEEP: return "Deep Audit (level 3-5)";
        case PROFILE_AGGRESSIVE: return "Aggressive (high risk)";
        default: return "Unknown";
    }
}

// Класс для выполнения sqlmap сканирования
class SQLMapRunner {
private:
    std::string sqlmapPath;
    std::string cookies;
    int worker_rank;

public:
    SQLMapRunner(int rank, const std::string& cookies = "", const std::string& path = "sqlmap")
        : worker_rank(rank), cookies(cookies), sqlmapPath(path) {}

    // Выполнение команды и получение вывода
    std::string executeCommand(const std::string& command, int& returnCode) {
        std::string result;
        char buffer[256];

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            returnCode = -1;
            return "Error: failed to execute command";
        }

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
            if (result.length() > MAX_RESULT_LEN - 256) break;
        }

        returnCode = pclose(pipe);
        return result;
    }

    // XML unescaping для URL из XML
    std::string unescapeXml(const std::string& input) {
        std::string output;
        size_t pos = 0;
        size_t i = 0;
        while (i < input.size()) {
            if (input[i] == '&') {
                output += input.substr(pos, i - pos);
                if (input.substr(i, 4) == "&lt;") { output += "<"; i += 4; }
                else if (input.substr(i, 4) == "&gt;") { output += ">"; i += 4; }
                else if (input.substr(i, 5) == "&amp;") { output += "&"; i += 5; }
                else if (input.substr(i, 6) == "&quot;") { output += "\""; i += 6; }
                else if (input.substr(i, 6) == "&apos;") { output += "'"; i += 6; }
                else { ++i; }
                pos = i;
            } else {
                ++i;
            }
        }
        output += input.substr(pos);
        return output;
    }

    // Выполнение сканирования одной цели
    ScanResult scan(const ScanTask& task) {
        ScanResult result;
        result.task_id = task.task_id;
        result.worker_rank = worker_rank;
        result.vulnerable = 0;
        result.success = 0;
        std::strncpy(result.url, task.url, MAX_URL_LEN - 1);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Получаем параметры профиля
        auto [level, risk] = get_profile_params(task.profile);

        // Формируем команду sqlmap
        std::stringstream ss;
        ss << sqlmapPath << " -u \"" << unescapeXml(std::string(task.url)) << "\""
           << " --batch"                    // Автоматический режим
           << " --level=" << level
           << " --risk=" << risk
           << " --timeout=30"               // Таймаут для стабильности
           << " --retries=2";               // Внутренние повторы sqlmap

        if (!cookies.empty()) {
            ss << " --cookie='" << cookies << "'";
        }

        if (strlen(task.data) > 0) {
            ss << " --data=\"" << unescapeXml(std::string(task.data)) << "\"";
        }

        ss << " --dbs 2>&1";  // Перенаправляем stderr в stdout

        std::string command = ss.str();

        std::cout << "[Worker " << worker_rank << "] Scanning: " << task.url
                  << " (Profile: " << get_profile_name(task.profile) << ")" << std::endl;

        int returnCode;
        std::string output = executeCommand(command, returnCode);

        auto end_time = std::chrono::high_resolution_clock::now();
        result.scan_time = std::chrono::duration<double>(end_time - start_time).count();

        // Анализ результата на наличие уязвимостей
        if (output.find("is vulnerable") != std::string::npos ||
            output.find("injectable") != std::string::npos ||
            output.find("available databases") != std::string::npos) {
            result.vulnerable = 1;
        }

        // Проверка успешности выполнения
        if (returnCode == 0 || output.find("shutting down") != std::string::npos) {
            result.success = 1;
        }

        // Сохраняем детали (первые MAX_RESULT_LEN символов)
        std::strncpy(result.details, output.c_str(), MAX_RESULT_LEN - 1);
        result.details[MAX_RESULT_LEN - 1] = '\0';

        return result;
    }
};

// MASTER процесс - распределяет цели и собирает результаты
void master_process(int world_size, const std::vector<TargetInfo>& targets, int profile) {
    std::cout << "=== MASTER: Запуск распределённого SQL-инъекция сканирования ===" << std::endl;
    std::cout << "Количество worker'ов: " << (world_size - 1) << std::endl;
    std::cout << "Всего целей для сканирования: " << targets.size() << std::endl;
    std::cout << "Профиль: " << get_profile_name(profile) << std::endl;

    std::vector<ScanResult> all_results;
    std::vector<WorkerStats> worker_stats(world_size);
    std::vector<int> failed_tasks;  // Задания для повторного запуска

    int next_task = 0;
    int tasks_sent = 0;
    int tasks_completed = 0;
    int total_tasks = targets.size();
    int total_vulnerabilities = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Открываем файл отчёта
    std::ofstream report("sqlmap_mpi_report.txt");
    report << "=== Distributed SQLMap Scanner Report ===" << std::endl;
    report << "Workers: " << (world_size - 1) << std::endl;
    report << "Targets: " << total_tasks << std::endl;
    report << "Profile: " << get_profile_name(profile) << std::endl;
    report << "==========================================\n" << std::endl;

    // Автоматическое распределение: раздаём начальные задания каждому worker'у
    for (int worker = 1; worker < world_size && next_task < total_tasks; worker++) {
        ScanTask task;
        std::strncpy(task.url, targets[next_task].url.c_str(), MAX_URL_LEN - 1);
        std::strncpy(task.data, targets[next_task].data.c_str(), MAX_DATA_LEN - 1);
        task.task_id = next_task;
        task.profile = profile;
        task.retry_count = 0;

        // MPI_Send - отправляем задание worker'у
        MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);

        std::cout << "MASTER: Отправил задание " << next_task << " worker'у " << worker << std::endl;
        next_task++;
        tasks_sent++;
    }

    // Динамическое распределение: получаем результаты и раздаём новые задания
    while (tasks_completed < tasks_sent) {
        ScanResult result;
        MPI_Status status;

        // MPI_Recv с MPI_ANY_SOURCE - получаем результат от любого worker'а
        MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        int worker = status.MPI_SOURCE;
        tasks_completed++;

        // Реал-тайм статистика
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();

        std::cout << "\n--- Реал-тайм статистика ---" << std::endl;
        std::cout << "Прогресс: " << tasks_completed << "/" << total_tasks
                  << " (" << (100 * tasks_completed / total_tasks) << "%)" << std::endl;
        std::cout << "Время: " << elapsed << " сек" << std::endl;
        std::cout << "Worker " << worker << ": завершил задание " << result.task_id
                  << " за " << result.scan_time << " сек" << std::endl;

        // Проверка на сбой и повторный запуск
        if (!result.success) {
            // Находим исходное задание для повторной попытки
            int task_id = result.task_id;
            if (task_id < total_tasks) {
                // Проверяем количество попыток
                static std::vector<int> retry_counts(total_tasks, 0);
                retry_counts[task_id]++;

                if (retry_counts[task_id] < MAX_RETRIES) {
                    std::cout << "MASTER: Повторный запуск задания " << task_id
                              << " (попытка " << retry_counts[task_id] << ")" << std::endl;

                    ScanTask retry_task;
                    std::strncpy(retry_task.url, targets[task_id].url.c_str(), MAX_URL_LEN - 1);
                    std::strncpy(retry_task.data, targets[task_id].data.c_str(), MAX_DATA_LEN - 1);
                    retry_task.task_id = task_id;
                    retry_task.profile = profile;
                    retry_task.retry_count = retry_counts[task_id];

                    // Отправляем повторное задание свободному worker'у
                    MPI_Send(&retry_task, sizeof(ScanTask), MPI_BYTE, worker, TAG_RETRY, MPI_COMM_WORLD);
                    tasks_sent++;
                    continue;  // Не добавляем неудачный результат
                } else {
                    std::cout << "MASTER: Задание " << task_id << " не удалось после "
                              << MAX_RETRIES << " попыток" << std::endl;
                }
            }
        }

        all_results.push_back(result);

        if (result.vulnerable) {
            total_vulnerabilities++;
            std::cout << "!!! УЯЗВИМОСТЬ НАЙДЕНА: " << result.url << " !!!" << std::endl;
        }

        // Записываем результат в отчёт
        report << "[Task " << result.task_id << "] " << result.url << std::endl;
        report << "Worker: " << result.worker_rank << std::endl;
        report << "Time: " << result.scan_time << " sec" << std::endl;
        report << "Vulnerable: " << (result.vulnerable ? "YES" : "NO") << std::endl;
        report << "Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
        report << "-------------------------------------------" << std::endl;

        // Автоматическое перераспределение: если есть ещё задания, отправляем worker'у
        if (next_task < total_tasks) {
            ScanTask task;
            std::strncpy(task.url, targets[next_task].url.c_str(), MAX_URL_LEN - 1);
            std::strncpy(task.data, targets[next_task].data.c_str(), MAX_DATA_LEN - 1);
            task.task_id = next_task;
            task.profile = profile;
            task.retry_count = 0;

            MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);

            std::cout << "MASTER: Отправил новое задание " << next_task << " worker'у " << worker << std::endl;
            next_task++;
            tasks_sent++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Отправляем сигнал завершения всем worker'ам
    for (int worker = 1; worker < world_size; worker++) {
        ScanTask terminate_task;
        terminate_task.task_id = -1;  // Сигнал завершения
        MPI_Send(&terminate_task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TERMINATE, MPI_COMM_WORLD);
    }

    // Вывод итоговой статистики
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== ИТОГОВАЯ СТАТИСТИКА ===" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Общее время: " << total_time << " сек" << std::endl;
    std::cout << "Просканировано целей: " << all_results.size() << std::endl;
    std::cout << "Найдено уязвимостей: " << total_vulnerabilities << std::endl;
    std::cout << "Средняя скорость: " << (all_results.size() / total_time) << " целей/сек" << std::endl;

    // Статистика по worker'ам
    std::cout << "\n--- Загрузка процессов ---" << std::endl;
    std::vector<int> tasks_per_worker(world_size, 0);
    std::vector<double> time_per_worker(world_size, 0.0);

    for (const auto& res : all_results) {
        tasks_per_worker[res.worker_rank]++;
        time_per_worker[res.worker_rank] += res.scan_time;
    }

    for (int w = 1; w < world_size; w++) {
        std::cout << "Worker " << w << ": " << tasks_per_worker[w] << " задач, "
                  << time_per_worker[w] << " сек" << std::endl;
    }

    // Список найденных уязвимостей
    if (total_vulnerabilities > 0) {
        std::cout << "\n--- НАЙДЕННЫЕ УЯЗВИМОСТИ ---" << std::endl;
        for (const auto& res : all_results) {
            if (res.vulnerable) {
                std::cout << "  * " << res.url << std::endl;
            }
        }
    }

    // Записываем итоги в отчёт
    report << "\n=== SUMMARY ===" << std::endl;
    report << "Total time: " << total_time << " sec" << std::endl;
    report << "Targets scanned: " << all_results.size() << std::endl;
    report << "Vulnerabilities found: " << total_vulnerabilities << std::endl;
    report.close();

    std::cout << "\nОтчёт сохранён в: sqlmap_mpi_report.txt" << std::endl;
}

// WORKER процесс - выполняет сканирование по заданиям от master'а
void worker_process(int rank, const std::string& cookies) {
    SQLMapRunner runner(rank, cookies);

    while (true) {
        ScanTask task;
        MPI_Status status;

        // MPI_Probe - проверяем тег сообщения без его получения
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Если получен сигнал завершения - выходим
        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            std::cout << "Worker " << rank << ": Получен сигнал завершения" << std::endl;
            break;
        }

        // Получаем задание (обычное или повторное)
        int tag = (status.MPI_TAG == TAG_RETRY) ? TAG_RETRY : TAG_TASK;
        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);

        if (tag == TAG_RETRY) {
            std::cout << "Worker " << rank << ": Повторное сканирование задания " << task.task_id
                      << " (попытка " << task.retry_count << ")" << std::endl;
        }

        // Выполняем сканирование
        ScanResult result = runner.scan(task);

        // Отправляем результат master'у
        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    // MPI_Init - инициализация MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;

    // MPI_Comm_size - получаем общее количество процессов
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // MPI_Comm_rank - получаем номер текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        std::cerr << "Требуется минимум 2 процесса (1 master + 1 worker)" << std::endl;
        std::cerr << "Запуск: mpirun -n 4 ./sqli [profile] [xml_file]" << std::endl;
        std::cerr << "Профили: 1=quick, 2=deep, 3=aggressive" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Параметры по умолчанию
    int profile = PROFILE_QUICK;
    std::string xml_file = "scan_results.xml";
    std::string cookies = "security=low; PHPSESSID=d5dc67afd50ed8143c1b4810684e282c";

    // Парсинг аргументов командной строки
    if (argc > 1) {
        profile = std::atoi(argv[1]);
        if (profile < 1 || profile > 3) profile = PROFILE_QUICK;
    }
    if (argc > 2) {
        xml_file = argv[2];
    }

    if (world_rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "  Distributed SQLMap Scanner (MPI)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Проверяем доступность sqlmap
        std::cout << "Проверка доступности sqlmap..." << std::endl;
        FILE* pipe = popen("sqlmap --version 2>&1", "r");
        if (pipe) {
            char buffer[128];
            std::string version;
            while (fgets(buffer, sizeof(buffer), pipe)) {
                version += buffer;
            }
            pclose(pipe);

            std::regex version_regex(R"(\d+\.\d+)");
            if (!std::regex_search(version, version_regex)) {
                std::cerr << "Ошибка: sqlmap не найден!" << std::endl;
                std::cerr << "Установка: pip install sqlmap" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::cout << "sqlmap доступен" << std::endl;
        }

        // Парсим XML с целями
        XMLParser parser;
        std::vector<TargetInfo> targets = parser.parseFile(xml_file);

        if (targets.empty()) {
            std::cerr << "Ошибка: нет целей для сканирования в " << xml_file << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Запускаем master процесс
        master_process(world_size, targets, profile);
    } else {
        // Запускаем worker процесс
        worker_process(world_rank, cookies);
    }

    // MPI_Finalize - завершение работы MPI
    MPI_Finalize();

    return 0;
}

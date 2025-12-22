#include <mpi.h>      // MPI - библиотека для распределённых вычислений
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <climits>
#include <filesystem>
#include <sys/stat.h>

namespace fs = std::filesystem;

// Теги для MPI сообщений
const int TAG_TASK = 1;         // Отправка задания worker'у
const int TAG_RESULT = 2;       // Получение результата от worker'а
const int TAG_TERMINATE = 3;    // Сигнал завершения работы
const int TAG_MORE_WORK = 4;    // Запрос дополнительной работы

// Максимальные размеры для передачи данных
const int MAX_PATH_LEN = 512;
const int MAX_IOC_LEN = 256;
const int MAX_DESC_LEN = 256;
const int MAX_CONTEXT_LEN = 512;

// Уровни логирования
enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
};

class Logger {
private:
    LogLevel current_level;
    int rank;
    std::ofstream log_file;

    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LOG_DEBUG: return "DEBUG";
            case LOG_INFO:  return "INFO ";
            case LOG_WARN:  return "WARN ";
            case LOG_ERROR: return "ERROR";
            default: return "?????";
        }
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

public:
    Logger(int mpi_rank, LogLevel level = LOG_INFO) : rank(mpi_rank), current_level(level) {
        std::string filename = "ioc_" + std::to_string(rank) + ".log";
        log_file.open(filename);
    }

    ~Logger() {
        if (log_file.is_open()) log_file.close();
    }

    void set_level(LogLevel level) { current_level = level; }

    void log(LogLevel level, const std::string& message) {
        if (level < current_level) return;

        std::stringstream ss;
        ss << "[" << get_timestamp() << "] [" << level_to_string(level) << "] "
           << "[Rank " << rank << "] " << message;

        std::string log_msg = ss.str();

        // Вывод в консоль (INFO и выше)
        if (level >= LOG_INFO) {
            std::cout << log_msg << std::endl;
        }

        // Запись в файл (все уровни)
        if (log_file.is_open()) {
            log_file << log_msg << std::endl;
            log_file.flush();
        }
    }

    void debug(const std::string& msg) { log(LOG_DEBUG, msg); }
    void info(const std::string& msg)  { log(LOG_INFO, msg); }
    void warn(const std::string& msg)  { log(LOG_WARN, msg); }
    void error(const std::string& msg) { log(LOG_ERROR, msg); }
};

// Структуры данных для MPI передачи
struct IOCPattern {
    char type[32];          // hash, domain, ip, string, filename, registry
    char value[MAX_IOC_LEN];
    char description[MAX_DESC_LEN];
    char severity[16];      // CRITICAL, HIGH, MEDIUM, LOW
};

struct ScanTask {
    char file_path[MAX_PATH_LEN];
    int task_id;
};

struct ScanResult {
    int task_id;
    int worker_rank;
    int matches_count;
    int success;            // 1 = успех, 0 = ошибка
    double scan_time;
    char file_path[MAX_PATH_LEN];
    char matched_ioc[MAX_IOC_LEN];
    char severity[16];
    char context[MAX_CONTEXT_LEN];
};

// Класс для загрузки IOC из различных форматов
class IOCLoader {
private:
    Logger* logger;

    // Проверка валидности IOC (устойчивость к некорректным данным)
    bool validate_ioc(const IOCPattern& ioc) {
        if (strlen(ioc.value) == 0) return false;
        if (strlen(ioc.type) == 0) return false;

        // Проверка типа
        std::vector<std::string> valid_types = {"hash", "domain", "ip", "string", "filename", "registry"};
        bool valid_type = false;
        for (const auto& t : valid_types) {
            if (t == ioc.type) { valid_type = true; break; }
        }
        if (!valid_type) {
            logger->warn("Invalid IOC type: " + std::string(ioc.type));
            return false;
        }

        return true;
    }

public:
    IOCLoader(Logger* log) : logger(log) {}

    // Загрузка из текстового файла (один IOC на строку: type:value:description:severity)
    std::vector<IOCPattern> load_from_txt(const std::string& filename) {
        std::vector<IOCPattern> iocs;
        std::ifstream file(filename);

        if (!file.is_open()) {
            logger->error("Cannot open IOC file: " + filename);
            return iocs;
        }

        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;
            if (line.empty() || line[0] == '#') continue;  // Пропуск пустых строк и комментариев

            IOCPattern ioc;
            std::memset(&ioc, 0, sizeof(IOCPattern));

            std::stringstream ss(line);
            std::string type, value, desc, severity;

            if (std::getline(ss, type, ':') &&
                std::getline(ss, value, ':') &&
                std::getline(ss, desc, ':') &&
                std::getline(ss, severity)) {

                std::strncpy(ioc.type, type.c_str(), 31);
                std::strncpy(ioc.value, value.c_str(), MAX_IOC_LEN - 1);
                std::strncpy(ioc.description, desc.c_str(), MAX_DESC_LEN - 1);
                std::strncpy(ioc.severity, severity.c_str(), 15);

                if (validate_ioc(ioc)) {
                    iocs.push_back(ioc);
                } else {
                    logger->warn("Skipping invalid IOC at line " + std::to_string(line_num));
                }
            } else {
                logger->warn("Malformed IOC at line " + std::to_string(line_num) + ": " + line);
            }
        }

        logger->info("Loaded " + std::to_string(iocs.size()) + " IOCs from TXT: " + filename);
        return iocs;
    }

    // Загрузка из CSV файла (type,value,description,severity)
    std::vector<IOCPattern> load_from_csv(const std::string& filename) {
        std::vector<IOCPattern> iocs;
        std::ifstream file(filename);

        if (!file.is_open()) {
            logger->error("Cannot open CSV file: " + filename);
            return iocs;
        }

        std::string line;
        bool first_line = true;
        int line_num = 0;

        while (std::getline(file, line)) {
            line_num++;
            if (first_line) { first_line = false; continue; }  // Пропуск заголовка
            if (line.empty()) continue;

            IOCPattern ioc;
            std::memset(&ioc, 0, sizeof(IOCPattern));

            std::stringstream ss(line);
            std::string type, value, desc, severity;

            if (std::getline(ss, type, ',') &&
                std::getline(ss, value, ',') &&
                std::getline(ss, desc, ',') &&
                std::getline(ss, severity)) {

                std::strncpy(ioc.type, type.c_str(), 31);
                std::strncpy(ioc.value, value.c_str(), MAX_IOC_LEN - 1);
                std::strncpy(ioc.description, desc.c_str(), MAX_DESC_LEN - 1);
                std::strncpy(ioc.severity, severity.c_str(), 15);

                if (validate_ioc(ioc)) {
                    iocs.push_back(ioc);
                }
            } else {
                logger->warn("Malformed CSV at line " + std::to_string(line_num));
            }
        }

        logger->info("Loaded " + std::to_string(iocs.size()) + " IOCs from CSV: " + filename);
        return iocs;
    }

    // Загрузка из JSON файла (упрощённый парсер)
    std::vector<IOCPattern> load_from_json(const std::string& filename) {
        std::vector<IOCPattern> iocs;
        std::ifstream file(filename);

        if (!file.is_open()) {
            logger->error("Cannot open JSON file: " + filename);
            return iocs;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Простой парсинг JSON (ищем объекты с полями type, value, description, severity)
        size_t pos = 0;
        while ((pos = content.find("\"type\"", pos)) != std::string::npos) {
            IOCPattern ioc;
            std::memset(&ioc, 0, sizeof(IOCPattern));

            auto extract_value = [&](const std::string& key, size_t start) -> std::string {
                size_t key_pos = content.find("\"" + key + "\"", start);
                if (key_pos == std::string::npos || key_pos > start + 500) return "";
                size_t colon = content.find(":", key_pos);
                size_t quote1 = content.find("\"", colon);
                size_t quote2 = content.find("\"", quote1 + 1);
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    return content.substr(quote1 + 1, quote2 - quote1 - 1);
                }
                return "";
            };

            std::string type = extract_value("type", pos);
            std::string value = extract_value("value", pos);
            std::string desc = extract_value("description", pos);
            std::string severity = extract_value("severity", pos);

            if (!type.empty() && !value.empty()) {
                std::strncpy(ioc.type, type.c_str(), 31);
                std::strncpy(ioc.value, value.c_str(), MAX_IOC_LEN - 1);
                std::strncpy(ioc.description, desc.c_str(), MAX_DESC_LEN - 1);
                std::strncpy(ioc.severity, severity.empty() ? "MEDIUM" : severity.c_str(), 15);

                if (validate_ioc(ioc)) {
                    iocs.push_back(ioc);
                }
            }
            pos++;
        }

        logger->info("Loaded " + std::to_string(iocs.size()) + " IOCs from JSON: " + filename);
        return iocs;
    }

    // Автоопределение формата по расширению
    std::vector<IOCPattern> load_auto(const std::string& filename) {
        if (filename.find(".json") != std::string::npos) {
            return load_from_json(filename);
        } else if (filename.find(".csv") != std::string::npos) {
            return load_from_csv(filename);
        } else {
            return load_from_txt(filename);
        }
    }

    // Загрузка IOC по умолчанию (для тестирования)
    std::vector<IOCPattern> load_default() {
        std::vector<IOCPattern> iocs;

        auto add_ioc = [&](const char* type, const char* value, const char* desc, const char* sev) {
            IOCPattern ioc;
            std::memset(&ioc, 0, sizeof(IOCPattern));
            std::strncpy(ioc.type, type, 31);
            std::strncpy(ioc.value, value, MAX_IOC_LEN - 1);
            std::strncpy(ioc.description, desc, MAX_DESC_LEN - 1);
            std::strncpy(ioc.severity, sev, 15);
            iocs.push_back(ioc);
        };

        add_ioc("hash", "malware_hash_12345", "Known ransomware signature", "CRITICAL");
        add_ioc("domain", "evil-domain.com", "C&C server domain", "HIGH");
        add_ioc("ip", "192.168.77.100", "Malicious IP address", "HIGH");
        add_ioc("string", "ransomware", "Ransomware signature string", "MEDIUM");
        add_ioc("hash", "trojan_hash_67890", "Trojan horse signature", "HIGH");
        add_ioc("domain", "suspicious-site.org", "Phishing domain", "MEDIUM");
        add_ioc("ip", "10.0.99.200", "Botnet C&C server", "CRITICAL");
        add_ioc("string", "backdoor", "Backdoor signature", "HIGH");
        add_ioc("filename", "mimikatz.exe", "Credential dumping tool", "CRITICAL");
        add_ioc("registry", "HKLM\\SOFTWARE\\malware", "Malware registry key", "HIGH");

        logger->info("Loaded " + std::to_string(iocs.size()) + " default IOCs");
        return iocs;
    }
};

// Класс для сканирования файлов
class IOCScanner {
private:
    std::vector<IOCPattern> known_iocs;
    Logger* logger;

public:
    IOCScanner(Logger* log) : logger(log) {}

    void set_iocs(const std::vector<IOCPattern>& iocs) {
        known_iocs = iocs;
    }

    // Проверка типа файла (симлинки и специальные файлы)
    bool should_scan_file(const std::string& path) {
        struct stat file_stat;

        if (lstat(path.c_str(), &file_stat) != 0) {
            logger->warn("Cannot stat file: " + path);
            return false;
        }

        // Пропуск симлинков
        if (S_ISLNK(file_stat.st_mode)) {
            logger->warn("Skipping symlink: " + path);
            return false;
        }

        // Пропуск специальных файлов (устройства, сокеты, FIFO)
        if (S_ISCHR(file_stat.st_mode) || S_ISBLK(file_stat.st_mode) ||
            S_ISSOCK(file_stat.st_mode) || S_ISFIFO(file_stat.st_mode)) {
            logger->warn("Skipping special file: " + path);
            return false;
        }

        // Пропуск директорий
        if (S_ISDIR(file_stat.st_mode)) {
            logger->debug("Skipping directory: " + path);
            return false;
        }

        return true;
    }

    // Сканирование файла на наличие IOC
    ScanResult scan_file(const ScanTask& task, int worker_rank) {
        ScanResult result;
        std::memset(&result, 0, sizeof(ScanResult));
        result.task_id = task.task_id;
        result.worker_rank = worker_rank;
        result.matches_count = 0;
        result.success = 0;
        std::strncpy(result.file_path, task.file_path, MAX_PATH_LEN - 1);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::string path(task.file_path);

        // Проверка типа файла
        if (!should_scan_file(path)) {
            result.success = 1;  // Не ошибка, просто пропуск
            std::strncpy(result.context, "Skipped (symlink/special)", MAX_CONTEXT_LEN - 1);
            return result;
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            logger->error("Cannot open file: " + path);
            std::strncpy(result.context, "File access error", MAX_CONTEXT_LEN - 1);
            return result;
        }

        std::string line;
        int line_number = 0;

        while (std::getline(file, line)) {
            line_number++;

            for (const auto& ioc : known_iocs) {
                // Поиск IOC в строке
                if (line.find(ioc.value) != std::string::npos) {
                    result.matches_count++;
                    std::strncpy(result.matched_ioc, ioc.value, MAX_IOC_LEN - 1);
                    std::strncpy(result.severity, ioc.severity, 15);

                    std::string ctx = "Line " + std::to_string(line_number) + ": " + ioc.description;
                    std::strncpy(result.context, ctx.c_str(), MAX_CONTEXT_LEN - 1);

                    logger->debug("Found IOC '" + std::string(ioc.value) + "' in " + path +
                                 " at line " + std::to_string(line_number));
                }
            }
        }

        file.close();
        result.success = 1;

        auto end_time = std::chrono::high_resolution_clock::now();
        result.scan_time = std::chrono::duration<double>(end_time - start_time).count();

        return result;
    }
};

// ============================================================================
// Сбор файлов для сканирования
// ============================================================================
std::vector<std::string> collect_files(const std::string& directory, Logger* logger) {
    std::vector<std::string> files;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory,
                fs::directory_options::skip_permission_denied)) {
            if (entry.is_regular_file()) {
                files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        logger->error("Filesystem error: " + std::string(e.what()));
    }

    return files;
}

// Создание тестовых файлов
void create_test_files(Logger* logger) {
    logger->info("Creating test files...");

    fs::create_directories("test_data/logs");
    fs::create_directories("test_data/configs");

    std::ofstream auth_log("test_data/logs/auth.log");
    auth_log << "User login: admin\n";
    auth_log << "Failed login attempt from 192.168.77.100\n";
    auth_log << "User logout: admin\n";
    auth_log.close();

    std::ofstream system_log("test_data/logs/system.log");
    system_log << "System started\n";
    system_log << "Process malware_hash_12345 detected\n";
    system_log << "Connection to evil-domain.com blocked\n";
    system_log.close();

    std::ofstream config("test_data/configs/system.conf");
    config << "hostname=server1\n";
    config << "backdoor_port=4444\n";
    config << "admin_ip=10.0.99.200\n";
    config.close();

    // Создание тестового файла IOC
    std::ofstream ioc_file("test_iocs.txt");
    ioc_file << "# IOC Database\n";
    ioc_file << "hash:malware_hash_12345:Known ransomware:CRITICAL\n";
    ioc_file << "domain:evil-domain.com:C&C server:HIGH\n";
    ioc_file << "ip:192.168.77.100:Malicious IP:HIGH\n";
    ioc_file << "string:backdoor:Backdoor signature:MEDIUM\n";
    ioc_file << "ip:10.0.99.200:Botnet C&C:CRITICAL\n";
    ioc_file.close();

    logger->info("Test files created in test_data/");
}

// ============================================================================
// MASTER процесс - распределяет файлы и собирает результаты
// ============================================================================
void master_process(int world_size, const std::string& scan_dir, const std::string& ioc_file) {
    Logger logger(0, LOG_DEBUG);
    logger.info("=== MASTER: Starting distributed IOC scan ===");
    logger.info("Workers: " + std::to_string(world_size - 1));

    // Создаём тестовые файлы если директория не существует
    if (!fs::exists(scan_dir)) {
        create_test_files(&logger);
    }

    // Загрузка IOC
    IOCLoader loader(&logger);
    std::vector<IOCPattern> iocs;

    if (!ioc_file.empty() && fs::exists(ioc_file)) {
        iocs = loader.load_auto(ioc_file);
    } else {
        iocs = loader.load_default();
    }

    // Рассылка IOC всем worker'ам через MPI_Bcast
    // Проверка размера данных перед передачей
    int ioc_count = iocs.size();
    MPI_Bcast(&ioc_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ioc_count > 0) {
        // Проверяем что структура POD и безопасна для MPI_BYTE передачи
        // IOCPattern содержит только char массивы фиксированного размера
        int message_size = ioc_count * sizeof(IOCPattern);
        if (message_size > INT_MAX / 2) {
            logger.error("IOC data too large for single broadcast");
            return;
        }
        MPI_Bcast(iocs.data(), message_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Сбор файлов для сканирования
    std::vector<std::string> files = collect_files(scan_dir, &logger);
    logger.info("Files to scan: " + std::to_string(files.size()));

    if (files.empty()) {
        logger.error("No files found in " + scan_dir);
        // Отправляем сигнал завершения
        for (int w = 1; w < world_size; w++) {
            ScanTask term;
            term.task_id = -1;
            MPI_Send(&term, sizeof(ScanTask), MPI_BYTE, w, TAG_TERMINATE, MPI_COMM_WORLD);
        }
        return;
    }

    std::vector<ScanResult> all_results;
    std::vector<double> worker_times(world_size, 0.0);
    std::vector<int> worker_tasks(world_size, 0);

    int next_task = 0;
    int tasks_sent = 0;
    int tasks_completed = 0;
    int total_tasks = files.size();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Динамическая балансировка: раздаём по одному заданию каждому worker'у
    for (int worker = 1; worker < world_size && next_task < total_tasks; worker++) {
        ScanTask task;
        std::strncpy(task.file_path, files[next_task].c_str(), MAX_PATH_LEN - 1);
        task.task_id = next_task;

        MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);

        logger.debug("Sent task " + std::to_string(next_task) + " to worker " + std::to_string(worker));
        next_task++;
        tasks_sent++;
    }

    // Получаем результаты и раздаём новые задания (минимизация простоя)
    while (tasks_completed < tasks_sent) {
        ScanResult result;
        MPI_Status status;

        // MPI_Recv с MPI_ANY_SOURCE - получаем от любого worker'а
        MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        int worker = status.MPI_SOURCE;
        tasks_completed++;
        worker_tasks[worker]++;
        worker_times[worker] += result.scan_time;

        all_results.push_back(result);

        // Реал-тайм прогресс
        if (tasks_completed % 10 == 0 || tasks_completed == total_tasks) {
            logger.info("Progress: " + std::to_string(tasks_completed) + "/" +
                       std::to_string(total_tasks) + " (" +
                       std::to_string(100 * tasks_completed / total_tasks) + "%)");
        }

        // Балансировка: отправляем новое задание освободившемуся worker'у
        if (next_task < total_tasks) {
            ScanTask task;
            std::strncpy(task.file_path, files[next_task].c_str(), MAX_PATH_LEN - 1);
            task.task_id = next_task;

            MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);

            logger.debug("Sent task " + std::to_string(next_task) + " to worker " + std::to_string(worker));
            next_task++;
            tasks_sent++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // Отправляем сигнал завершения
    for (int worker = 1; worker < world_size; worker++) {
        ScanTask term;
        term.task_id = -1;
        MPI_Send(&term, sizeof(ScanTask), MPI_BYTE, worker, TAG_TERMINATE, MPI_COMM_WORLD);
    }

    // ============================================================================
    // Генерация отчёта
    // ============================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== IOC SCAN REPORT ===" << std::endl;
    std::cout << "========================================" << std::endl;

    int total_matches = 0;
    std::map<std::string, int> severity_stats;
    std::vector<ScanResult> findings;

    for (const auto& res : all_results) {
        if (res.matches_count > 0) {
            total_matches += res.matches_count;
            severity_stats[res.severity]++;
            findings.push_back(res);
        }
    }

    std::cout << "\nTotal time: " << total_time << " sec" << std::endl;
    std::cout << "Files scanned: " << all_results.size() << std::endl;
    std::cout << "IOC matches found: " << total_matches << std::endl;

    // Статистика по уровням
    std::cout << "\n--- Severity Statistics ---" << std::endl;
    for (const auto& [sev, count] : severity_stats) {
        std::cout << "  " << sev << ": " << count << std::endl;
    }

    // Загрузка worker'ов
    std::cout << "\n--- Worker Load (balancing) ---" << std::endl;
    for (int w = 1; w < world_size; w++) {
        std::cout << "  Worker " << w << ": " << worker_tasks[w] << " tasks, "
                  << worker_times[w] << " sec" << std::endl;
    }

    // Найденные угрозы
    if (!findings.empty()) {
        std::cout << "\n--- DETECTED THREATS ---" << std::endl;
        for (const auto& f : findings) {
            std::cout << "[" << f.severity << "] " << f.file_path << std::endl;
            std::cout << "    IOC: " << f.matched_ioc << std::endl;
            std::cout << "    Context: " << f.context << std::endl;
        }
    }

    // Линейное ускорение
    double speedup = (world_size - 1) > 0 ? total_time / (world_size - 1) : total_time;
    std::cout << "\n--- Performance ---" << std::endl;
    std::cout << "Workers: " << (world_size - 1) << std::endl;
    std::cout << "Avg time per worker: " << speedup << " sec" << std::endl;

    // Сохранение отчёта в файл
    std::ofstream report("ioc_report.txt");
    report << "=== IOC Scan Report ===" << std::endl;
    report << "Files scanned: " << all_results.size() << std::endl;
    report << "Matches found: " << total_matches << std::endl;
    report << "Total time: " << total_time << " sec" << std::endl;
    for (const auto& f : findings) {
        report << f.severity << ": " << f.file_path << " - " << f.matched_ioc << std::endl;
    }
    report.close();

    logger.info("Report saved to ioc_report.txt");
}

// ============================================================================
// WORKER процесс - сканирует файлы
// ============================================================================
void worker_process(int rank) {
    Logger logger(rank, LOG_DEBUG);
    logger.info("Worker started");

    // Получаем IOC от master'а через MPI_Bcast
    int ioc_count;
    MPI_Bcast(&ioc_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Проверка размера перед выделением памяти
    if (ioc_count < 0 || ioc_count > 1000000) {
        logger.error("Invalid IOC count received: " + std::to_string(ioc_count));
        return;
    }

    std::vector<IOCPattern> iocs(ioc_count);
    if (ioc_count > 0) {
        int message_size = ioc_count * sizeof(IOCPattern);
        MPI_Bcast(iocs.data(), message_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    logger.debug("Received " + std::to_string(ioc_count) + " IOCs");

    IOCScanner scanner(&logger);
    scanner.set_iocs(iocs);

    while (true) {
        ScanTask task;
        MPI_Status status;

        // MPI_Probe - проверяем наличие сообщения и его тег
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Получаем размер сообщения для проверки
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);

        // Проверка размера сообщения
        if (count != sizeof(ScanTask)) {
            logger.error("Received message with wrong size: " + std::to_string(count));
            continue;
        }

        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            logger.info("Received termination signal");
            break;
        }

        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);

        logger.debug("Scanning: " + std::string(task.file_path));

        ScanResult result = scanner.scan_file(task, rank);

        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }

    logger.info("Worker finished");
}

int main(int argc, char** argv) {
    // MPI_Init - инициализация MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        std::cerr << "Requires at least 2 processes (1 master + 1 worker)" << std::endl;
        std::cerr << "Usage: mpirun -n 4 ./ioc [scan_dir] [ioc_file]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Параметры
    std::string scan_dir = "test_data";
    std::string ioc_file = "";

    if (argc > 1) scan_dir = argv[1];
    if (argc > 2) ioc_file = argv[2];

    if (world_rank == 0) {
        master_process(world_size, scan_dir, ioc_file);
    } else {
        worker_process(world_rank);
    }

    MPI_Finalize();
    return 0;
}

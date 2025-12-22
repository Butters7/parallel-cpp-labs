#include <mpi.h>      // MPI - библиотека для распределённых вычислений
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>

// Теги для MPI сообщений
const int TAG_TASK = 1;        // Отправка задания worker'у
const int TAG_RESULT = 2;      // Получение результата от worker'а
const int TAG_TERMINATE = 3;   // Сигнал завершения работы
const int TAG_HEARTBEAT = 4;   // Проверка живости worker'а

// Максимальные размеры для передачи данных
const int MAX_IP_LEN = 64;
const int MAX_RESULT_LEN = 4096;

struct ScanResult {
    char ip[MAX_IP_LEN];
    char result[MAX_RESULT_LEN];
    int success;
    int worker_rank;  // Какой worker выполнил задание
    double scan_time; // Время сканирования
};

// Функция для выполнения сканирования одного IP
// scan_rate - задержка между запросами (мс) для регулирования скорости
ScanResult scan_ip(const std::string& ip, int worker_rank, int scan_rate = 100) {
    ScanResult result;
    // Безопасное копирование с гарантированной нуль-терминацией
    std::memset(&result, 0, sizeof(ScanResult));
    std::strncpy(result.ip, ip.c_str(), MAX_IP_LEN - 1);
    result.ip[MAX_IP_LEN - 1] = '\0';
    result.success = 0;
    result.worker_rank = worker_rank;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Rate limiting для предотвращения перегрузки сети
    // sleep_for безопасен в MPI worker'ах - не блокирует другие процессы
    std::this_thread::sleep_for(std::chrono::milliseconds(scan_rate));

    std::stringstream command;
    // Добавляем таймаут для nmap чтобы избежать зависания
    command << "timeout 30 nmap -sV -sC -T4 " << ip << " 2>/dev/null";

    // popen в MPI worker: каждый worker выполняет в своем процессе
    // не создает конкуренции между процессами
    FILE* pipe = popen(command.str().c_str(), "r");
    if (pipe) {
        char buffer[128];
        std::string output;
        // Ограничиваем время чтения вывода
        auto read_start = std::chrono::steady_clock::now();
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            output += buffer;
            if (output.length() > MAX_RESULT_LEN - 128) break;

            // Защита от зависания на чтении
            auto elapsed = std::chrono::steady_clock::now() - read_start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 35) {
                break;
            }
        }
        int status = pclose(pipe);

        std::strncpy(result.result, output.c_str(), MAX_RESULT_LEN - 1);
        result.result[MAX_RESULT_LEN - 1] = '\0';
        result.success = (status == 0) ? 1 : 0;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.scan_time = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

// Функция для генерации диапазона IP-адресов
std::vector<std::string> generate_ip_range(const std::string& base_ip, int start, int end) {
    std::vector<std::string> ips;
    for (int i = start; i <= end; i++) {
        std::string ip = base_ip + "." + std::to_string(i);
        ips.push_back(ip);
    }
    return ips;
}

// Генерация нескольких подсетей для сканирования
std::vector<std::string> generate_multiple_subnets() {
    std::vector<std::string> all_ips;

    // Оптимизация маршрутов: группируем IP по подсетям для эффективности
    std::vector<std::string> subnets = {"192.168.1", "192.168.2", "10.0.0"};

    for (const auto& subnet : subnets) {
        auto ips = generate_ip_range(subnet, 1, 5);
        all_ips.insert(all_ips.end(), ips.begin(), ips.end());
    }

    return all_ips;
}

// MASTER процесс (rank 0) - распределяет задания между workers
void master_process(int world_size, const std::vector<std::string>& ip_range) {
    std::cout << "=== MASTER: Запуск распределённого сканирования ===" << std::endl;
    std::cout << "Количество worker'ов: " << (world_size - 1) << std::endl;
    std::cout << "Всего IP для сканирования: " << ip_range.size() << std::endl;

    std::vector<ScanResult> all_results;
    std::vector<bool> worker_alive(world_size, true);  // Отслеживание живости worker'ов
    std::vector<int> worker_tasks(world_size, 0);      // Счётчик задач на worker

    int next_task = 0;           // Индекс следующего задания
    int tasks_sent = 0;          // Отправлено заданий
    int tasks_completed = 0;     // Завершено заданий
    int total_tasks = ip_range.size();

    // Динамическое планирование: сначала раздаём по одному заданию каждому worker'у
    // MPI_Send отправляет данные конкретному процессу
    for (int worker = 1; worker < world_size && next_task < total_tasks; worker++) {
        char ip_buffer[MAX_IP_LEN];
        std::memset(ip_buffer, 0, MAX_IP_LEN);
        std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);
        ip_buffer[MAX_IP_LEN - 1] = '\0';

        // MPI_Send(data, count, datatype, dest, tag, comm)
        // Отправляем IP-адрес worker'у для сканирования
        MPI_Send(ip_buffer, MAX_IP_LEN, MPI_CHAR, worker, TAG_TASK, MPI_COMM_WORLD);

        std::cout << "MASTER: Отправил задание " << ip_buffer << " worker'у " << worker << std::endl;
        worker_tasks[worker]++;
        next_task++;
        tasks_sent++;
    }

    // Динамическая балансировка: получаем результаты и раздаём новые задания
    while (tasks_completed < tasks_sent) {
        ScanResult result;
        MPI_Status status;

        // Используем MPI_Probe для проверки наличия сообщения
        // Это позволяет добавить логику таймаута через неблокирующий опрос
        int flag = 0;
        auto timeout_start = std::chrono::steady_clock::now();

        // Polling loop с таймаутом 60 секунд
        while (!flag) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status);
            if (!flag) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                auto elapsed = std::chrono::steady_clock::now() - timeout_start;
                if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 60) {
                    std::cerr << "MASTER: Timeout waiting for worker response" << std::endl;
                    break;
                }
            }
        }

        if (!flag) {
            // Таймаут - пропускаем
            continue;
        }

        // MPI_Recv с MPI_ANY_SOURCE - получаем результат от любого worker'а
        MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        int worker = status.MPI_SOURCE;
        tasks_completed++;
        all_results.push_back(result);

        std::cout << "MASTER: Получен результат от worker'а " << worker
                  << " для " << result.ip
                  << " (время: " << result.scan_time << "с)" << std::endl;

        // Автоматическое перераспределение: если есть ещё задания, отправляем worker'у
        if (next_task < total_tasks) {
            char ip_buffer[MAX_IP_LEN];
            std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);

            MPI_Send(ip_buffer, MAX_IP_LEN, MPI_CHAR, worker, TAG_TASK, MPI_COMM_WORLD);

            std::cout << "MASTER: Отправил новое задание " << ip_buffer << " worker'у " << worker << std::endl;
            worker_tasks[worker]++;
            next_task++;
            tasks_sent++;
        }
    }

    // Отправляем сигнал завершения всем worker'ам
    for (int worker = 1; worker < world_size; worker++) {
        char terminate_signal[MAX_IP_LEN] = "TERMINATE";
        MPI_Send(terminate_signal, MAX_IP_LEN, MPI_CHAR, worker, TAG_TERMINATE, MPI_COMM_WORLD);
    }

    // Вывод результатов
    std::cout << "\n=== РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ===" << std::endl;

    int successful_scans = 0;
    for (const auto& result : all_results) {
        std::cout << "\n[Worker " << result.worker_rank << "] " << result.ip << ":" << std::endl;
        if (result.success) {
            successful_scans++;
            // Выводим только первые 500 символов результата
            std::string res(result.result);
            if (res.length() > 500) res = res.substr(0, 500) + "...";
            std::cout << res << std::endl;
        } else {
            std::cout << "Хост недоступен или сканирование не удалось" << std::endl;
        }
    }

    // Статистика по worker'ам
    std::cout << "\n=== СТАТИСТИКА ===" << std::endl;
    std::cout << "Успешных сканирований: " << successful_scans << " из " << total_tasks << std::endl;
    std::cout << "\nЗадач на каждого worker'а:" << std::endl;
    for (int w = 1; w < world_size; w++) {
        std::cout << "  Worker " << w << ": " << worker_tasks[w] << " задач" << std::endl;
    }
}

// WORKER процесс - выполняет сканирование по заданиям от master'а
void worker_process(int rank) {
    int scan_rate = 100;  // Начальная скорость (мс задержки)

    while (true) {
        char ip_buffer[MAX_IP_LEN];
        MPI_Status status;

        // MPI_Probe - проверяем тег сообщения БЕЗ его получения
        // Позволяет узнать тип сообщения перед его обработкой
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Если получен сигнал завершения - выходим
        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(ip_buffer, MAX_IP_LEN, MPI_CHAR, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            std::cout << "Worker " << rank << ": Получен сигнал завершения" << std::endl;
            break;
        }

        // Получаем задание (IP для сканирования)
        MPI_Recv(ip_buffer, MAX_IP_LEN, MPI_CHAR, 0, TAG_TASK, MPI_COMM_WORLD, &status);

        std::cout << "Worker " << rank << ": Сканирую " << ip_buffer << std::endl;

        // Выполняем сканирование с текущей скоростью
        ScanResult result = scan_ip(std::string(ip_buffer), rank, scan_rate);

        // Динамическое регулирование скорости:
        // Если сканирование слишком быстрое - увеличиваем задержку
        // Если слишком медленное - уменьшаем
        if (result.scan_time < 1.0 && scan_rate < 500) {
            scan_rate += 50;  // Замедляем
        } else if (result.scan_time > 5.0 && scan_rate > 50) {
            scan_rate -= 25;  // Ускоряем
        }

        // Отправляем результат master'у
        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    // MPI_Init - инициализация MPI, должна быть вызвана первой
    MPI_Init(&argc, &argv);

    int world_size, world_rank;

    // MPI_Comm_size - получаем общее количество процессов
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // MPI_Comm_rank - получаем номер текущего процесса (0 до world_size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        std::cerr << "Требуется минимум 2 процесса (1 master + 1 worker)" << std::endl;
        std::cerr << "Запуск: mpirun -n 4 ./nmap" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0) {
        // Процесс 0 - MASTER (координатор)
        // Генерируем список IP для сканирования (несколько подсетей)
        std::vector<std::string> ip_range = generate_multiple_subnets();

        master_process(world_size, ip_range);
    } else {
        // Остальные процессы - WORKERS (исполнители)
        worker_process(world_rank);
    }

    // MPI_Finalize - завершение работы MPI, должна быть вызвана последней
    MPI_Finalize();

    return 0;
}
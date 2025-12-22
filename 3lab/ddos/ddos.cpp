#include <mpi.h>      // MPI - библиотека для распределённых вычислений
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <climits>
#include <algorithm>  // std::find

// Теги для MPI сообщений
const int TAG_PACKETS = 1;      // Отправка пакетов worker'у
const int TAG_STATS = 2;        // Получение статистики от worker'а
const int TAG_TERMINATE = 3;    // Сигнал завершения работы

// Максимальные размеры для передачи данных
const int MAX_IP_LEN = 16;
const int BATCH_SIZE = 1000;    // Размер сегмента пакетов

// Типы TCP флагов для обнаружения SYN-флуда
const int FLAG_SYN = 1;
const int FLAG_SYN_ACK = 2;
const int FLAG_ACK = 3;
const int FLAG_OTHER = 4;

// Структура сетевого пакета для MPI передачи
struct NetworkPacket {
    char source_ip[MAX_IP_LEN];
    char dest_ip[MAX_IP_LEN];
    int packet_size;
    long timestamp;
    int protocol;       // TCP=1, UDP=2
    int tcp_flags;      // SYN, SYN-ACK, ACK для обнаружения SYN-флуда
};

// Структура для агрегированной статистики от worker'а
struct WorkerStats {
    int worker_rank;
    int packets_processed;
    int attacks_detected;
    int syn_flood_detected;
    // Статистика по IP (передаём топ-подозрительных)
    char top_attacker_ip[MAX_IP_LEN];
    int top_attacker_count;
    double syn_to_synack_ratio;  // Соотношение SYN к SYN-ACK
};

// Класс для обнаружения DDoS атак (используется worker'ами)
class DDOSDetector {
private:
    // Счётчики запросов по IP-адресам
    std::unordered_map<std::string, int> request_count;
    std::unordered_map<std::string, long> first_request_time;

    // Счётчики для обнаружения SYN-флуда
    std::unordered_map<std::string, int> syn_count;      // SYN пакеты по dest_ip
    std::unordered_map<std::string, int> syn_ack_count;  // SYN-ACK пакеты по dest_ip

    const int RATE_LIMIT = 100;           // Порог запросов для обнаружения атаки
    const long TIME_WINDOW = 1000000000;  // Временное окно (1 секунда в наносекундах)
    const double SYN_FLOOD_THRESHOLD = 3.0;  // Порог соотношения SYN/SYN-ACK

    std::vector<std::string> detected_attackers;
    int worker_rank;
    std::ofstream log_file;

public:
    DDOSDetector(int rank) : worker_rank(rank) {
        // Открываем лог-файл для каждого worker'а
        std::string log_name = "ddos_worker_" + std::to_string(rank) + ".log";
        log_file.open(log_name);
        log_file << "=== DDoS Detector Worker " << rank << " Log ===" << std::endl;
    }

    ~DDOSDetector() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    // Логирование и вывод оповещений
    void alert(const std::string& message) {
        std::string full_msg = "[Worker " + std::to_string(worker_rank) + "] ALERT: " + message;
        std::cout << full_msg << std::endl;
        if (log_file.is_open()) {
            log_file << full_msg << std::endl;
        }
    }

    // Анализ пакета на предмет DDoS атаки (rate limiting)
    bool analyze_packet(const NetworkPacket& packet) {
        std::string src_ip(packet.source_ip);
        std::string dst_ip(packet.dest_ip);
        auto now = packet.timestamp;

        // Подсчёт SYN и SYN-ACK для обнаружения SYN-флуда
        if (packet.tcp_flags == FLAG_SYN) {
            syn_count[dst_ip]++;
        } else if (packet.tcp_flags == FLAG_SYN_ACK) {
            syn_ack_count[dst_ip]++;
        }

        // Проверка rate limit по source IP
        if (first_request_time.count(src_ip)) {
            long time_diff = now - first_request_time[src_ip];

            if (time_diff < TIME_WINDOW) {
                request_count[src_ip]++;
                if (request_count[src_ip] > RATE_LIMIT) {
                    if (std::find(detected_attackers.begin(), detected_attackers.end(), src_ip)
                        == detected_attackers.end()) {
                        detected_attackers.push_back(src_ip);
                        alert("Rate limit exceeded for IP: " + src_ip);
                    }
                    return true;
                }
            } else {
                // Сброс счётчика для нового временного окна
                request_count[src_ip] = 1;
                first_request_time[src_ip] = now;
            }
        } else {
            request_count[src_ip] = 1;
            first_request_time[src_ip] = now;
        }
        return false;
    }

    // Обнаружение SYN-флуда: высокое соотношение SYN к SYN-ACK
    bool detect_syn_flood(std::string& target_ip, double& ratio) {
        for (const auto& [ip, syn_cnt] : syn_count) {
            int synack_cnt = syn_ack_count.count(ip) ? syn_ack_count[ip] : 1;
            double current_ratio = static_cast<double>(syn_cnt) / synack_cnt;

            if (current_ratio > SYN_FLOOD_THRESHOLD && syn_cnt > 50) {
                target_ip = ip;
                ratio = current_ratio;
                alert("SYN-flood detected! Target: " + ip +
                      ", SYN/SYN-ACK ratio: " + std::to_string(current_ratio));
                return true;
            }
        }
        return false;
    }

    // Получение статистики для отправки master'у
    WorkerStats get_stats() {
        WorkerStats stats;
        stats.worker_rank = worker_rank;
        stats.packets_processed = 0;
        stats.attacks_detected = detected_attackers.size();
        stats.syn_flood_detected = 0;
        stats.syn_to_synack_ratio = 0.0;
        std::memset(stats.top_attacker_ip, 0, MAX_IP_LEN);

        // Находим топ-атакующего по количеству запросов
        int max_count = 0;
        std::string top_ip;
        for (const auto& [ip, count] : request_count) {
            stats.packets_processed += count;
            if (count > max_count) {
                max_count = count;
                top_ip = ip;
            }
        }

        if (!top_ip.empty()) {
            std::strncpy(stats.top_attacker_ip, top_ip.c_str(), MAX_IP_LEN - 1);
            stats.top_attacker_count = max_count;
        }

        // Проверяем SYN-флуд
        std::string syn_target;
        double ratio;
        if (detect_syn_flood(syn_target, ratio)) {
            stats.syn_flood_detected = 1;
            stats.syn_to_synack_ratio = ratio;
        }

        return stats;
    }

    const std::vector<std::string>& get_attackers() const {
        return detected_attackers;
    }
};

// ============================================================================
// Генератор тестового трафика с имитацией атак
// ============================================================================
std::vector<NetworkPacket> generate_test_packets(int count, bool include_syn_flood = true) {
    std::vector<NetworkPacket> packets;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> ip_octet(1, 255);
    std::uniform_int_distribution<> size_dist(64, 1500);
    std::uniform_int_distribution<> proto_dist(1, 2);
    std::uniform_int_distribution<> flag_dist(1, 4);

    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < count; i++) {
        NetworkPacket pkt;

        // 5% пакетов - от "атакующих" IP (много запросов с одного IP)
        if (i % 100 < 5) {
            std::string src = "192.168.100." + std::to_string(i % 5 + 1);
            std::strncpy(pkt.source_ip, src.c_str(), MAX_IP_LEN - 1);
        } else {
            std::string src = "10.0." + std::to_string(ip_octet(gen)) + "." + std::to_string(ip_octet(gen));
            std::strncpy(pkt.source_ip, src.c_str(), MAX_IP_LEN - 1);
        }
        pkt.source_ip[MAX_IP_LEN - 1] = '\0';

        std::strncpy(pkt.dest_ip, "192.168.1.1", MAX_IP_LEN - 1);
        pkt.dest_ip[MAX_IP_LEN - 1] = '\0';

        pkt.packet_size = size_dist(gen);
        pkt.timestamp = now + (i * 1000000);  // Наносекунды между пакетами
        pkt.protocol = proto_dist(gen);

        // Генерация TCP флагов для SYN-флуда
        if (include_syn_flood && i % 100 < 10) {
            // 10% - SYN пакеты (имитация SYN-флуда)
            pkt.tcp_flags = FLAG_SYN;
        } else if (i % 100 < 12) {
            // 2% - SYN-ACK (нормальный ответ)
            pkt.tcp_flags = FLAG_SYN_ACK;
        } else {
            pkt.tcp_flags = flag_dist(gen);
        }

        packets.push_back(pkt);
    }
    return packets;
}

// MASTER процесс - распределяет пакеты и агрегирует статистику
void master_process(int world_size, int total_packets) {
    std::cout << "=== MASTER: Запуск распределённого анализа DDoS ===" << std::endl;
    std::cout << "Количество worker'ов: " << (world_size - 1) << std::endl;
    std::cout << "Всего пакетов для анализа: " << total_packets << std::endl;

    // Генерируем тестовый трафик
    auto start_gen = std::chrono::high_resolution_clock::now();
    std::vector<NetworkPacket> all_packets = generate_test_packets(total_packets);
    auto end_gen = std::chrono::high_resolution_clock::now();

    std::cout << "Генерация трафика завершена за "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count()
              << " мс" << std::endl;

    // Сегментирование по размеру: делим пакеты на сегменты для каждого worker'а
    int packets_per_worker = total_packets / (world_size - 1);
    int remaining = total_packets % (world_size - 1);

    auto start_analysis = std::chrono::high_resolution_clock::now();

    // MPI_Send - отправляем сегменты пакетов каждому worker'у
    int offset = 0;
    for (int worker = 1; worker < world_size; worker++) {
        int count = packets_per_worker + (worker <= remaining ? 1 : 0);

        // Сначала отправляем количество пакетов
        MPI_Send(&count, 1, MPI_INT, worker, TAG_PACKETS, MPI_COMM_WORLD);

        // Проверка размера данных перед отправкой
        int message_size = count * sizeof(NetworkPacket);
        if (message_size > INT_MAX / 2) {
            std::cerr << "MASTER: Packet batch too large for worker " << worker << std::endl;
            continue;
        }

        // Затем отправляем сами пакеты (NetworkPacket - POD структура)
        MPI_Send(&all_packets[offset], message_size, MPI_BYTE,
                 worker, TAG_PACKETS, MPI_COMM_WORLD);

        std::cout << "MASTER: Отправил " << count << " пакетов worker'у " << worker << std::endl;
        offset += count;
    }

    // Агрегация статистики со всех worker'ов
    std::vector<WorkerStats> all_stats;
    int total_attacks = 0;
    int total_syn_floods = 0;
    int total_processed = 0;

    // MPI_Recv - получаем статистику от каждого worker'а
    for (int worker = 1; worker < world_size; worker++) {
        WorkerStats stats;
        MPI_Status status;

        MPI_Recv(&stats, sizeof(WorkerStats), MPI_BYTE, worker,
                 TAG_STATS, MPI_COMM_WORLD, &status);

        all_stats.push_back(stats);
        total_attacks += stats.attacks_detected;
        total_syn_floods += stats.syn_flood_detected;
        total_processed += stats.packets_processed;

        std::cout << "MASTER: Получил статистику от worker'а " << worker << std::endl;
    }

    auto end_analysis = std::chrono::high_resolution_clock::now();

    // Отправляем сигнал завершения всем worker'ам
    for (int worker = 1; worker < world_size; worker++) {
        int terminate = -1;
        MPI_Send(&terminate, 1, MPI_INT, worker, TAG_TERMINATE, MPI_COMM_WORLD);
    }

    // Вывод глобальной агрегированной статистики
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== ГЛОБАЛЬНАЯ СТАТИСТИКА DDoS АНАЛИЗА ===" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nВремя анализа: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_analysis - start_analysis).count()
              << " мс" << std::endl;
    std::cout << "Обработано пакетов: " << total_processed << std::endl;
    std::cout << "Обнаружено DDoS атак (rate limit): " << total_attacks << std::endl;
    std::cout << "Обнаружено SYN-flood атак: " << total_syn_floods << std::endl;

    std::cout << "\n--- Статистика по worker'ам ---" << std::endl;
    for (const auto& stats : all_stats) {
        std::cout << "Worker " << stats.worker_rank << ":" << std::endl;
        std::cout << "  Пакетов обработано: " << stats.packets_processed << std::endl;
        std::cout << "  Атак обнаружено: " << stats.attacks_detected << std::endl;
        if (stats.top_attacker_count > 0) {
            std::cout << "  Топ-атакующий: " << stats.top_attacker_ip
                      << " (" << stats.top_attacker_count << " пакетов)" << std::endl;
        }
        if (stats.syn_flood_detected) {
            std::cout << "  SYN-flood! SYN/SYN-ACK ratio: " << stats.syn_to_synack_ratio << std::endl;
        }
    }

    // Запись итогов в общий лог
    std::ofstream master_log("ddos_master.log");
    master_log << "=== DDoS Analysis Summary ===" << std::endl;
    master_log << "Total packets: " << total_packets << std::endl;
    master_log << "Workers: " << (world_size - 1) << std::endl;
    master_log << "Attacks detected: " << total_attacks << std::endl;
    master_log << "SYN-floods detected: " << total_syn_floods << std::endl;
    master_log.close();

    if (total_attacks > 0 || total_syn_floods > 0) {
        std::cout << "\n!!! DDoS АТАКА ОБНАРУЖЕНА !!!" << std::endl;
    } else {
        std::cout << "\nDDoS атаки не обнаружены" << std::endl;
    }
}

// WORKER процесс - выполняет локальный анализ полученных пакетов
void worker_process(int rank) {
    DDOSDetector detector(rank);

    while (true) {
        int packet_count;
        MPI_Status status;

        // MPI_Probe - проверяем тег сообщения без его получения
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Если получен сигнал завершения - выходим
        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            std::cout << "Worker " << rank << ": Получен сигнал завершения" << std::endl;
            break;
        }

        // Получаем количество пакетов
        MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

        // Проверка размера перед выделением памяти
        if (packet_count < 0 || packet_count > 10000000) {
            std::cerr << "Worker " << rank << ": Invalid packet count: " << packet_count << std::endl;
            break;
        }

        // Получаем сами пакеты
        std::vector<NetworkPacket> packets(packet_count);
        int message_size = packet_count * sizeof(NetworkPacket);
        MPI_Recv(packets.data(), message_size, MPI_BYTE,
                 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

        std::cout << "Worker " << rank << ": Получил " << packet_count << " пакетов для анализа" << std::endl;

        // Локальный анализ каждого пакета
        for (const auto& packet : packets) {
            detector.analyze_packet(packet);
        }

        // Отправляем статистику master'у
        WorkerStats stats = detector.get_stats();
        MPI_Send(&stats, sizeof(WorkerStats), MPI_BYTE, 0, TAG_STATS, MPI_COMM_WORLD);

        std::cout << "Worker " << rank << ": Анализ завершён, обнаружено атак: "
                  << stats.attacks_detected << std::endl;
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
        std::cerr << "Запуск: mpirun -n 4 ./ddos" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Количество пакетов для анализа (можно передать как аргумент)
    int total_packets = 10000;
    if (argc > 1) {
        total_packets = std::atoi(argv[1]);
    }

    if (world_rank == 0) {
        // Процесс 0 - MASTER (координатор)
        master_process(world_size, total_packets);
    } else {
        // Остальные процессы - WORKERS (исполнители)
        worker_process(world_rank);
    }

    // MPI_Finalize - завершение работы MPI, должна быть вызвана последней
    MPI_Finalize();

    return 0;
}

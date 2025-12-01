# Лабораторная работа 3: MPI (Message Passing Interface)

## Задача 1: Обнаружение DDoS атак (ddos)

### Что конкретно сделал

#### Теги для сообщений и структуры данных (строки 14-49)
```cpp
const int TAG_PACKETS = 1;      // Отправка пакетов worker'у
const int TAG_STATS = 2;        // Получение статистики от worker'а
const int TAG_TERMINATE = 3;    // Сигнал завершения работы

struct NetworkPacket {
    char source_ip[MAX_IP_LEN];
    char dest_ip[MAX_IP_LEN];
    int packet_size;
    long timestamp;
    int protocol;
    int tcp_flags;
};

struct WorkerStats {
    int worker_rank;
    int packets_processed;
    int attacks_detected;
    int syn_flood_detected;
    char top_attacker_ip[MAX_IP_LEN];
    int top_attacker_count;
    double syn_to_synack_ratio;
};
```
В MPI нужно заранее определить теги для разных типов сообщений. Структуры должны быть простыми (POD типы) чтобы их можно было передавать как байты.

#### MASTER процесс — распределение работы
```cpp
void master_process(int world_size, int total_packets) {
    // Генерируем тестовый трафик
    std::vector<NetworkPacket> all_packets = generate_test_packets(total_packets);

    // Делим пакеты между worker'ами
    int packets_per_worker = total_packets / (world_size - 1);
    int remaining = total_packets % (world_size - 1);

    int offset = 0;
    for (int worker = 1; worker < world_size; worker++) {
        int count = packets_per_worker + (worker <= remaining ? 1 : 0);

        // Сначала отправляем количество пакетов
        MPI_Send(&count, 1, MPI_INT, worker, TAG_PACKETS, MPI_COMM_WORLD);

        // Затем отправляем сами пакеты
        MPI_Send(&all_packets[offset], count * sizeof(NetworkPacket), MPI_BYTE,
                 worker, TAG_PACKETS, MPI_COMM_WORLD);

        offset += count;
    }

    // Собираем статистику от всех worker'ов
    std::vector<WorkerStats> all_stats;
    for (int worker = 1; worker < world_size; worker++) {
        WorkerStats stats;
        MPI_Status status;

        MPI_Recv(&stats, sizeof(WorkerStats), MPI_BYTE, worker,
                 TAG_STATS, MPI_COMM_WORLD, &status);

        all_stats.push_back(stats);
    }

    // Отправляем сигнал завершения
    for (int worker = 1; worker < world_size; worker++) {
        int terminate = -1;
        MPI_Send(&terminate, 1, MPI_INT, worker, TAG_TERMINATE, MPI_COMM_WORLD);
    }
}
```

Cхема Master-Worker:
1. **`MPI_Send`** — отправляем данные конкретному процессу. Параметры: данные, размер, тип, кому, тег, коммуникатор.
2. **`MPI_Recv`** — получаем данные. Блокирующая операция — ждём пока придёт сообщение.
3. **Сегментирование по размеру** — делим пакеты поровну между worker'ами.

#### WORKER процесс — анализ пакетов
```cpp
void worker_process(int rank) {
    DDOSDetector detector(rank);

    while (true) {
        int packet_count;
        MPI_Status status;

        // MPI_Probe — смотрим что за сообщение пришло (не забирая его)
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Если сигнал завершения — выходим
        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            break;
        }

        // Получаем количество пакетов
        MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

        // Получаем сами пакеты
        std::vector<NetworkPacket> packets(packet_count);
        MPI_Recv(packets.data(), packet_count * sizeof(NetworkPacket), MPI_BYTE,
                 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

        // Анализируем каждый пакет
        for (const auto& packet : packets) {
            detector.analyze_packet(packet);
        }

        // Отправляем статистику master'у
        WorkerStats stats = detector.get_stats();
        MPI_Send(&stats, sizeof(WorkerStats), MPI_BYTE, 0, TAG_STATS, MPI_COMM_WORLD);
    }
}
```

**`MPI_Probe`** позволяет посмотреть тег сообщения не забирая его. Так можно понять — это данные или сигнал завершения.

#### Инициализация MPI
```cpp
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        master_process(world_size, total_packets);
    } else {
        worker_process(world_rank);
    }

    MPI_Finalize();
    return 0;
}
```

Стандартный шаблон MPI программы:
- **`MPI_Init`** — инициализация, вызываем первой
- **`MPI_Comm_size`** — сколько всего процессов
- **`MPI_Comm_rank`** — мой номер (0 до size-1)
- **`MPI_Finalize`** — завершение, вызываем последней

Процесс 0 обычно master, остальные — worker'ы.

## Задача 2: Сканер IOC (ioc)

### Что конкретно сделал

#### Broadcast IOC базы всем worker'ам
```cpp
// Рассылка IOC всем worker'ам через MPI_Bcast
int ioc_count = iocs.size();
MPI_Bcast(&ioc_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (ioc_count > 0) {
    MPI_Bcast(iocs.data(), ioc_count * sizeof(IOCPattern), MPI_BYTE, 0, MPI_COMM_WORLD);
}
```

**`MPI_Bcast`** — коллективная операция. Один процесс (root=0) отправляет данные ВСЕМ остальным за один вызов. Гораздо эффективнее чем слать каждому по отдельности.

#### Динамическая балансировка нагрузки
```cpp
// Раздаём по одному заданию каждому worker'у
for (int worker = 1; worker < world_size && next_task < total_tasks; worker++) {
    ScanTask task;
    std::strncpy(task.file_path, files[next_task].c_str(), MAX_PATH_LEN - 1);
    task.task_id = next_task;

    MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);
    next_task++;
}

// Получаем результаты и раздаём новые задания
while (tasks_completed < tasks_sent) {
    ScanResult result;
    MPI_Status status;

    // MPI_ANY_SOURCE — получаем от любого worker'а
    MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
             TAG_RESULT, MPI_COMM_WORLD, &status);

    int worker = status.MPI_SOURCE;  // Кто прислал
    tasks_completed++;

    // Отправляем новое задание освободившемуся worker'у
    if (next_task < total_tasks) {
        ScanTask task;
        std::strncpy(task.file_path, files[next_task].c_str(), MAX_PATH_LEN - 1);
        task.task_id = next_task;

        MPI_Send(&task, sizeof(ScanTask), MPI_BYTE, worker, TAG_TASK, MPI_COMM_WORLD);
        next_task++;
        tasks_sent++;
    }
}
```

Это важный паттерн — **динамическая балансировка**:
1. Сначала раздаём каждому по одному заданию
2. Когда кто-то закончил — даём ему следующее
3. **`MPI_ANY_SOURCE`** — принимаем от любого, кто первый закончит
4. **`status.MPI_SOURCE`** — узнаём кто прислал, чтобы дать ему новую работу

Так файлы разного размера обрабатываются эффективно — никто не простаивает.

#### Worker с получением IOC
```cpp
void worker_process(int rank) {
    Logger logger(rank, LOG_DEBUG);

    // Получаем IOC от master'а через MPI_Bcast
    int ioc_count;
    MPI_Bcast(&ioc_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<IOCPattern> iocs(ioc_count);
    if (ioc_count > 0) {
        MPI_Bcast(iocs.data(), ioc_count * sizeof(IOCPattern), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    IOCScanner scanner(&logger);
    scanner.set_iocs(iocs);

    while (true) {
        ScanTask task;
        MPI_Status status;

        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            break;
        }

        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);

        ScanResult result = scanner.scan_file(task, rank);

        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}
```

`MPI_Bcast` вызывается и на master и на worker'ах! Это коллективная операция, все процессы должны её вызвать.

## Задача 3: Распределённый nmap (nmap)

### Что конкретно сделал

#### Динамическое распределение IP-адресов
```cpp
void master_process(int world_size, const std::vector<std::string>& ip_range) {
    std::vector<int> worker_tasks(world_size, 0);  // Счётчик задач

    int next_task = 0;
    int tasks_sent = 0;
    int tasks_completed = 0;

    // Раздаём начальные задания
    for (int worker = 1; worker < world_size && next_task < total_tasks; worker++) {
        char ip_buffer[MAX_IP_LEN];
        std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);

        MPI_Send(ip_buffer, MAX_IP_LEN, MPI_CHAR, worker, TAG_TASK, MPI_COMM_WORLD);

        worker_tasks[worker]++;
        next_task++;
        tasks_sent++;
    }

    // Получаем результаты и раздаём новые задания
    while (tasks_completed < tasks_sent) {
        ScanResult result;
        MPI_Status status;

        MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        int worker = status.MPI_SOURCE;
        tasks_completed++;

        // Автоматическое перераспределение
        if (next_task < total_tasks) {
            char ip_buffer[MAX_IP_LEN];
            std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);

            MPI_Send(ip_buffer, MAX_IP_LEN, MPI_CHAR, worker, TAG_TASK, MPI_COMM_WORLD);

            worker_tasks[worker]++;
            next_task++;
            tasks_sent++;
        }
    }
}
```

Та же динамическая балансировка. Ведём статистику `worker_tasks[worker]` чтобы видеть загрузку.

#### Worker с динамическим регулированием скорости (строки 192-229)
```cpp
void worker_process(int rank) {
    int scan_rate = 100;  // Начальная задержка (мс)

    while (true) {
        char ip_buffer[MAX_IP_LEN];
        MPI_Status status;

        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(ip_buffer, MAX_IP_LEN, MPI_CHAR, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            break;
        }

        MPI_Recv(ip_buffer, MAX_IP_LEN, MPI_CHAR, 0, TAG_TASK, MPI_COMM_WORLD, &status);

        ScanResult result = scan_ip(std::string(ip_buffer), rank, scan_rate);

        // Динамическое регулирование скорости
        if (result.scan_time < 1.0 && scan_rate < 500) {
            scan_rate += 50;  // Замедляем
        } else if (result.scan_time > 5.0 && scan_rate > 50) {
            scan_rate -= 25;  // Ускоряем
        }

        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}
```

Интересная фишка — адаптивная скорость сканирования. Если сканирование слишком быстрое — притормаживаем, чтобы не перегрузить сеть.

---

## Задача 4: Распределённый sqlmap (sqlmap)

### Суть задачи
Параллельное тестирование веб-приложений на SQL-инъекции. Цели берём из XML файла, раздаём worker'ам.

### Файлы
- `sqlmap/sqli.cpp` — основной код
- `sqlmap/xml_parser.cpp` — парсер XML
- `sqlmap/xml_parser.h` — заголовочник парсера

### Что конкретно сделал

#### Профили сканирования (строки 29-86)
```cpp
enum ScanProfile {
    PROFILE_QUICK = 1,      // Быстрое (level 1-2, risk 1)
    PROFILE_DEEP = 2,       // Глубокий аудит (level 3-5, risk 2)
    PROFILE_AGGRESSIVE = 3  // Агрессивное (level 5, risk 3)
};

std::pair<int, int> get_profile_params(int profile) {
    switch (profile) {
        case PROFILE_QUICK:
            return {1, 1};
        case PROFILE_DEEP:
            return {3, 2};
        case PROFILE_AGGRESSIVE:
            return {5, 3};
        default:
            return {1, 1};
    }
}
```

Разные профили для разных ситуаций — быстрый скан или глубокий аудит.

#### Master с повторными попытками при сбоях (строки 206-388)
```cpp
void master_process(int world_size, const std::vector<TargetInfo>& targets, int profile) {
    // ...

    while (tasks_completed < tasks_sent) {
        ScanResult result;
        MPI_Status status;

        MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
                 TAG_RESULT, MPI_COMM_WORLD, &status);

        int worker = status.MPI_SOURCE;
        tasks_completed++;

        // Проверка на сбой и повторный запуск
        if (!result.success) {
            int task_id = result.task_id;
            static std::vector<int> retry_counts(total_tasks, 0);
            retry_counts[task_id]++;

            if (retry_counts[task_id] < MAX_RETRIES) {
                std::cout << "MASTER: Повторный запуск задания " << task_id
                          << " (попытка " << retry_counts[task_id] << ")" << std::endl;

                ScanTask retry_task;
                // ... заполняем retry_task
                retry_task.retry_count = retry_counts[task_id];

                MPI_Send(&retry_task, sizeof(ScanTask), MPI_BYTE, worker, TAG_RETRY, MPI_COMM_WORLD);
                tasks_sent++;
                continue;
            }
        }

        all_results.push_back(result);

        // Раздаём новое задание
        if (next_task < total_tasks) {
            // ...
        }
    }
}
```

Тут добавили **fault tolerance** — если задание провалилось, пробуем ещё раз (до MAX_RETRIES). Отдельный тег `TAG_RETRY` для повторных попыток.

#### Worker с обработкой retry (строки 390-423)
```cpp
void worker_process(int rank, const std::string& cookies) {
    SQLMapRunner runner(rank, cookies);

    while (true) {
        ScanTask task;
        MPI_Status status;

        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_TERMINATE) {
            MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
            break;
        }

        // Получаем задание (обычное или повторное)
        int tag = (status.MPI_TAG == TAG_RETRY) ? TAG_RETRY : TAG_TASK;
        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, tag, MPI_COMM_WORLD, &status);

        if (tag == TAG_RETRY) {
            std::cout << "Worker " << rank << ": Повторное сканирование задания " << task.task_id << std::endl;
        }

        ScanResult result = runner.scan(task);

        MPI_Send(&result, sizeof(ScanResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}
```

Worker умеет принимать и обычные задания и retry. По тегу понимает что это.

---

## Итого по лабе

Основные функции MPI которые использовали:

**Инициализация:**
- **`MPI_Init`** — запуск MPI
- **`MPI_Finalize`** — завершение MPI
- **`MPI_Comm_size`** — сколько процессов
- **`MPI_Comm_rank`** — мой номер

**Point-to-point коммуникация:**
- **`MPI_Send`** — отправка сообщения
- **`MPI_Recv`** — получение сообщения
- **`MPI_Probe`** — проверка сообщения без получения
- **`MPI_ANY_SOURCE`** — принять от любого

**Коллективные операции:**
- **`MPI_Bcast`** — рассылка всем от одного

**Паттерны:**
- Master-Worker — один координирует, остальные работают
- Динамическая балансировка — раздаём работу по мере освобождения
- Fault tolerance — повторные попытки при сбоях

Главное отличие от потоков — процессы не разделяют память, всё через сообщения. Зато можно запускать на разных машинах!

**Запуск:** `mpirun -n 4 ./program` — запускает 4 процесса

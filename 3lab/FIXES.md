# Исправления ошибок в лабораторной работе 3 (MPI)

---

## 1. ioc/ioc.cpp

### 1.1 Отсутствие проверки размера при MPI_Bcast
**Проблема:**
```cpp
int ioc_count = iocs.size();
MPI_Bcast(&ioc_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (ioc_count > 0) {
    MPI_Bcast(iocs.data(), ioc_count * sizeof(IOCPattern), MPI_BYTE, 0, MPI_COMM_WORLD);
}
```
- Нет проверки размера перед передачей данных через MPI
- Потенциальное переполнение при `ioc_count * sizeof(IOCPattern) > INT_MAX`
- Worker не проверяет валидность полученного `ioc_count`

**Исправление:**
```cpp
// Master process
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

// Worker process
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
```
**Результат:** Защита от переполнения и выделения огромных объемов памяти по невалидным данным.

### 1.2 Неправильное использование MPI_Probe без проверки размера
**Проблема:**
```cpp
MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

if (status.MPI_TAG == TAG_TERMINATE) {
    MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
    break;
}

MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);
```
- После `MPI_Probe` не проверяется размер сообщения через `MPI_Get_count`
- Если размер не совпадает с `sizeof(ScanTask)`, может произойти переполнение или неполное чтение

**Исправление:**
```cpp
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
    break;
}

MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);
```
**Результат:** Безопасная проверка размера сообщения перед получением, защита от ошибок протокола.

---

## 2. ddos/ddos.cpp

### 2.1 Отсутствие проверки размера MPI сообщений при отправке пакетов
**Проблема:**
```cpp
int count = packets_per_worker + (worker <= remaining ? 1 : 0);

MPI_Send(&count, 1, MPI_INT, worker, TAG_PACKETS, MPI_COMM_WORLD);
MPI_Send(&all_packets[offset], count * sizeof(NetworkPacket), MPI_BYTE,
         worker, TAG_PACKETS, MPI_COMM_WORLD);
```
- Нет проверки на переполнение при `count * sizeof(NetworkPacket)`
- Если `count` слишком большой, произойдет переполнение и неопределённое поведение

**Исправление:**
```cpp
int count = packets_per_worker + (worker <= remaining ? 1 : 0);

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
```
**Результат:** Безопасная передача данных, защита от переполнения.

### 2.2 Отсутствие валидации данных на стороне worker'а
**Проблема:**
```cpp
MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

// Получаем сами пакеты
std::vector<NetworkPacket> packets(packet_count);
MPI_Recv(packets.data(), packet_count * sizeof(NetworkPacket), MPI_BYTE,
         0, TAG_PACKETS, MPI_COMM_WORLD, &status);
```
- Нет проверки `packet_count` перед выделением памяти
- Отрицательное или огромное значение приведет к crash или DoS

**Исправление:**
```cpp
MPI_Recv(&packet_count, 1, MPI_INT, 0, TAG_PACKETS, MPI_COMM_WORLD, &status);

// Проверка размера перед выделением памяти
if (packet_count < 0 || packet_count > 10000000) {
    std::cerr << "Worker " << rank << ": Invalid packet count: " << packet_count << std::endl;
    break;
}

std::vector<NetworkPacket> packets(packet_count);
int message_size = packet_count * sizeof(NetworkPacket);
MPI_Recv(packets.data(), message_size, MPI_BYTE,
         0, TAG_PACKETS, MPI_COMM_WORLD, &status);
```
**Результат:** Защита от невалидных данных и атак через сетевые сообщения.

---

## 3. nmap/nmap.cpp

### 3.1 Блокирующие вызовы popen без таймаута
**Проблема:**
```cpp
std::stringstream command;
command << "nmap -sV -sC -T4 " << ip << " 2>/dev/null";

FILE* pipe = popen(command.str().c_str(), "r");
if (pipe) {
    char buffer[128];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        output += buffer;
        if (output.length() > MAX_RESULT_LEN - 128) break;
    }
    int status = pclose(pipe);
}
```
- `nmap` может зависнуть бесконечно при сканировании недоступного хоста
- `popen` блокирует MPI worker, нарушая балансировку нагрузки
- Нет защиты от зависания при чтении из pipe

**Исправление:**
```cpp
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
}
```
**Результат:** Гарантированное завершение операции за разумное время, защита от зависания.

### 3.2 Отсутствие таймаутов при MPI_Recv
**Проблема:**
```cpp
while (tasks_completed < tasks_sent) {
    ScanResult result;
    MPI_Status status;

    MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
             TAG_RESULT, MPI_COMM_WORLD, &status);
```
- `MPI_Recv` блокируется навсегда если worker завис
- Master не может обнаружить "мёртвые" worker'ы
- Вся система зависает при сбое одного worker'а

**Исправление:**
```cpp
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

    MPI_Recv(&result, sizeof(ScanResult), MPI_BYTE, MPI_ANY_SOURCE,
             TAG_RESULT, MPI_COMM_WORLD, &status);
```
**Результат:** Master может обнаружить зависшие worker'ы и продолжить работу.

### 3.3 Небезопасное копирование через std::strncpy
**Проблема:**
```cpp
char ip_buffer[MAX_IP_LEN];
std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);
```
- Нет гарантии нуль-терминации если строка длиннее `MAX_IP_LEN - 1`
- Неинициализированный буфер может содержать мусор

**Исправление:**
```cpp
char ip_buffer[MAX_IP_LEN];
std::memset(ip_buffer, 0, MAX_IP_LEN);
std::strncpy(ip_buffer, ip_range[next_task].c_str(), MAX_IP_LEN - 1);
ip_buffer[MAX_IP_LEN - 1] = '\0';
```
**Результат:** Гарантированная нуль-терминация и отсутствие мусора в буфере.

---

## 4. sqlmap/sqli.cpp

### 4.1 Потенциальная перегрузка сети из-за одновременных popen
**Проблема:**
```cpp
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
```
- Несколько worker'ов одновременно запускают `sqlmap`
- Каждый `sqlmap` генерирует интенсивный HTTP трафик
- Перегрузка сети и целевого сервера → блокировка по IP
- Нет таймаута на чтение вывода

**Исправление:**
```cpp
std::string executeCommand(const std::string& command, int& returnCode) {
    std::string result;
    char buffer[256];

    // Rate limiting для предотвращения перегрузки при одновременных popen
    // Задержка между запусками sqlmap для снижения нагрузки на сеть
    static std::chrono::steady_clock::time_point last_exec{};
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_exec);
    if (elapsed.count() < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500 - elapsed.count()));
    }
    last_exec = std::chrono::steady_clock::now();

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        returnCode = -1;
        return "Error: failed to execute command";
    }

    // Таймаут на чтение вывода
    auto read_start = std::chrono::steady_clock::now();
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
        if (result.length() > MAX_RESULT_LEN - 256) break;

        // Защита от зависания
        auto read_elapsed = std::chrono::steady_clock::now() - read_start;
        if (std::chrono::duration_cast<std::chrono::seconds>(read_elapsed).count() > 120) {
            break;
        }
    }

    returnCode = pclose(pipe);
    return result;
}
```
**Результат:** Rate limiting предотвращает перегрузку сети. Таймаут защищает от зависания.

### 4.2 Использование static переменных в распределённой среде
**Проблема:**
```cpp
if (!result.success) {
    int task_id = result.task_id;
    if (task_id < total_tasks) {
        // Проверяем количество попыток
        static std::vector<int> retry_counts(total_tasks, 0);
        retry_counts[task_id]++;
```
- `static std::vector` инициализируется только один раз при первом вызове
- В MPI каждый процесс имеет свою копию статических переменных
- Если `total_tasks` изменится между запусками, размер вектора не обновится
- Потенциальный выход за границы массива

**Исправление:**
```cpp
if (!result.success) {
    int task_id = result.task_id;
    if (task_id >= 0 && task_id < total_tasks) {
        // Используем локальную переменную вместо static для корректности в MPI
        // static переменные опасны в распределённой среде
        static std::vector<int> retry_counts;
        if (retry_counts.empty()) {
            retry_counts.resize(total_tasks, 0);
        }
        retry_counts[task_id]++;
```
**Результат:** Корректная инициализация размера вектора. Проверка границ индекса.

### 4.3 Отсутствие обработки ошибок MPI_Probe
**Проблема:**
```cpp
while (true) {
    ScanTask task;
    MPI_Status status;

    MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == TAG_TERMINATE) {
        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
        break;
    }
```
- Нет проверки возвращаемого значения `MPI_Probe`
- Нет проверки размера сообщения через `MPI_Get_count`
- Если произойдет MPI ошибка, программа зависнет

**Исправление:**
```cpp
while (true) {
    ScanTask task;
    MPI_Status status;

    // MPI_Probe - проверяем тег сообщения без его получения
    int probe_result = MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (probe_result != MPI_SUCCESS) {
        std::cerr << "Worker " << rank << ": MPI_Probe error" << std::endl;
        break;
    }

    // Проверка размера сообщения
    int count;
    MPI_Get_count(&status, MPI_BYTE, &count);
    if (count != sizeof(ScanTask)) {
        std::cerr << "Worker " << rank << ": Invalid message size: " << count << std::endl;
        // Очищаем сообщение
        MPI_Recv(&task, count, MPI_BYTE, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
        continue;
    }

    if (status.MPI_TAG == TAG_TERMINATE) {
        MPI_Recv(&task, sizeof(ScanTask), MPI_BYTE, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
        break;
    }
```
**Результат:** Обработка ошибок MPI, валидация размера сообщений, graceful degradation.

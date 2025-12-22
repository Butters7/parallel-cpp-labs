# Исправления ошибок в лабораторной работе 1

---

## 1. http_light/main.cpp

### 1.1 Утечка ресурсов CURL при исключениях
**Проблема:**
```cpp
std::string make_get(const std::string& url) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    // ...
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return response_data;
}
```
- `curl_global_init/cleanup` вызывались для каждого HTTP-запроса (крайне неэффективно)
- Если между `curl_easy_init` и `curl_easy_cleanup` происходило исключение, ресурс не освобождался

**Исправление:**
```cpp
// RAII обертка для CURL - автоматически освобождает ресурсы
class CurlHandle {
public:
    CurlHandle() : curl(curl_easy_init()) {}
    ~CurlHandle() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;

    CURL* get() { return curl; }
    operator bool() const { return curl != nullptr; }
private:
    CURL* curl;
};

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    // ... весь код программы ...
    curl_global_cleanup();
}
```
**Результат:** RAII гарантирует освобождение ресурсов даже при исключениях. Глобальная инициализация вызывается один раз.

### 1.2 Гонка данных при доступе к processed_pages
**Проблема:**
```cpp
int processed_pages = 0;  // Обычная переменная
// ...
if (processed_pages >= max_pages) {  // Чтение без синхронизации
    done = true;
}
processed_pages++;  // Инкремент под мьютексом, но не атомарный
```

- Несколько потоков могли одновременно читать/изменять `processed_pages`
- Возможна гонка между проверкой и инкрементом

**Исправление:**
```cpp
std::atomic<int> processed_pages{0};
// ...
if (processed_pages.load() >= max_pages) {
    done.store(true);
}
processed_pages.fetch_add(1);  // Атомарный инкремент
```
**Результат:** Безопасный доступ из нескольких потоков без гонок данных.

### 1.3 Неполная защита переменной active_tasks
**Проблема:**
```cpp
int active_tasks = 0;  // Обычная переменная
active_tasks++;        // Под мьютексом, но не атомарно
active_tasks--;
if (to_visit.empty() && active_tasks == 0) {  // Состояние гонки
    done = true;
}
```
- Проверка условия завершения могла дать неверный результат между операциями

**Исправление:**
```cpp
std::atomic<int> active_tasks{0};
std::atomic<bool> done{false};

active_tasks.fetch_add(1);
// ... работа ...
active_tasks.fetch_sub(1);

if (to_visit.empty() && active_tasks.load() == 0) {
    done.store(true);
}
```
**Результат:** Корректная проверка условия завершения работы краулера.

---

## 2. airport.cpp

### 2.1 Гонка данных при доступе к currentTime
**Проблема:**
```cpp
int currentTime;  // Обычная переменная

// В главном потоке:
for (currentTime = 0; currentTime <= SIMULATION_TIME; currentTime += TIME_STEP) {
    lock_guard<mutex> lock(mtx);
    processScheduledFlights();
}

// В рабочих потоках:
flightIt->landingTime = currentTime;  // Чтение без синхронизации!
```
- Главный поток изменяет `currentTime` под своим локом
- Рабочие потоки читают `currentTime` без синхронизации
- Потоки могут видеть устаревшее значение времени

**Исправление:**
```cpp
atomic<int> currentTime;

// В главном потоке:
for (int t = 0; t <= SIMULATION_TIME; t += TIME_STEP) {
    currentTime.store(t);  // Атомарная запись
    // ...
}

// В рабочих потоках:
flightIt->landingTime = currentTime.load();  // Атомарное чтение

// В функциях обработки:
void processScheduledFlights() {
    int time = currentTime.load();  // Локальная копия
    for (auto& flight : flights) {
        if (flight.scheduledTime <= time) {
            // ...
        }
    }
}
```
**Результат:** Все потоки видят актуальное время симуляции.

### 2.2 Активное ожидание в terminalThread
**Проблема:**
```cpp
thread terminalThread([this]() {
    while (simulationRunning) {
        {
            lock_guard<mutex> lock(mtx);
            updateTerminals();
            // ...
        }
        this_thread::sleep_for(chrono::milliseconds(50));  // Активное ожидание!
    }
});
```
- Поток просыпается каждые 50мс независимо от событий
- Трата CPU на пустые проверки
- Задержка до 50мс на обработку событий

**Исправление:**
```cpp
condition_variable terminalCv;  // Отдельная CV для терминалов

thread terminalThread([this]() {
    while (true) {
        unique_lock<mutex> lock(mtx);
        terminalCv.wait(lock, [this]() {
            return !simulationRunning.load() || hasActiveTerminals();
        });

        if (!simulationRunning.load()) break;

        updateTerminals();
        // ... обработка ...
    }
});

// Уведомление при назначении рейса в терминал:
terminal.assignFlight(flight);
terminalCv.notify_all();
```
**Результат:** Поток просыпается только при реальных событиях, нет активного ожидания.

### 2.3 Риск голодания и взаимной блокировки
**Проблема:**
- Отсутствие проверки на наличие активных терминалов
- Потенциальные проблемы с освобождением ресурсов

**Исправление:**
```cpp
bool hasActiveTerminals() const {
    for (const auto& terminal : terminals) {
        if (!terminal.available) {
            return true;
        }
    }
    return false;
}
```
**Результат:** Корректная проверка состояния терминалов для синхронизации.

### 2.4 Неправильное использование флагов
**Проблема:**
```cpp
bool simulationRunning;  // Обычная переменная
```

**Исправление:**
```cpp
atomic<bool> simulationRunning;

simulationRunning.store(false);
cv.notify_all();
terminalCv.notify_all();
```
**Результат:** Безопасная проверка флага из всех потоков.

---

## 3. floyd_warshall.cpp

### 3.1 Ложное разделение кэша (false sharing)
**Проблема:**
```cpp
for (int i = startRow; i < endRow; i++) {
    for (int j = 0; j < n; j++) {
        if (dist[i][j] > dist[i][k] + dist[k][j]) {
            dist[i][j] = dist[i][k] + dist[k][j];  // Много обращений к памяти
        }
    }
}
```
- Многократное чтение `dist[i][k]` и `dist[k][j]` из памяти
- Соседние строки могут находиться в одной cache line, вызывая false sharing

**Исправление:**
```cpp
for (int i = startRow; i < endRow; i++) {
    // Кэшируем значение в регистре процессора
    int dist_ik = dist[i][k];
    if (dist_ik != INT_MAX) {
        for (int j = 0; j < n; j++) {
            int dist_kj = dist[k][j];
            if (dist_kj != INT_MAX) {
                int new_dist = dist_ik + dist_kj;
                if (dist[i][j] > new_dist) {
                    dist[i][j] = new_dist;
                    next[i][j] = next[i][k];
                }
            }
        }
    }
}
```
**Результат:** Уменьшение числа обращений к памяти, лучшая локальность данных.

### 3.2 Перерасход памяти из-за создания потоков на каждой итерации
**Проблема:**
```cpp
for (int k = 0; k < n; k++) {
    vector<thread> threads;

    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t, k]() {
            // ... работа ...
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}
```
- Для каждого k создается `numThreads` новых потоков
- Создание/уничтожение потоков — дорогая операция
- Для матрицы 1000x1000 создается 1000 * numThreads потоков

**Исправление:**
```cpp
barrier sync_point(numThreads);  // Барьер для синхронизации
vector<thread> threads;

for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t]() {
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? n : startRow + rowsPerThread;

        // Каждый поток выполняет ВСЕ итерации k
        for (int k = 0; k < n; k++) {
            // Обработка своих строк
            for (int i = startRow; i < endRow; i++) {
                // ...
            }

            // Ждем все потоки перед следующим k
            sync_point.arrive_and_wait();
        }
    });
}

// Потоки создаются один раз
for (auto& t : threads) {
    t.join();
}
```
**Результат:**
- Потоки создаются один раз
- `std::barrier` обеспечивает синхронизацию между итерациями
- Значительное ускорение за счет отсутствия overhead на создание потоков

### 3.3 Возможная перегрузка системы потоками
**Исправление включает:**
- Использование фиксированного числа потоков (по числу ядер)
- Повторное использование потоков через барьер

---

## 4. word_count.cpp

### 4.1 Несбалансированная нагрузка между потоками
**Проблема:**
```cpp
int linesPerThread = lines.size() / numThreads;
int startLine = t * linesPerThread;
int endLine = (t == numThreads - 1) ? lines.size() : startLine + linesPerThread;

for (int i = startLine; i < endLine; i++) {
    // Обработка строки
}
```
- Статическое деление: каждому потоку назначается фиксированный диапазон строк
- Если в одних строках много слов, а в других мало — дисбаланс нагрузки
- Некоторые потоки заканчивают раньше и простаивают

**Исправление:**
```cpp
std::atomic<size_t> nextLine{0};

threads.emplace_back([&, t]() {
    while (true) {
        size_t lineIndex = nextLine.fetch_add(1);  // Берем следующую строку
        if (lineIndex >= lines.size()) break;

        // Обработка строки lineIndex
        std::istringstream iss(lines[lineIndex]);
        // ...
    }
});
```
**Результат:**
- Динамическая балансировка нагрузки
- Каждый поток берет новую строку, как только освобождается
- Все потоки загружены до конца работы

### 4.2 Последовательное объединение результатов снижает производительность
**Проблема:**
```cpp
std::map<std::string, int> wordCountMap;
for (const auto& localMap : localMaps) {
    for (const auto& pair : localMap) {
        wordCountMap[pair.first] += pair.second;  // Последовательно!
    }
}
```
- После параллельного подсчета идет последовательное слияние
- Может занимать значительное время для больших данных
- Узкое место производительности

**Исправление:**
```cpp
// Иерархическое параллельное слияние (бинарное дерево)
int numMaps = numThreads;
while (numMaps > 1) {
    std::vector<std::thread> mergeThreads;
    int pairs = numMaps / 2;

    // Сливаем пары карт параллельно
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

wordCountMap = std::move(localMaps[0]);
```
**Результат:**
- Слияние O(log N) этапов вместо O(N)
- Параллельное слияние пар карт
- Значительное ускорение на больших объемах данных

### 4.3 Отсутствие обработки исключений при stoi
**Проблема:**
```cpp
if (argc > 2) {
    topN = std::stoi(argv[2]);  // Может выбросить исключение!
}
```
- Если пользователь передает невалидное число, программа падает
- Нет проверки на отрицательные значения

**Исправление:**
```cpp
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
```
**Результат:** Корректная обработка некорректного ввода пользователя.

### 4.4 Неэффективное использование памяти
**Проблема:**
- Загрузка всего файла в память может быть проблемой для очень больших файлов

**Частичное решение:**
```cpp
if (lines.empty()) {
    std::cout << "Файл пуст" << std::endl;
    return 0;
}
```
- Добавлена проверка на пустой файл
- Для production можно добавить потоковую обработку файла

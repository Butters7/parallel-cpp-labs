# Лабораторная работа 1: Многопоточка на C++ (std::thread)

## Задача 1: Алгоритм Флойда-Уоршелла (floyd_warshall)

### Что конкретно сделал

#### Определяем сколько потоков создавать
```cpp
int numThreads = thread::hardware_concurrency();
if (numThreads == 0) numThreads = 4;
if (numThreads > n) numThreads = n;
```
Тут смотрим сколько ядер на проце через `hardware_concurrency()`. Если функция вернула 0, то ставим 4 по дефолту. Ну и если вершин меньше чем ядер — нет смысла создавать лишние потоки.

#### Параллелим обработку строк матрицы
```cpp
for (int t = 0; t < numThreads; t++) {
    threads.emplace_back([&, t, k]() {
        int rowsPerThread = n / numThreads;
        int startRow = t * rowsPerThread;
        int endRow = (t == numThreads - 1) ? n : startRow + rowsPerThread;

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    if (dist[i][j] > dist[i][k] + dist[k][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                        next[i][j] = next[i][k];
                    }
                }
            }
        }
    });
}
```
Делим матрицу на части — каждый поток обрабатывает свои строки. Например, если 4 потока и 100 строк, то первый берёт 0-24, второй 25-49 и т.д. Последний поток забирает остаток (если не делится нацело).

Мьютексы тут не нужны — потоки пишут в разные строки, конфликтов нет.

Важный момент с лямбдой: `[&, t, k]` — тут `t` и `k` копируем, а не берём по ссылке. Иначе все потоки увидят последнее значение `t` из цикла и будет баг.

#### Ждём пока все потоки закончат
```cpp
for (auto& t : threads) {
    t.join();
}
```
Это барьер — пока все не закончат текущую итерацию по `k`, дальше не идём. Это важно, потому что на следующей итерации нужны обновлённые значения из строки `k`.

## Задача 2: Подсчёт слов (word_count)

### Что конкретно сделал

#### Читаем файл целиком в память
```cpp
std::vector<std::string> lines;
std::string line;
while (std::getline(file, line)) {
    lines.push_back(line);
}
file.close();
```
Сначала грузим весь файл в вектор строк. Так проще потом делить между потоками.

#### Создаём отдельную map для каждого потока
```cpp
std::vector<std::map<std::string, int>> localMaps(numThreads);
```
У каждого потока своя map. Так не нужно лочить общую map при каждой записи.

#### Каждый поток обрабатывает свой кусок файла
```cpp
threads.emplace_back([&, t]() {
    int linesPerThread = lines.size() / numThreads;
    int startLine = t * linesPerThread;
    int endLine = (t == numThreads - 1) ? lines.size() : startLine + linesPerThread;

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
```
Каждый поток берёт свой диапазон строк и считает слова в свою локальную map. 

#### Склеиваем результаты
```cpp
std::map<std::string, int> wordCountMap;
for (const auto& localMap : localMaps) {
    for (const auto& pair : localMap) {
        wordCountMap[pair.first] += pair.second;
    }
}
```
Когда все потоки отработали, сливаем их результаты в одну map. Это уже последовательно, но оно быстро — просто пробежаться по готовым данным.

## Задача 3: HTTP-краулер (http_light)

### Что конкретно сделал

#### Общие данные + синхронизация
```cpp
std::set<std::string> visited;
std::queue<std::string> to_visit;
std::vector<std::string> all_links;

std::mutex mtx;
std::condition_variable cv;
int active_tasks = 0;
```
Тут у нас:
- `visited` — какие URL уже обошли (чтобы не ходить дважды)
- `to_visit` — очередь URL на обработку
- `mtx` — мьютекс чтобы потоки не поломали общие данные
- `cv` — condition variable чтобы будить потоки когда появляются новые URL
- `active_tasks` — считаем сколько потоков сейчас работают (нужно чтобы понять когда всё закончилось)

#### Рабочие потоки
```cpp
threads.emplace_back([&]() {
    while (true) {
        std::string current_url;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() {
                return !to_visit.empty() || done;
            });

            if (done && to_visit.empty()) return;

            current_url = to_visit.front();
            to_visit.pop();
            active_tasks++;
        }

        std::vector<std::string> new_links = extractURLs(current_url, target_domain);

        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const std::string& link : new_links) {
                if (visited.find(link) == visited.end()) {
                    visited.insert(link);
                    to_visit.push(link);
                    all_links.push_back(link);
                }
            }
            active_tasks--;

            if (to_visit.empty() && active_tasks == 0) {
                done = true;
            }
        }
        cv.notify_all();
    }
});
```

1. Поток ждёт на `cv.wait()` пока не появится URL в очереди
2. Берёт URL из очереди (под локом!)
3. Качает страницу и парсит ссылки (это долго, лок отпущен)
4. Добавляет новые ссылки в очередь (опять под локом)
5. Будит других потоков через `notify_all()`

Важно: `unique_lock` используем потому что `cv.wait()` должен уметь отпускать мьютекс пока ждёт.

#### Пул потоков
```cpp
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock,
                        [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}
```
Классический пул потоков — воркеры сидят и ждут задачи из очереди. Когда появляется задача — кто-то её хватает и выполняет.

## Задача 4: Симуляция аэропорта (airport)

#### Поток для посадок
```cpp
thread landingThread([this]() {
    while (true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this]() {
            return !landingQueue.empty() || !simulationRunning;
        });

        if (!simulationRunning && landingQueue.empty()) break;

        if (!landingQueue.empty()) {
            string flightId = landingQueue.front();
            auto flightIt = findFlight(flightId);

            if (flightIt != flights.end()) {
                for (auto& runway : runways) {
                    if (runway.available && runway.requestLanding(*flightIt)) {
                        cout << "[Поток посадки] Рейс " << flightId
                             << " начал посадку на полосу " << runway.id << endl;
                        landingQueue.pop();
                        break;
                    }
                }
            }
        }
    }
});
```
Отдельный поток следит за очередью на посадку.

#### Поток для терминалов
```cpp
thread terminalThread([this]() {
    while (simulationRunning) {
        {
            lock_guard<mutex> lock(mtx);
            updateTerminals();

            for (auto& flight : flights) {
                if (flight.status == FlightStatus::AT_TERMINAL && flight.terminalId > 0) {
                    Terminal& terminal = terminals[flight.terminalId - 1];

                    if (terminal.timeRemaining == 0) {
                        flight.status = FlightStatus::READY_FOR_TAKEOFF;
                        cout << "[Поток терминалов] Рейс " << flight.id
                             << " готов к вылету." << endl;
                        terminal.release();
                        takeoffQueue.push(flight.id);
                        cv.notify_all();
                    }
                }
            }
        }
        this_thread::sleep_for(chrono::milliseconds(50));
    }
});
```
Этот поток периодически проверяет терминалы

#### Главный цикл
```cpp
for (currentTime = 0; currentTime <= SIMULATION_TIME; currentTime += TIME_STEP) {
    {
        lock_guard<mutex> lock(mtx);
        processScheduledFlights();
        processLandingCompletion();
        processTakeoffCompletion();
    }
    cv.notify_all();
    displayStatus();
    this_thread::sleep_for(chrono::milliseconds(100));
}
```
Основной цикл двигает время и обрабатывает события. После каждого шага будит рабочие потоки.

#### Нормальное завершение
```cpp
{
    lock_guard<mutex> lock(mtx);
    simulationRunning = false;
}
cv.notify_all();

landingThread.join();
takeoffThread.join();
terminalThread.join();
```

Ставим флаг завершения, будим все потоки чтобы они увидели флаг и завершились, ждём их через join.

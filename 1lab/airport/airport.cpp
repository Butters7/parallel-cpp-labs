#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <exception>

using namespace std;
using namespace std::chrono;

// Константы для настройки имитации
const int NUM_RUNWAYS = 3;
const int NUM_TERMINALS = 5;
const int NUM_AIRCRAFTS = 15;
const int SIMULATION_TIME = 120;
const int TIME_STEP = 1;

// Перечисление для статусов рейсов
enum class FlightStatus {
    SCHEDULED,
    LANDING,
    LANDED,
    AT_TERMINAL,
    READY_FOR_TAKEOFF,
    TAKING_OFF,
    DEPARTED
};

// Класс для представления рейса
class Flight {
public:
    string id;
    FlightStatus status;
    int scheduledTime;
    int landingTime;
    int terminalTime;
    int takeoffTime;
    int passengers;
    int terminalId;
    int runwayId;

    Flight(string flightId, int time)
        : id(flightId), status(FlightStatus::SCHEDULED), scheduledTime(time),
          landingTime(0), terminalTime(0), takeoffTime(0), passengers(0),
          terminalId(-1), runwayId(-1) {
        passengers = 100 + rand() % 200;
    }

    string getStatusString() const {
        switch(status) {
            case FlightStatus::SCHEDULED: return "По расписанию";
            case FlightStatus::LANDING: return "Посадка";
            case FlightStatus::LANDED: return "Приземлился";
            case FlightStatus::AT_TERMINAL: return "В терминале";
            case FlightStatus::READY_FOR_TAKEOFF: return "Готов к вылету";
            case FlightStatus::TAKING_OFF: return "Взлетает";
            case FlightStatus::DEPARTED: return "Вылетел";
            default: return "Неизвестно";
        }
    }
};

// Класс для взлетно-посадочной полосы
class Runway {
public:
    int id;
    bool available;
    string currentFlight;

    Runway(int runwayId) : id(runwayId), available(true), currentFlight("") {}

    bool requestLanding(Flight& flight) {
        if (!available) return false;

        available = false;
        currentFlight = flight.id;
        flight.runwayId = id;
        flight.status = FlightStatus::LANDING;
        return true;
    }

    bool requestTakeoff(Flight& flight) {
        if (!available) return false;

        available = false;
        currentFlight = flight.id;
        flight.runwayId = id;
        flight.status = FlightStatus::TAKING_OFF;
        return true;
    }

    void release() {
        available = true;
        currentFlight = "";
    }
};

// Класс для терминала
class Terminal {
public:
    int id;
    bool available;
    string currentFlight;
    int processingTime;
    int timeRemaining;

    Terminal(int terminalId) : id(terminalId), available(true), currentFlight(""),
                               processingTime(0), timeRemaining(0) {}

    bool assignFlight(Flight& flight) {
        if (!available) return false;

        available = false;
        currentFlight = flight.id;
        flight.terminalId = id;
        flight.status = FlightStatus::AT_TERMINAL;

        processingTime = 5 + rand() % 6;
        timeRemaining = processingTime;

        return true;
    }

    void process() {
        if (!available && timeRemaining > 0) {
            timeRemaining--;
        }
    }

    bool isProcessingComplete() const {
        return !available && timeRemaining <= 0;
    }

    void release() {
        available = true;
        currentFlight = "";
        timeRemaining = 0;
    }
};

// Класс для управления аэропортом
class Airport {
private:
    vector<Runway> runways;
    vector<Terminal> terminals;
    vector<Flight> flights;
    queue<string> landingQueue;
    queue<string> takeoffQueue;
    atomic<int> currentTime;

    // Единый мьютекс для всех данных
    mutable mutex mtx;
    // Единый condition_variable для всех потоков
    condition_variable cv;

    atomic<bool> simulationRunning;

    // Мьютекс для вывода в консоль
    mutex coutMtx;

public:
    Airport() : currentTime(0), simulationRunning(false) {
        for (int i = 0; i < NUM_RUNWAYS; i++) {
            runways.emplace_back(i + 1);
        }

        for (int i = 0; i < NUM_TERMINALS; i++) {
            terminals.emplace_back(i + 1);
        }

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, SIMULATION_TIME / 2);

        for (int i = 0; i < NUM_AIRCRAFTS; i++) {
            string flightId = "SU" + to_string(1000 + i);
            int scheduledTime = dis(gen);
            flights.emplace_back(flightId, scheduledTime);
        }

        sort(flights.begin(), flights.end(),
             [](const Flight& a, const Flight& b) {
                 return a.scheduledTime < b.scheduledTime;
             });
    }

    void runSimulation() {
        {
            lock_guard<mutex> lock(coutMtx);
            cout << "Запуск имитации работы аэропорта..." << endl;
            cout << "Длительность: " << SIMULATION_TIME << " единиц времени" << endl;
            cout << "Количество ВПП: " << NUM_RUNWAYS << endl;
            cout << "Количество терминалов: " << NUM_TERMINALS << endl;
            cout << "Количество рейсов: " << NUM_AIRCRAFTS << endl << endl;
        }

        simulationRunning = true;

        // Поток для обработки посадок
        thread landingThread([this]() {
            try {
                while (true) {
                    unique_lock<mutex> lock(mtx);

                    cv.wait(lock, [this]() {
                        return !landingQueue.empty() || !simulationRunning.load();
                    });

                    if (!simulationRunning.load() && landingQueue.empty()) break;

                    if (!landingQueue.empty()) {
                        string flightId = landingQueue.front();

                        // Ищем рейс под мьютексом
                        Flight* flightPtr = findFlightPtr(flightId);

                        if (flightPtr != nullptr) {
                            bool landed = false;
                            for (auto& runway : runways) {
                                if (runway.available && runway.requestLanding(*flightPtr)) {
                                    flightPtr->landingTime = currentTime.load();
                                    landingQueue.pop();
                                    landed = true;

                                    // Выводим под отдельным мьютексом
                                    {
                                        lock_guard<mutex> coutLock(coutMtx);
                                        cout << "[Поток посадки] Рейс " << flightId
                                             << " начал посадку на полосу " << runway.id << endl;
                                    }
                                    break;
                                }
                            }
                            // Если не удалось посадить - оставляем в очереди
                        } else {
                            // Рейс не найден - удаляем из очереди
                            landingQueue.pop();
                        }
                    }
                }
            } catch (const exception& e) {
                lock_guard<mutex> lock(coutMtx);
                cerr << "[Поток посадки] Исключение: " << e.what() << endl;
            }
        });

        // Поток для обработки взлётов
        thread takeoffThread([this]() {
            try {
                while (true) {
                    unique_lock<mutex> lock(mtx);

                    cv.wait(lock, [this]() {
                        return !takeoffQueue.empty() || !simulationRunning.load();
                    });

                    if (!simulationRunning.load() && takeoffQueue.empty()) break;

                    if (!takeoffQueue.empty()) {
                        string flightId = takeoffQueue.front();

                        Flight* flightPtr = findFlightPtr(flightId);

                        if (flightPtr != nullptr && flightPtr->status == FlightStatus::READY_FOR_TAKEOFF) {
                            bool tookOff = false;
                            for (auto& runway : runways) {
                                if (runway.available && runway.requestTakeoff(*flightPtr)) {
                                    flightPtr->takeoffTime = currentTime.load();
                                    takeoffQueue.pop();
                                    tookOff = true;

                                    {
                                        lock_guard<mutex> coutLock(coutMtx);
                                        cout << "[Поток взлёта] Рейс " << flightId
                                             << " начал взлет с полосы " << runway.id << endl;
                                    }
                                    break;
                                }
                            }
                        } else if (flightPtr == nullptr) {
                            takeoffQueue.pop();
                        }
                        // Если статус неправильный - оставляем в очереди
                    }
                }
            } catch (const exception& e) {
                lock_guard<mutex> lock(coutMtx);
                cerr << "[Поток взлёта] Исключение: " << e.what() << endl;
            }
        });

        // Поток для обработки терминалов
        thread terminalThread([this]() {
            try {
                while (true) {
                    unique_lock<mutex> lock(mtx);

                    // Используем тот же cv с проверкой активных терминалов
                    cv.wait_for(lock, chrono::milliseconds(50), [this]() {
                        return !simulationRunning.load() || hasActiveTerminals();
                    });

                    if (!simulationRunning.load() && !hasActiveTerminals()) break;

                    // Обновляем терминалы
                    for (auto& terminal : terminals) {
                        terminal.process();
                    }

                    // Проверяем готовность рейсов к вылету
                    for (auto& flight : flights) {
                        if (flight.status == FlightStatus::AT_TERMINAL) {
                            int termIdx = flight.terminalId - 1;

                            // Проверяем корректность индекса
                            if (termIdx >= 0 && termIdx < static_cast<int>(terminals.size())) {
                                Terminal& terminal = terminals[termIdx];

                                if (terminal.isProcessingComplete()) {
                                    flight.status = FlightStatus::READY_FOR_TAKEOFF;
                                    terminal.release();
                                    flight.terminalId = -1;
                                    takeoffQueue.push(flight.id);

                                    {
                                        lock_guard<mutex> coutLock(coutMtx);
                                        cout << "[Поток терминалов] Рейс " << flight.id
                                             << " готов к вылету." << endl;
                                    }

                                    // Уведомляем поток взлёта
                                    cv.notify_all();
                                }
                            }
                        }
                    }
                }
            } catch (const exception& e) {
                lock_guard<mutex> lock(coutMtx);
                cerr << "[Поток терминалов] Исключение: " << e.what() << endl;
            }
        });

        // Основной цикл симуляции
        for (int t = 0; t <= SIMULATION_TIME; t += TIME_STEP) {
            currentTime.store(t);

            {
                lock_guard<mutex> lock(mtx);
                processScheduledFlights();
                processLandingCompletion();
                processTakeoffCompletion();
            }

            // Уведомляем все потоки
            cv.notify_all();

            displayStatus();
            this_thread::sleep_for(chrono::milliseconds(100));
        }

        // Завершаем симуляцию
        simulationRunning.store(false);
        cv.notify_all();

        // Ожидаем завершения всех потоков
        landingThread.join();
        takeoffThread.join();
        terminalThread.join();

        {
            lock_guard<mutex> lock(coutMtx);
            cout << "Имитация завершена." << endl;
        }
        displayStatistics();
    }

private:
    bool hasActiveTerminals() const {
        for (const auto& terminal : terminals) {
            if (!terminal.available) {
                return true;
            }
        }
        return false;
    }

    // Возвращает указатель на рейс или nullptr
    Flight* findFlightPtr(const string& flightId) {
        for (auto& flight : flights) {
            if (flight.id == flightId) {
                return &flight;
            }
        }
        return nullptr;
    }

    void processScheduledFlights() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::SCHEDULED && flight.scheduledTime <= time) {
                {
                    lock_guard<mutex> coutLock(coutMtx);
                    cout << "Рейс " << flight.id << " запрашивает посадку." << endl;
                }
                landingQueue.push(flight.id);
                flight.status = FlightStatus::LANDED; // В ожидании посадки
            }
        }
    }

    void processLandingCompletion() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::LANDING &&
                flight.landingTime > 0 &&
                time - flight.landingTime >= 2) {

                flight.status = FlightStatus::LANDED;

                {
                    lock_guard<mutex> coutLock(coutMtx);
                    cout << "Рейс " << flight.id << " завершил посадку." << endl;
                }

                // Освобождаем полосу с проверкой индекса
                int runwayIdx = flight.runwayId - 1;
                if (runwayIdx >= 0 && runwayIdx < static_cast<int>(runways.size())) {
                    runways[runwayIdx].release();
                }
                flight.runwayId = -1;

                // Назначаем терминал
                for (auto& terminal : terminals) {
                    if (terminal.available && terminal.assignFlight(flight)) {
                        {
                            lock_guard<mutex> coutLock(coutMtx);
                            cout << "Рейс " << flight.id << " направлен в терминал "
                                 << terminal.id << endl;
                        }
                        flight.terminalTime = time;
                        break;
                    }
                }
            }
        }
    }

    void processTakeoffCompletion() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::TAKING_OFF &&
                flight.takeoffTime > 0 &&
                time - flight.takeoffTime >= 2) {

                flight.status = FlightStatus::DEPARTED;

                {
                    lock_guard<mutex> coutLock(coutMtx);
                    cout << "Рейс " << flight.id << " вылетел." << endl;
                }

                // Освобождаем полосу с проверкой индекса
                int runwayIdx = flight.runwayId - 1;
                if (runwayIdx >= 0 && runwayIdx < static_cast<int>(runways.size())) {
                    runways[runwayIdx].release();
                }
                flight.runwayId = -1;
            }
        }
    }

    void displayStatus() {
        int time = currentTime.load();
        if (time % 10 == 0) {
            lock_guard<mutex> lock(mtx);
            lock_guard<mutex> coutLock(coutMtx);

            cout << "\n=== Время: " << time << " ===" << endl;
            cout << "Очередь на посадку: " << landingQueue.size() << endl;
            cout << "Очередь на взлет: " << takeoffQueue.size() << endl;

            cout << "Полосы: ";
            for (const auto& runway : runways) {
                cout << "[" << runway.id << ": "
                     << (runway.available ? "Свободна" : runway.currentFlight) << "] ";
            }
            cout << endl;

            cout << "Терминалы: ";
            for (const auto& terminal : terminals) {
                cout << "[" << terminal.id << ": "
                     << (terminal.available ? "Свободен" : terminal.currentFlight);
                if (!terminal.available) {
                    cout << " (" << max(0, terminal.timeRemaining) << ")";
                }
                cout << "] ";
            }
            cout << endl;

            cout << "Рейсы:" << endl;
            for (const auto& flight : flights) {
                if (flight.status != FlightStatus::DEPARTED) {
                    cout << "  " << flight.id << ": " << flight.getStatusString();
                    if (flight.terminalId > 0) cout << " (Терминал " << flight.terminalId << ")";
                    if (flight.runwayId > 0) cout << " (Полоса " << flight.runwayId << ")";
                    cout << endl;
                }
            }
            cout << "======================" << endl;
        }
    }

    void displayStatistics() {
        lock_guard<mutex> lock(mtx);
        lock_guard<mutex> coutLock(coutMtx);

        cout << "\n=== СТАТИСТИКА ===" << endl;

        int departedCount = 0;
        int totalWaitTime = 0;

        for (const auto& flight : flights) {
            if (flight.status == FlightStatus::DEPARTED) {
                departedCount++;
                int totalTime = flight.takeoffTime - flight.scheduledTime;
                totalWaitTime += totalTime;
                cout << flight.id << ": " << totalTime << " ед. времени от запроса до вылета" << endl;
            } else {
                cout << flight.id << ": не завершил обработку (" << flight.getStatusString() << ")" << endl;
            }
        }

        cout << "Обработано рейсов: " << departedCount << " из " << NUM_AIRCRAFTS << endl;
        if (departedCount > 0) {
            cout << "Среднее время обработки: " << totalWaitTime / departedCount << " ед. времени" << endl;
        }
    }
};

int main() {
    srand(static_cast<unsigned>(time(nullptr)));
    setlocale(LC_ALL, "Russian");

    Airport airport;
    airport.runSimulation();

    return 0;
}

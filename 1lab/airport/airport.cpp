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

using namespace std;
using namespace std::chrono;

// Константы для настройки имитации
const int NUM_RUNWAYS = 3;
const int NUM_TERMINALS = 5;
const int NUM_AIRCRAFTS = 15;
const int SIMULATION_TIME = 120; // в условных единицах времени
const int TIME_STEP = 1; // шаг времени

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
        // Генерируем случайное количество пассажиров
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
        
        // Устанавливаем время обработки пассажиров
        processingTime = 5 + rand() % 6; // 5-10 единиц времени
        timeRemaining = processingTime;
        
        return true;
    }

    // Уменьшаем время обработки, но НЕ освобождаем терминал автоматически
    // Терминал освободится когда рейс перейдёт в статус READY_FOR_TAKEOFF
    void process() {
        if (!available && timeRemaining > 0) {
            timeRemaining--;
        }
    }

    // Освобождение терминала вызывается явно
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

    mutex mtx;
    condition_variable cv;
    condition_variable terminalCv;

    atomic<bool> simulationRunning;

public:
    Airport() : currentTime(0), simulationRunning(false) {
        // Инициализируем взлетно-посадочные полосы
        for (int i = 0; i < NUM_RUNWAYS; i++) {
            runways.emplace_back(i + 1);
        }

        // Инициализируем терминалы
        for (int i = 0; i < NUM_TERMINALS; i++) {
            terminals.emplace_back(i + 1);
        }

        // Создаем рейсы со случайным временем прибытия
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, SIMULATION_TIME / 2);

        for (int i = 0; i < NUM_AIRCRAFTS; i++) {
            string flightId = "SU" + to_string(1000 + i);
            int scheduledTime = dis(gen);
            flights.emplace_back(flightId, scheduledTime);
        }

        // Сортируем рейсы по времени прибытия
        sort(flights.begin(), flights.end(),
             [](const Flight& a, const Flight& b) {
                 return a.scheduledTime < b.scheduledTime;
             });
    }
    
    void runSimulation() {
        cout << "Запуск имитации работы аэропорта..." << endl;
        cout << "Длительность: " << SIMULATION_TIME << " единиц времени" << endl;
        cout << "Количество ВПП: " << NUM_RUNWAYS << endl;
        cout << "Количество терминалов: " << NUM_TERMINALS << endl;
        cout << "Количество рейсов: " << NUM_AIRCRAFTS << endl << endl;

        simulationRunning = true;

        // Поток для обработки посадок - обрабатывает очередь на посадку
        thread landingThread([this]() {
            while (true) {
                unique_lock<mutex> lock(mtx);
                cv.wait(lock, [this]() {
                    return !landingQueue.empty() || !simulationRunning.load();
                });

                if (!simulationRunning.load() && landingQueue.empty()) break;

                if (!landingQueue.empty()) {
                    string flightId = landingQueue.front();
                    auto flightIt = findFlight(flightId);

                    if (flightIt != flights.end()) {
                        for (auto& runway : runways) {
                            if (runway.available && runway.requestLanding(*flightIt)) {
                                cout << "[Поток посадки] Рейс " << flightId << " начал посадку на полосу " << runway.id << endl;
                                flightIt->landingTime = currentTime.load();
                                landingQueue.pop();
                                break;
                            }
                        }
                    }
                }
            }
        });

        // Поток для обработки взлётов - обрабатывает очередь на взлёт
        thread takeoffThread([this]() {
            while (true) {
                unique_lock<mutex> lock(mtx);
                cv.wait(lock, [this]() {
                    return !takeoffQueue.empty() || !simulationRunning.load();
                });

                if (!simulationRunning.load() && takeoffQueue.empty()) break;

                if (!takeoffQueue.empty()) {
                    string flightId = takeoffQueue.front();
                    auto flightIt = findFlight(flightId);

                    if (flightIt != flights.end()) {
                        for (auto& runway : runways) {
                            if (runway.available && runway.requestTakeoff(*flightIt)) {
                                cout << "[Поток взлёта] Рейс " << flightId << " начал взлет с полосы " << runway.id << endl;
                                flightIt->takeoffTime = currentTime.load();
                                takeoffQueue.pop();
                                break;
                            }
                        }
                    }
                }
            }
        });

        // Поток для обработки терминалов - обновляет состояние терминалов
        thread terminalThread([this]() {
            while (true) {
                unique_lock<mutex> lock(mtx);
                terminalCv.wait(lock, [this]() {
                    return !simulationRunning.load() || hasActiveTerminals();
                });

                if (!simulationRunning.load()) break;

                updateTerminals();

                // Проверяем готовность рейсов к вылету
                for (auto& flight : flights) {
                    if (flight.status == FlightStatus::AT_TERMINAL && flight.terminalId > 0) {
                        Terminal& terminal = terminals[flight.terminalId - 1];

                        if (terminal.timeRemaining == 0) {
                            flight.status = FlightStatus::READY_FOR_TAKEOFF;
                            cout << "[Поток терминалов] Рейс " << flight.id << " готов к вылету." << endl;
                            terminal.release();
                            flight.terminalId = -1;
                            takeoffQueue.push(flight.id);
                            cv.notify_all();
                        }
                    }
                }
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
            cv.notify_all();
            terminalCv.notify_all();
            displayStatus();
            this_thread::sleep_for(chrono::milliseconds(100));
        }

        // Завершаем симуляцию
        simulationRunning.store(false);
        cv.notify_all();
        terminalCv.notify_all();

        // Ожидаем завершения всех потоков
        landingThread.join();
        takeoffThread.join();
        terminalThread.join();

        cout << "Имитация завершена." << endl;
        displayStatistics();
    }
    
private:
    // Проверяет есть ли активные терминалы с рейсами
    bool hasActiveTerminals() const {
        for (const auto& terminal : terminals) {
            if (!terminal.available) {
                return true;
            }
        }
        return false;
    }

    // Обработка рейсов по расписанию - добавляет в очередь на посадку
    void processScheduledFlights() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::SCHEDULED && flight.scheduledTime <= time) {
                cout << "Рейс " << flight.id << " запрашивает посадку." << endl;
                landingQueue.push(flight.id);
                flight.status = FlightStatus::LANDED;
            }
        }
    }

    // Обработка завершения посадки и назначение терминала
    void processLandingCompletion() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::LANDING &&
                flight.landingTime > 0 &&
                time - flight.landingTime >= 2) {

                flight.status = FlightStatus::LANDED;
                cout << "Рейс " << flight.id << " завершил посадку." << endl;

                if (flight.runwayId > 0) {
                    runways[flight.runwayId - 1].release();
                    flight.runwayId = -1;
                }

                for (auto& terminal : terminals) {
                    if (terminal.available && terminal.assignFlight(flight)) {
                        cout << "Рейс " << flight.id << " направлен в терминал " << terminal.id << endl;
                        flight.terminalTime = time;
                        terminalCv.notify_all();
                        break;
                    }
                }
            }
        }
    }

    // Обработка завершения взлёта
    void processTakeoffCompletion() {
        int time = currentTime.load();
        for (auto& flight : flights) {
            if (flight.status == FlightStatus::TAKING_OFF &&
                flight.takeoffTime > 0 &&
                time - flight.takeoffTime >= 2) {

                flight.status = FlightStatus::DEPARTED;
                cout << "Рейс " << flight.id << " вылетел." << endl;

                if (flight.runwayId > 0) {
                    runways[flight.runwayId - 1].release();
                    flight.runwayId = -1;
                }
            }
        }
    }
    
    void updateTerminals() {
        for (auto& terminal : terminals) {
            terminal.process();
        }
    }
    
    void displayStatus() {
        int time = currentTime.load();
        if (time % 10 == 0) {
            cout << "\n=== Время: " << time << " ===" << endl;
            cout << "Очередь на посадку: " << landingQueue.size() << endl;
            cout << "Очередь на взлет: " << takeoffQueue.size() << endl;
            
            cout << "Полосы: ";
            for (const auto& runway : runways) {
                cout << "[" << runway.id << ": " << (runway.available ? "Свободна" : runway.currentFlight) << "] ";
            }
            cout << endl;
            
            cout << "Терминалы: ";
            for (const auto& terminal : terminals) {
                cout << "[" << terminal.id << ": " << (terminal.available ? "Свободен" : terminal.currentFlight);
                if (!terminal.available) {
                    cout << " (" << terminal.timeRemaining << ")";
                }
                cout << "] ";
            }
            cout << endl;
            
            cout << "Рейсы:" << endl;
            for (const auto& flight : flights) {
                if (flight.status != FlightStatus::DEPARTED) {
                    cout << "  " << flight.id << ": " << flight.getStatusString();
                    if (flight.terminalId != -1) cout << " (Терминал " << flight.terminalId << ")";
                    if (flight.runwayId != -1) cout << " (Полоса " << flight.runwayId << ")";
                    cout << endl;
                }
            }
            cout << "======================" << endl;
        }
    }
    
    void displayStatistics() {
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
    
    vector<Flight>::iterator findFlight(const string& flightId) {
        return find_if(flights.begin(), flights.end(),
                      [&flightId](const Flight& f) { return f.id == flightId; });
    }
};

int main() {
    srand(time(nullptr));
    setlocale(LC_ALL, "Russian");
    
    Airport airport;
    airport.runSimulation();
    
    return 0;
}
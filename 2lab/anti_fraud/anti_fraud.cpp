#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <omp.h>  // OpenMP для параллельной обработки

// Структура для представления транзакции
struct Transaction {
    int id;
    double amount;
    std::string currency;
    std::string merchant_type;
    std::string country;
    int user_id;
    long timestamp;
    std::string device_id;
    double user_avg_amount;
    int transactions_count_24h;
    bool is_fraud;
};

// Структура для результатов анализа
struct FraudScore {
    int transaction_id;
    double risk_score;
    std::vector<std::string> risk_factors;
    bool flagged;
    bool blocked;  // Флаг блокировки транзакции
};

class AntiFraudSystem {
private:
    std::vector<Transaction> transactions;
    std::vector<FraudScore> fraud_scores;
    std::unordered_map<int, std::vector<Transaction>> user_history;
    
    // Паттерны для обнаружения мошенничества
    std::vector<std::string> high_risk_merchants = {
        "online_gambling", "cryptocurrency", "high_end_jewelry", 
        "electronics", "digital_services"
    };
    
    std::vector<std::string> high_risk_countries = {
        "CN", "RU", "NG", "UA", "TR"
    };

public:
    // Загрузка транзакций из CSV файла
    void load_transactions_from_csv(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        
        std::cout << "Загрузка транзакций..." << std::endl;
        
        while (std::getline(file, line)) {
            Transaction t;
            std::stringstream ss(line);
            std::string token;
            
            std::getline(ss, token, ',');
            t.id = std::stoi(token);
            
            std::getline(ss, token, ',');
            t.amount = std::stod(token);
            
            std::getline(ss, t.currency, ',');
            std::getline(ss, t.merchant_type, ',');
            std::getline(ss, t.country, ',');
            
            std::getline(ss, token, ',');
            t.user_id = std::stoi(token);
            
            std::getline(ss, token, ',');
            t.timestamp = std::stol(token);
            
            std::getline(ss, t.device_id, ',');
            
            std::getline(ss, token, ',');
            t.user_avg_amount = std::stod(token);
            
            std::getline(ss, token, ',');
            t.transactions_count_24h = std::stoi(token);
            
            std::getline(ss, token, ',');
            t.is_fraud = (token == "1");
            
            transactions.push_back(t);
        }
        
        std::cout << "Загружено транзакций: " << transactions.size() << std::endl;
    }

    // DEPRECATED: Используйте analyze_fraud() вместо этих методов
    // Эти методы НЕ ПОТОКОБЕЗОПАСНЫ - не вызывайте их параллельно!
    // Они оставлены только для совместимости

    // Анализ аномалий по сумме
    void analyze_amount_anomalies() {
        std::cout << "Анализ аномалий по сумме..." << std::endl;
        
        for (int i = 0; i < transactions.size(); i++) {
            double amount_ratio = transactions[i].amount / transactions[i].user_avg_amount;
            
            if (amount_ratio > 5.0) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.3;
                score.risk_factors.push_back("Сумма превышает среднюю в " + 
                                            std::to_string(amount_ratio) + " раз");
                fraud_scores.push_back(score);
            }
            
            if (transactions[i].amount > 10000.0) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.2;
                score.risk_factors.push_back("Крупная сумма: " + 
                                            std::to_string(transactions[i].amount));
                fraud_scores.push_back(score);
            }
        }
    }

    // Анализ геолокации
    void analyze_geolocation_risks() {
        std::cout << "Анализ геолокационных рисков..." << std::endl;
        
        for (int i = 0; i < transactions.size(); i++) {
            // Проверка высокорисковых стран
            if (std::find(high_risk_countries.begin(), high_risk_countries.end(), 
                         transactions[i].country) != high_risk_countries.end()) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.25;
                score.risk_factors.push_back("Транзакция из подозрительной страны: " + 
                                            transactions[i].country);
                fraud_scores.push_back(score);
            }
            
            // Проверка высокорисковых мерчантов
            if (std::find(high_risk_merchants.begin(), high_risk_merchants.end(), 
                         transactions[i].merchant_type) != high_risk_merchants.end()) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.15;
                score.risk_factors.push_back("Транзакция у подозрительного мерчанта: " + 
                                            transactions[i].merchant_type);
                fraud_scores.push_back(score);
            }
        }
    }

    // Анализ поведенческих паттернов
    void analyze_behavioral_patterns() {
        std::cout << "Анализ поведенческих паттернов..." << std::endl;
        
        for (int i = 0; i < transactions.size(); i++) {
            // Проверка частоты транзакций
            if (transactions[i].transactions_count_24h > 10) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.2;
                score.risk_factors.push_back("Высокая частота транзакций: " + 
                                            std::to_string(transactions[i].transactions_count_24h) + 
                                            " за 24 часа");
                fraud_scores.push_back(score);
            }
            
            // Проверка ночной активности (2:00 - 6:00)
            long transaction_hour = (transactions[i].timestamp % 86400) / 3600;
            if (transaction_hour >= 2 && transaction_hour <= 6) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.1;
                score.risk_factors.push_back("Ночная транзакция в " + 
                                            std::to_string(transaction_hour) + " часов");
                fraud_scores.push_back(score);
            }
        }
    }

    // Машинное обучение - упрощенная модель оценки риска
    void machine_learning_risk_assessment() {
        std::cout << "ML оценка рисков..." << std::endl;
        
        for (int i = 0; i < transactions.size(); i++) {
            double ml_score = 0.0;
            std::vector<std::string> factors;
            
            // Нейросетевая-like оценка (упрощенная)
            ml_score += transactions[i].amount / 10000.0 * 0.3;
            ml_score += transactions[i].transactions_count_24h / 20.0 * 0.2;
            
            if (std::find(high_risk_countries.begin(), high_risk_countries.end(), 
                         transactions[i].country) != high_risk_countries.end()) {
                ml_score += 0.25;
                factors.push_back("ML:риск страна");
            }
            
            if (std::find(high_risk_merchants.begin(), high_risk_merchants.end(), 
                         transactions[i].merchant_type) != high_risk_merchants.end()) {
                ml_score += 0.15;
                factors.push_back("ML:риск мерчант");
            }
            
            // Нормализация скора
            ml_score = std::min(ml_score, 1.0);
            
            if (ml_score > 0.3) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score = ml_score;
                score.risk_factors.insert(score.risk_factors.end(), 
                                        factors.begin(), factors.end());
                score.flagged = (ml_score > 0.6);
                fraud_scores.push_back(score);
            }
        }
    }

    // Анализ устройств
    void analyze_device_risks() {
        std::cout << "Анализ рисков устройств..." << std::endl;
        
        std::unordered_map<std::string, std::vector<int>> device_transactions;
        
        // Сбор статистики по устройствам
        for (int i = 0; i < transactions.size(); i++) {
            device_transactions[transactions[i].device_id].push_back(transactions[i].user_id);
        }
        
        // Анализ подозрительных устройств
        for (int i = 0; i < transactions.size(); i++) {
            auto& users = device_transactions[transactions[i].device_id];
            if (users.size() > 3) {
                FraudScore score;
                score.transaction_id = transactions[i].id;
                score.risk_score += 0.3;
                score.risk_factors.push_back("Устройство используется " + 
                                            std::to_string(users.size()) + 
                                            " пользователями");
                fraud_scores.push_back(score);
            }
        }
    }

    // Агрегация результатов
    void aggregate_results() {
        std::cout << "Агрегация результатов..." << std::endl;
        
        std::unordered_map<int, FraudScore> aggregated_scores;
        
        for (int i = 0; i < fraud_scores.size(); i++) {
            int trans_id = fraud_scores[i].transaction_id;
            
            if (aggregated_scores.find(trans_id) == aggregated_scores.end()) {
                aggregated_scores[trans_id] = fraud_scores[i];
            } else {
                aggregated_scores[trans_id].risk_score += fraud_scores[i].risk_score;
                aggregated_scores[trans_id].risk_factors.insert(
                    aggregated_scores[trans_id].risk_factors.end(),
                    fraud_scores[i].risk_factors.begin(),
                    fraud_scores[i].risk_factors.end()
                );
                if (fraud_scores[i].flagged) {
                    aggregated_scores[trans_id].flagged = true;
                }
            }
        }
        
        // Обновление fraud_scores агрегированными результатами
        fraud_scores.clear();
        for (auto& pair : aggregated_scores) {
            fraud_scores.push_back(pair.second);
        }
    }

    // Применение всех правил к одной транзакции (вызывается параллельно)
    FraudScore applyAllRules(const Transaction& t) {
        FraudScore score;
        score.transaction_id = t.id;
        score.risk_score = 0.0;
        score.flagged = false;
        score.blocked = false;

        // Правило 1: Сумма превышает 50000 руб
        if (t.amount > 50000.0) {
            score.risk_score += 0.3;
            score.risk_factors.push_back("Сумма > 50000 руб: " + std::to_string(t.amount));
        }

        // Правило 2: Сумма превышает среднюю в 5 раз
        if (t.user_avg_amount > 0 && t.amount / t.user_avg_amount > 5.0) {
            score.risk_score += 0.25;
            score.risk_factors.push_back("Сумма превышает среднюю в " +
                std::to_string(t.amount / t.user_avg_amount) + " раз");
        }

        // Правило 3: Транзакция из высокорисковой страны
        if (std::find(high_risk_countries.begin(), high_risk_countries.end(),
                     t.country) != high_risk_countries.end()) {
            score.risk_score += 0.2;
            score.risk_factors.push_back("Транзакция из подозрительной страны: " + t.country);
        }

        // Правило 4: Высокорисковый мерчант
        if (std::find(high_risk_merchants.begin(), high_risk_merchants.end(),
                     t.merchant_type) != high_risk_merchants.end()) {
            score.risk_score += 0.15;
            score.risk_factors.push_back("Подозрительный мерчант: " + t.merchant_type);
        }

        // Правило 5: Высокая частота транзакций (>10 за 24ч)
        if (t.transactions_count_24h > 10) {
            score.risk_score += 0.2;
            score.risk_factors.push_back("Частота: " +
                std::to_string(t.transactions_count_24h) + " транзакций за 24ч");
        }

        // Правило 6: Ночная активность (2:00-6:00)
        long hour = (t.timestamp % 86400) / 3600;
        if (hour >= 2 && hour <= 6) {
            score.risk_score += 0.1;
            score.risk_factors.push_back("Ночная транзакция: " + std::to_string(hour) + ":00");
        }

        // Правило 7: Крупная сумма + высокорисковая страна (комбинация)
        if (t.amount > 30000.0 &&
            std::find(high_risk_countries.begin(), high_risk_countries.end(),
                     t.country) != high_risk_countries.end()) {
            score.risk_score += 0.3;
            score.risk_factors.push_back("Комбо: крупная сумма + рискованная страна");
        }

        // Определяем флаги и блокировку
        score.flagged = (score.risk_score > 0.5);
        score.blocked = (score.risk_score > 0.8);  // Автоматическая блокировка при высоком риске

        return score;
    }

    // Основной метод анализа - параллельная обработка пакетами (batch)
    // ПОТОКОБЕЗОПАСНЫЙ метод - можно вызывать из нескольких потоков
    void analyze_fraud() {
        if (transactions.empty()) {
            std::cout << "Нет транзакций для анализа" << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "=== ЗАПУСК АНТИФРОД СИСТЕМЫ ===" << std::endl;

        size_t n = transactions.size();
        int batchSize = 1000;  // Размер пакета для batch-обработки
        int numBatches = (n + batchSize - 1) / batchSize;

        std::cout << "Обработка " << n << " транзакций в " << numBatches << " пакетах" << std::endl;

        // Вектор для хранения результатов (предварительно выделяем память)
        std::vector<FraudScore> allScores(n);

        // Вектор для контроля коллизий - отслеживаем статус каждой транзакции
        // 0 = не обработана, 1 = обрабатывается, 2 = решение принято
        // Используется для избежания обработки одной транзакции несколькими потоками
        std::vector<int> transactionStatus(n, 0);

        // #pragma omp parallel for - параллельная обработка транзакций
        // schedule(dynamic, batchSize) - динамическое распределение пакетами
        // каждый поток берёт пакет из batchSize транзакций
        //
        // ПОТОКОБЕЗОПАСНОСТЬ:
        // 1. Каждый поток пишет в свой элемент allScores[i] - нет конфликтов
        // 2. Atomic операции с transactionStatus предотвращают двойную обработку
        // 3. Чтение из transactions - безопасно (только чтение)
        // 4. applyAllRules() - чистая функция без побочных эффектов
        #pragma omp parallel for schedule(dynamic, batchSize)
        for (size_t i = 0; i < n; ++i) {
            // Атомарная проверка и установка статуса для контроля коллизий
            // #pragma omp atomic capture - атомарно читаем и пишем
            // Гарантирует что транзакция обрабатывается только одним потоком
            // (защита от race condition при dynamic scheduling)
            int expected = 0;
            #pragma omp atomic capture
            {
                expected = transactionStatus[i];
                transactionStatus[i] = 1;  // Помечаем как "обрабатывается"
            }

            if (expected == 0) {  // Если ещё не обработана
                // Применяем все правила к транзакции
                // applyAllRules() - потокобезопасная функция (только чтение общих данных)
                allScores[i] = applyAllRules(transactions[i]);

                // Помечаем как "решение принято"
                #pragma omp atomic write
                transactionStatus[i] = 2;
            }
        }

        // Собираем только ненулевые результаты
        fraud_scores.clear();
        for (size_t i = 0; i < n; ++i) {
            if (allScores[i].risk_score > 0) {
                fraud_scores.push_back(allScores[i]);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Анализ завершён за " << duration.count() << " мс" << std::endl;
        std::cout << "Использовано потоков: " << omp_get_max_threads() << std::endl;
    }

    // Генерация тестовых данных
    void generate_test_data(int num_transactions) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> amount_dist(10.0, 50000.0);
        std::uniform_int_distribution<int> user_dist(1, 1000);
        std::uniform_int_distribution<int> country_dist(0, 4);
        std::uniform_int_distribution<int> merchant_dist(0, 4);
        
        std::vector<std::string> currencies = {"USD", "EUR", "GBP", "RUB", "CNY"};
        std::vector<std::string> merchants = {"retail", "online_gambling", "cryptocurrency", 
                                             "travel", "digital_services"};
        
        transactions.clear();
        
        for (int i = 0; i < num_transactions; i++) {
            Transaction t;
            t.id = i + 1;
            t.amount = amount_dist(gen);
            t.currency = currencies[i % currencies.size()];
            t.merchant_type = merchants[merchant_dist(gen)];
            t.country = high_risk_countries[country_dist(gen)];
            t.user_id = user_dist(gen);
            t.timestamp = 1700000000 + (i * 3600); // Распределение по времени
            t.device_id = "device_" + std::to_string(user_dist(gen) % 500);
            t.user_avg_amount = amount_dist(gen) / 2.0;
            t.transactions_count_24h = user_dist(gen) % 20;
            t.is_fraud = (i % 50 == 0); // 2% мошеннических транзакций
            
            transactions.push_back(t);
        }
        
        std::cout << "Сгенерировано " << transactions.size() << " тестовых транзакций" << std::endl;
    }

    // Вывод результатов
    void print_results() {
        std::cout << "\n=== РЕЗУЛЬТАТЫ АНТИФРОД АНАЛИЗА ===" << std::endl;

        int flagged_count = 0;
        int blocked_count = 0;
        double total_risk = 0.0;

        for (size_t i = 0; i < fraud_scores.size(); i++) {
            if (fraud_scores[i].flagged) flagged_count++;
            if (fraud_scores[i].blocked) blocked_count++;
            total_risk += fraud_scores[i].risk_score;
        }

        std::cout << "Всего проанализировано транзакций: " << transactions.size() << std::endl;
        std::cout << "Подозрительных (риск > 0.5): " << flagged_count << std::endl;
        std::cout << "Заблокировано (риск > 0.8): " << blocked_count << std::endl;
        if (!fraud_scores.empty()) {
            std::cout << "Средний риск: " << (total_risk / fraud_scores.size()) << std::endl;
        }

        // Топ-10 самых рискованных транзакций
        std::sort(fraud_scores.begin(), fraud_scores.end(),
                 [](const FraudScore& a, const FraudScore& b) {
                     return a.risk_score > b.risk_score;
                 });

        std::cout << "\n--- ТОП-10 САМЫХ РИСКОВАННЫХ ТРАНЗАКЦИЙ ---" << std::endl;
        for (int i = 0; i < std::min(10, (int)fraud_scores.size()); i++) {
            std::cout << "Транзакция " << fraud_scores[i].transaction_id
                      << " - Риск: " << fraud_scores[i].risk_score
                      << " - Правил: " << fraud_scores[i].risk_factors.size();
            if (fraud_scores[i].blocked) {
                std::cout << " [ЗАБЛОКИРОВАНА]";
            } else if (fraud_scores[i].flagged) {
                std::cout << " [ПОДОЗРИТЕЛЬНАЯ]";
            }
            std::cout << std::endl;

            if (!fraud_scores[i].risk_factors.empty()) {
                std::cout << "   Факторы: ";
                for (size_t j = 0; j < std::min((size_t)3, fraud_scores[i].risk_factors.size()); j++) {
                    std::cout << fraud_scores[i].risk_factors[j];
                    if (j < 2 && j < fraud_scores[i].risk_factors.size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }

    // Экспорт результатов в CSV
    void export_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        file << "transaction_id,risk_score,flagged,risk_factors_count\n";
        
        for (int i = 0; i < fraud_scores.size(); i++) {
            file << fraud_scores[i].transaction_id << ","
                 << fraud_scores[i].risk_score << ","
                 << (fraud_scores[i].flagged ? "1" : "0") << ","
                 << fraud_scores[i].risk_factors.size() << "\n";
        }
        
        file.close();
        std::cout << "Результаты экспортированы в " << filename << std::endl;
    }
};

int main() {
    AntiFraudSystem afs;
    
    // Генерация тестовых данных
    afs.generate_test_data(10000);
    
    // Запуск анализа
    afs.analyze_fraud();
    
    // Вывод результатов
    afs.print_results();
    
    // Экспорт результатов
    afs.export_to_csv("fraud_results.csv");
    
    return 0;
}
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <omp.h>

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
};

struct FraudResult {
    int transaction_id;
    double risk_score;
    std::vector<std::string> risk_factors;
    bool blocked;
};

class AntiFraudSystem {
private:
    // Быстрый поиск через unordered_set
    std::unordered_set<std::string> high_risk_countries = {"CN", "RU", "NG", "UA", "TR"};
    std::unordered_set<std::string> high_risk_merchants = {
        "online_gambling", "cryptocurrency", "high_end_jewelry", "electronics"
    };

    // Применение ВСЕХ правил к одной транзакции (атомарно, без коллизий)
    FraudResult applyAllRules(const Transaction& t,
                              const std::unordered_map<std::string, int>& deviceUserCount) const {
        FraudResult result;
        result.transaction_id = t.id;
        result.risk_score = 0.0;
        result.blocked = false;

        // Правило 1: Сумма > 50000
        if (t.amount > 50000.0) {
            result.risk_score += 0.4;
            result.risk_factors.push_back("Сумма > 50000");
        }

        // Правило 2: Сумма превышает среднюю в 5+ раз
        if (t.user_avg_amount > 0 && t.amount / t.user_avg_amount > 5.0) {
            result.risk_score += 0.3;
            result.risk_factors.push_back("Сумма >> средней");
        }

        // Правило 3: Высокорисковая страна
        if (high_risk_countries.count(t.country)) {
            result.risk_score += 0.25;
            result.risk_factors.push_back("Страна: " + t.country);
        }

        // Правило 4: Высокорисковый мерчант
        if (high_risk_merchants.count(t.merchant_type)) {
            result.risk_score += 0.2;
            result.risk_factors.push_back("Мерчант: " + t.merchant_type);
        }

        // Правило 5: Высокая частота транзакций
        if (t.transactions_count_24h > 10) {
            result.risk_score += 0.2;
            result.risk_factors.push_back("Частота: " + std::to_string(t.transactions_count_24h) + "/24ч");
        }

        // Правило 6: Ночная активность (2:00-6:00)
        int hour = (t.timestamp % 86400) / 3600;
        if (hour >= 2 && hour <= 6) {
            result.risk_score += 0.15;
            result.risk_factors.push_back("Ночь: " + std::to_string(hour) + "ч");
        }

        // Правило 7: Устройство используется многими пользователями
        auto it = deviceUserCount.find(t.device_id);
        if (it != deviceUserCount.end() && it->second > 3) {
            result.risk_score += 0.3;
            result.risk_factors.push_back("Устройство: " + std::to_string(it->second) + " юзеров");
        }

        // Правило 8: Крупная сумма + высокорисковая страна (комбо)
        if (t.amount > 10000.0 && high_risk_countries.count(t.country)) {
            result.risk_score += 0.35;
            result.risk_factors.push_back("Комбо: сумма+страна");
        }

        // Нормализация и решение о блокировке
        result.risk_score = std::min(result.risk_score, 1.0);
        result.blocked = (result.risk_score >= 0.7);

        return result;
    }

public:
    // Пакетный анализ транзакций
    std::vector<FraudResult> analyzeBatch(const std::vector<Transaction>& transactions) {
        if (transactions.empty()) return {};

        std::cout << "Анализ пакета: " << transactions.size() << " транзакций" << std::endl;

        // Предварительный расчёт статистики по устройствам (последовательно)
        std::unordered_map<std::string, std::unordered_set<int>> deviceUsers;
        for (const auto& t : transactions) {
            deviceUsers[t.device_id].insert(t.user_id);
        }

        std::unordered_map<std::string, int> deviceUserCount;
        for (const auto& entry : deviceUsers) {
            deviceUserCount[entry.first] = static_cast<int>(entry.second.size());
        }

        // Локальные результаты для каждого потока
        const int maxThreads = omp_get_max_threads();
        std::vector<std::vector<FraudResult>> threadResults(maxThreads);

        // Параллельный анализ - каждая транзакция полностью в одном потоке
        // (контроль коллизий: один поток = одно решение)
        #pragma omp parallel for schedule(dynamic) default(none) \
            shared(transactions, deviceUserCount, threadResults)
        for (size_t i = 0; i < transactions.size(); ++i) {
            int tid = omp_get_thread_num();

            // Все правила применяются атомарно к одной транзакции
            FraudResult result = applyAllRules(transactions[i], deviceUserCount);

            // Только значимые результаты
            if (result.risk_score > 0.1) {
                threadResults[tid].push_back(result);
            }
        }

        // Слияние результатов (последовательно)
        std::vector<FraudResult> allResults;
        for (const auto& results : threadResults) {
            allResults.insert(allResults.end(), results.begin(), results.end());
        }

        return allResults;
    }

    void printResults(const std::vector<FraudResult>& results) const {
        int blocked = 0, suspicious = 0;
        double totalRisk = 0.0;

        for (const auto& r : results) {
            if (r.blocked) blocked++;
            else if (r.risk_score > 0.3) suspicious++;
            totalRisk += r.risk_score;
        }

        std::cout << "\n=== РЕЗУЛЬТАТЫ ===" << std::endl;
        std::cout << "Проанализировано: " << results.size() << std::endl;
        std::cout << "Заблокировано: " << blocked << std::endl;
        std::cout << "Подозрительных: " << suspicious << std::endl;
        if (!results.empty()) {
            std::cout << "Средний риск: " << (totalRisk / results.size()) << std::endl;
        }

        // Топ-10
        std::vector<const FraudResult*> sorted;
        for (const auto& r : results) sorted.push_back(&r);
        std::sort(sorted.begin(), sorted.end(),
            [](const FraudResult* a, const FraudResult* b) {
                return a->risk_score > b->risk_score;
            });

        std::cout << "\n--- ТОП-10 ---" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sorted.size()); ++i) {
            std::cout << "ID:" << sorted[i]->transaction_id
                      << " Риск:" << sorted[i]->risk_score
                      << (sorted[i]->blocked ? " [BLOCKED]" : "") << std::endl;
        }
    }

    void exportCSV(const std::vector<FraudResult>& results, const std::string& filename) const {
        std::ofstream file(filename);
        file << "id,risk,blocked,factors\n";
        for (const auto& r : results) {
            file << r.transaction_id << "," << r.risk_score << ","
                 << (r.blocked ? 1 : 0) << "," << r.risk_factors.size() << "\n";
        }
        std::cout << "Экспорт: " << filename << std::endl;
    }

    int getNumThreads() const { return omp_get_max_threads(); }
};

// Генерация тестовых данных
std::vector<Transaction> generateTestData(int count) {
    std::vector<Transaction> transactions(count);

    std::vector<std::string> countries = {"US", "GB", "DE", "CN", "RU", "NG", "FR", "JP"};
    std::vector<std::string> merchants = {"retail", "online_gambling", "cryptocurrency", "travel", "food"};

    unsigned int seed = 42;

    #pragma omp parallel default(none) shared(transactions, countries, merchants, count, seed)
    {
        // Каждый поток имеет свой генератор
        unsigned int threadSeed = seed + omp_get_thread_num();
        std::mt19937 gen(threadSeed);
        std::uniform_real_distribution<double> amountDist(10.0, 100000.0);
        std::uniform_int_distribution<int> countryDist(0, countries.size() - 1);
        std::uniform_int_distribution<int> merchantDist(0, merchants.size() - 1);
        std::uniform_int_distribution<int> userDist(1, 1000);
        std::uniform_int_distribution<int> freqDist(1, 25);

        #pragma omp for schedule(static)
        for (int i = 0; i < count; ++i) {
            Transaction& t = transactions[i];
            t.id = i + 1;
            t.amount = amountDist(gen);
            t.country = countries[countryDist(gen)];
            t.merchant_type = merchants[merchantDist(gen)];
            t.user_id = userDist(gen);
            t.device_id = "dev_" + std::to_string(userDist(gen) % 300);
            t.timestamp = 1700000000 + i * 60;
            t.user_avg_amount = amountDist(gen) / 3.0;
            t.transactions_count_24h = freqDist(gen);
        }
    }

    return transactions;
}

int main() {
    AntiFraudSystem afs;

    std::cout << "=== ANTI-FRAUD (OpenMP, " << afs.getNumThreads() << " потоков) ===" << std::endl;

    // Генерация данных
    auto transactions = generateTestData(100000);
    std::cout << "Сгенерировано: " << transactions.size() << " транзакций" << std::endl;

    // Анализ
    double start = omp_get_wtime();
    auto results = afs.analyzeBatch(transactions);
    double end = omp_get_wtime();

    // Результаты
    afs.printResults(results);
    afs.exportCSV(results, "fraud_results.csv");

    std::cout << "\nВремя: " << (end - start) << " сек" << std::endl;

    return 0;
}

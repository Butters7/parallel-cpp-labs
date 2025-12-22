# Исправления ошибок в лабораторной работе 2 (OpenMP)

## 1. k-NN/knn.cpp

### 1.1 Вложенный параллелизм в euclideanDistance

**Проблема:**
```cpp
double KNN::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double distance = 0.0;
    #pragma omp parallel for reduction(+:distance)  // ПАРАЛЛЕЛЬНО
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}

int KNN::predict(const std::vector<double>& x) {
    #pragma omp parallel for schedule(static)  // ПАРАЛЛЕЛЬНО
    for (size_t i = 0; i < n; ++i) {
        double dist = euclideanDistance(x, X_train[i]);  // Вызов параллельной функции!
    }
}
```

**Проблемы:**
- euclideanDistance параллелизуется через `#pragma omp parallel for`
- Она вызывается ИЗ параллельного цикла в predict()
- Вложенный параллелизм создает слишком много потоков (threads * threads)
- Деградация производительности вместо ускорения

**Исправление:**
```cpp
double KNN::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    // Проверка размеров векторов
    if (a.size() != b.size()) {
        std::cerr << "Ошибка: размеры векторов не совпадают" << std::endl;
        return std::numeric_limits<double>::max();
    }

    double distance = 0.0;

    // УБРАЛИ #pragma omp parallel for - функция вызывается изнутри
    // параллельного цикла, вложенный параллелизм приводит к деградации
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}
```

**Результат:**
- Нет вложенного параллелизма
- Функция безопасна для вызова из параллельного кода
- Добавлена проверка размеров векторов

### 1.2 Непроверенное условие k <= n в partial_sort

**Проблема:**
```cpp
std::partial_sort(
    distances.begin(),
    distances.begin() + k,  // Если k > n, выход за границы!
    distances.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; }
);
```

Если k > количества точек, происходит UB (undefined behavior).

**Исправление:**
```cpp
// Проверка: должно быть достаточно точек
if (n == 0) {
    std::cerr << "Ошибка: нет обучающих данных" << std::endl;
    return -1;
}

// Проверка: k не должно превышать количество точек
int k_actual = k;
if (k > static_cast<int>(n)) {
    std::cerr << "Предупреждение: k=" << k << " больше n=" << n
              << ", используем k=" << n << std::endl;
    k_actual = n;
}

std::partial_sort(
    distances.begin(),
    distances.begin() + k_actual,
    distances.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; }
);
```

**Результат:**
- Безопасная работа при любых значениях k
- Информативное сообщение пользователю

### 1.3 Отсутствие проверки размеров векторов

**Проблема:**
Если векторы `a` и `b` имеют разные размеры, вычисление расстояния некорректно.

**Исправление:**
Добавлена проверка в начале euclideanDistance (см. выше).

---

## 2. spam/spam_filter.cpp + spam_filter.h

### 2.1 Гонка данных при обращении к vocabulary

**Проблема:**
```cpp
void train(const std::string& spamFile, const std::string& hamFile) {
    while (std::getline(spamStream, line)) {
        for (const auto& token : tokens) {
            vocabulary[token].spamCount++;  // Модификация
        }
        totalSpam++;  // Не атомарно!
    }
}

bool classify(const std::string& email) {
    #pragma omp parallel for  // ПАРАЛЛЕЛЬНО
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = vocabulary.find(token);  // Чтение
        // ...
        double ratio = spamProb / (totalSpam + alpha * vocabSize);  // Чтение totalSpam
    }
}
```

**Проблемы:**
- `totalSpam` и `totalHam` модифицируются в train() без синхронизации
- Они читаются в classify() из параллельного цикла
- Если train() вызывается параллельно с classify(), гонка данных
- Даже при последовательном вызове нет гарантии видимости изменений

**Исправление:**

**Заголовок (spam_filter.h):**
```cpp
class SpamFilter {
private:
    std::unordered_map<std::string, WordStats> vocabulary;
    std::atomic<int> totalSpam{0};  // Атомарный счетчик
    std::atomic<int> totalHam{0};   // Атомарный счетчик
    std::atomic<bool> is_trained{false};  // Флаг обученности

public:
    // train() должен вызываться ОДИН РАЗ до любых вызовов classify()
    void train(...);

    // classify() потокобезопасен после завершения train()
    bool classify(...);
};
```

**Реализация (spam_filter.cpp):**
```cpp
void SpamFilter::train(const std::string& spamFile, const std::string& hamFile) {
    vocabulary.clear();
    totalSpam.store(0);
    totalHam.store(0);

    int local_spam_count = 0;
    int local_ham_count = 0;

    // Обучение...
    while (std::getline(spamStream, line)) {
        // ...
        local_spam_count++;
    }

    // Атомарное обновление
    totalSpam.store(local_spam_count);
    totalHam.store(local_ham_count);
    is_trained.store(true);
}

bool SpamFilter::classify(const std::string& email) {
    if (!is_trained.load()) {
        std::cerr << "Ошибка: модель не обучена" << std::endl;
        return false;
    }

    // Загружаем атомарные значения один раз для консистентности
    int spam_count = totalSpam.load();
    int ham_count = totalHam.load();

    #pragma omp parallel for reduction(+:spamSum, hamSum)
    for (size_t i = 0; i < tokens.size(); ++i) {
        // ...
        spamSum += std::log(spamProb / (spam_count + alpha * vocabSize));
    }
}
```

**Результат:**
- `std::atomic` обеспечивает потокобезопасный доступ
- Флаг `is_trained` предотвращает использование необученной модели
- vocabulary используется только для чтения после train() - безопасно
- Добавлен `#include <iostream>` для std::cerr

#### 2.2 Отсутствие синхронизации при чтении/записи totalSpam и totalHam

Решено через `std::atomic`

---

## 3. sast/sast.cpp

### 3.1 Непотокобезопасное чтение файлов в параллельном цикле

**Проблема:**
```cpp
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < files.size(); ++i) {
    std::vector<std::string> lines = readFile(file);  // Потокобезопасно ли?
}

std::vector<std::string> readFile(const std::string& filename) {
    std::ifstream file(filename);  // Каждый поток создает свой ifstream
    // ...
}
```

**Анализ:**
- На самом деле это БЕЗОПАСНО, т.к. каждый поток создает СВОЙ локальный `ifstream`
- НО нет проверки ошибок открытия файла

**Исправление:**
```cpp
std::vector<std::string> readFile(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Предупреждение: не удалось открыть файл " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}
```

### 3.2 Потенциальные deadlocks при рекурсивном обходе директорий

**Проблема:**
```cpp
std::vector<std::string> collectSourceFiles(const std::string& directory) {
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && isSourceFile(entry.path().string())) {
            files.push_back(entry.path().string());
        }
    }
}
```

**Проблемы:**
- Отсутствует обработка исключений при нет прав доступа
- Может выбросить `fs::filesystem_error`

**Исправление:**
```cpp
std::vector<std::string> collectSourceFiles(const std::string& directory) {
    std::vector<std::string> files;

    try {
        // skip_permission_denied - пропускаем недоступные директории
        for (const auto& entry : fs::recursive_directory_iterator(
            directory,
            fs::directory_options::skip_permission_denied)) {

            try {
                if (entry.is_regular_file() && isSourceFile(entry.path().string())) {
                    files.push_back(entry.path().string());
                }
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Предупреждение: " << e.what() << std::endl;
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Ошибка при обходе директории: " << e.what() << std::endl;
    }

    return files;
}
```

### 3.3 Улучшения в analyzeDirectory

**Добавлено:**
```cpp
void analyzeDirectory(const std::string& directory) {
    std::vector<std::string> files = collectSourceFiles(directory);

    if (files.empty()) {
        std::cout << "Нет файлов для анализа" << std::endl;
        return;
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < files.size(); ++i) {
        std::vector<std::string> lines = readFile(file);

        if (lines.empty()) {
            continue;  // Пропускаем пустые/недоступные файлы
        }
        // ...
    }
}
```

**Результат:**
- Корректная обработка ошибок доступа к файлам
- Улучшенные комментарии о потокобезопасности
- Проверка на пустые результаты

---

## 4. anti_fraud/anti_fraud.cpp

### 4.1 Старые методы не потокобезопасны

**Проблема:**
Методы `analyze_amount_anomalies()`, `analyze_geolocation_risks()` и др. пишут в общий `fraud_scores` без синхронизации:

```cpp
void analyze_amount_anomalies() {
    for (int i = 0; i < transactions.size(); i++) {
        if (amount_ratio > 5.0) {
            FraudScore score;
            // ...
            fraud_scores.push_back(score);  // Гонка данных если вызвать параллельно!
        }
    }
}
```

**Проблемы:**
- Если эти методы вызвать параллельно, `fraud_scores.push_back()` вызовет гонку данных
- НО эти методы НЕ ИСПОЛЬЗУЮТСЯ в main() - там используется только `analyze_fraud()`

**Исправление:**
Добавлены комментарии:

```cpp
// DEPRECATED: Используйте analyze_fraud() вместо этих методов
// Эти методы НЕ ПОТОКОБЕЗОПАСНЫ - не вызывайте их параллельно!
// Они оставлены только для совместимости

void analyze_amount_anomalies() { ... }
void analyze_geolocation_risks() { ... }
void analyze_behavioral_patterns() { ... }
void machine_learning_risk_assessment() { ... }
void analyze_device_risks() { ... }
void aggregate_results() { ... }
```

### 4.2 Основной метод analyze_fraud() уже безопасен

**Текущая реализация (правильная):**
```cpp
void analyze_fraud() {
    size_t n = transactions.size();
    std::vector<FraudScore> allScores(n);  // Предвыделенный вектор
    std::vector<int> transactionStatus(n, 0);

    #pragma omp parallel for schedule(dynamic, batchSize)
    for (size_t i = 0; i < n; ++i) {
        // Atomic операции для контроля коллизий
        int expected = 0;
        #pragma omp atomic capture
        {
            expected = transactionStatus[i];
            transactionStatus[i] = 1;
        }

        if (expected == 0) {
            allScores[i] = applyAllRules(transactions[i]);

            #pragma omp atomic write
            transactionStatus[i] = 2;
        }
    }

    // Последовательный сбор результатов
    fraud_scores.clear();
    for (size_t i = 0; i < n; ++i) {
        if (allScores[i].risk_score > 0) {
            fraud_scores.push_back(allScores[i]);
        }
    }
}
```

**Почему это безопасно:**
1. Каждый поток пишет в СВОЙ элемент `allScores[i]` - нет конфликтов
2. `#pragma omp atomic capture` предотвращает двойную обработку одной транзакции
3. Чтение из `transactions` - безопасно (только чтение)
4. `applyAllRules()` - чистая функция без побочных эффектов
5. Финальный сбор в `fraud_scores` - последовательный

**Добавлены улучшения:**
```cpp
void analyze_fraud() {
    if (transactions.empty()) {
        std::cout << "Нет транзакций для анализа" << std::endl;
        return;
    }

    // ... улучшенные комментарии:

    // ПОТОКОБЕЗОПАСНОСТЬ:
    // 1. Каждый поток пишет в свой элемент allScores[i] - нет конфликтов
    // 2. Atomic операции с transactionStatus предотвращают двойную обработку
    // 3. Чтение из transactions - безопасно (только чтение)
    // 4. applyAllRules() - чистая функция без побочных эффектов
}
```

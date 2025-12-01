# Лабораторная работа 2: OpenMP

## Задача 1: k-NN классификатор (k-NN)

### Что конкретно сделал

#### Параллельное вычисление евклидова расстояния
```cpp
double KNN::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double distance = 0.0;

    #pragma omp parallel for reduction(+:distance)
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(distance);
}
```
Тут используем `reduction(+:distance)` Каждый поток считает свою частичную сумму, а в конце OpenMP автоматически складывает всё вместе. Без reduction была бы гонка данных — потоки бы одновременно писали в одну переменную.

#### Параллельный расчёт расстояний до всех точек
```cpp
int KNN::predict(const std::vector<double>& x) {
    size_t n = X_train.size();
    std::vector<std::pair<double, int>> distances(n);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double dist = euclideanDistance(x, X_train[i]);
        distances[i] = std::make_pair(dist, y_train[i]);
    }
    // ... дальше сортировка и голосование
}
```
`schedule(static)` — статическое распределение итераций. Значит каждый поток заранее знает какие итерации ему достались. Это эффективно когда все итерации примерно одинаковые по времени.

Тут reduction не нужен — каждый поток пишет в свой элемент `distances[i]`, конфликтов нет.

Сортировку (`partial_sort`) не параллелим — там сложно и выигрыш сомнительный.

## Задача 2: SAST анализатор (sast)

### Что конкретно сделал

#### Параллельный анализ файлов
```cpp
void analyzeDirectory(const std::string& directory) {
    std::vector<std::string> files = collectSourceFiles(directory);

    std::vector<std::vector<SecurityIssue>> threadResults(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < files.size(); ++i) {
        int threadId = omp_get_thread_num();
        const std::string& file = files[i];

        std::vector<std::string> lines = readFile(file);

        for (size_t lineNum = 0; lineNum < lines.size(); ++lineNum) {
            const std::string& line = lines[lineNum];

            for (const auto& func : dangerous_functions) {
                if (line.find(func) != std::string::npos) {
                    SecurityIssue issue;
                    issue.filename = file;
                    issue.line_number = lineNum + 1;
                    threadResults[threadId].push_back(issue);
                }
            }
        }
    }

    for (const auto& results : threadResults) {
        issues.insert(issues.end(), results.begin(), results.end());
    }
}
```

Тут несколько важных моментов:

1. **`schedule(dynamic)`** — динамическое распределение. Файлы разного размера, так что статическое было бы неэффективно. При динамическом потоки берут новую работу когда освободились.

2. **`omp_get_thread_num()`** — получаем номер текущего потока. Нужно чтобы писать в свой вектор результатов.

3. **Отдельный вектор для каждого потока** — `threadResults[threadId]`. Так избегаем синхронизации при добавлении результатов. В конце просто склеиваем.

## Задача 3: Антифрод система (anti_fraud)

### Что конкретно сделал

#### Параллельная обработка транзакций с контролем коллизий
```cpp
void analyze_fraud() {
    size_t n = transactions.size();
    int batchSize = 1000;

    std::vector<FraudScore> allScores(n);
    std::vector<int> transactionStatus(n, 0);  // 0=не обработана, 1=в процессе, 2=готово

    #pragma omp parallel for schedule(dynamic, batchSize)
    for (size_t i = 0; i < n; ++i) {
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
}
```

1. **`schedule(dynamic, batchSize)`** — динамическое распределение пакетами по 1000. Каждый поток берёт пачку транзакций.

2. **`#pragma omp atomic capture`** — атомарная операция "прочитать и записать". Гарантирует что транзакцию обработает только один поток. Это типа compare-and-swap.

3. **`#pragma omp atomic write`** — атомарная запись. Помечаем транзакцию как обработанную.

По сути это защита от ситуации когда два потока схватят одну транзакцию.

#### Применение правил к одной транзакции
```cpp
FraudScore applyAllRules(const Transaction& t) {
    FraudScore score;
    score.risk_score = 0.0;

    if (t.amount > 50000.0) {
        score.risk_score += 0.3;
        score.risk_factors.push_back("Сумма > 50000 руб");
    }

    if (t.user_avg_amount > 0 && t.amount / t.user_avg_amount > 5.0) {
        score.risk_score += 0.25;
        // ...
    }

    score.flagged = (score.risk_score > 0.5);
    score.blocked = (score.risk_score > 0.8);

    return score;
}
```
Это чистая функция — не меняет глобальное состояние, можно безопасно вызывать из разных потоков.

## Задача 4: Спам-фильтр (spam)

### Что конкретно сделал

#### Параллельная классификация
```cpp
bool SpamFilter::classify(const std::string& email) {
    auto tokens = tokenize(email);
    double spamLogProb = std::log(static_cast<double>(totalSpam) / (totalSpam + totalHam));
    double hamLogProb = std::log(static_cast<double>(totalHam) / (totalSpam + totalHam));
    int vocabSize = vocabulary.size();

    double spamSum = 0.0;
    double hamSum = 0.0;

    #pragma omp parallel for reduction(+:spamSum, hamSum) schedule(dynamic)
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        auto it = vocabulary.find(token);
        double spamProb = alpha;
        double hamProb = alpha;

        if (it != vocabulary.end()) {
            spamProb += it->second.spamCount;
            hamProb += it->second.hamCount;
        }

        spamSum += std::log(spamProb / (totalSpam + alpha * vocabSize));
        hamSum += std::log(hamProb / (totalHam + alpha * vocabSize));
    }

    spamLogProb += spamSum;
    hamLogProb += hamSum;

    return spamLogProb > hamLogProb;
}
```

Тут:
1. **`reduction(+:spamSum, hamSum)`** — две переменные в reduction Каждый поток накапливает свои суммы, потом всё складывается.

2. **`schedule(dynamic)`** — динамическое распределение, потому что поиск в `vocabulary` (это map) может занимать разное время.

3. Читаем из `vocabulary` — это безопасно, несколько потоков могут читать одновременно.

## Задача 5: Брутфорс SHA1 на GPU (sha1-bf)

### Суть задачи
Перебор паролей для взлома SHA1 хеша. Используем OpenMP offload для запуска на GPU.

### Файлы
- `sha1-bf/sha1_bruteforce.cpp` — брутфорс
- `sha1-bf/sha1_bruteforce.h` — заголовочник
- `sha1-bf/main.cpp` — запуск

### Что конкретно сделал

#### Атака по словарю на GPU
```cpp
void dictionary_attack(uint8_t* target_hash, const char* dict_filename, size_t start, size_t count) {

    int found = 0;
    size_t found_index = 0;

    #pragma omp target teams distribute parallel for \
                schedule(static) \
                map(to: target_hash[0:20], dict_words[0:dict_size*MAX_PASSWORD_LEN], word_lengths[0:dict_size], dict_size) \
                map(tofrom: found, found_index)
    for (size_t i = 0; i < dict_size; i++) {
        if (found) continue; 

        const char* candidate = dict_words + i * MAX_PASSWORD_LEN;
        size_t length = word_lengths[i];

        uint8_t hash[20];
        sha1(candidate, length, hash);

        if (hash_matches(hash, target_hash)) {
            int old_found;
            #pragma omp atomic capture
            { old_found = found; found = 1; }
            if (old_found == 0) {
                found_index = i;
            }
        }
    }
}
```

1. **`#pragma omp target`** — запускаем код на GPU в Collab

2. **`teams distribute parallel for`** — создаём команды потоков на GPU и распределяем итерации

3. **`map(to: ...)`** — копируем данные с хоста на девайс

4. **`map(tofrom: ...)`** — копируем туда и обратно

5. **`schedule(static)`** — для GPU статическое планирование обязательно

6. **Early termination** — `if (found) continue;` — если нашли пароль, остальные итерации пропускаем

7. **`#pragma omp atomic capture`** — атомарная операция на GPU для безопасного обновления `found`

#### Прямой перебор
```cpp
void brute_force_attack(uint8_t* target_hash, int min_length, int max_length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    int charset_len = sizeof(charset) - 1;

    for (int len = min_length; len <= max_length && !found; len++) {
        long long max_i = 1;
        for (int j = 0; j < len; j++) max_i *= charset_len;

        #pragma omp target teams distribute parallel for \
                    schedule(static) \
                    map(to: target_hash[0:20], charset[0:charset_len+1], len, max_i, charset_len) \
                    map(tofrom: found, found_index)
        for (long long i = 0; i < max_i; i++) {
            if (found) continue;

            char candidate[32];
            long long temp = i;
            for (int pos = 0; pos < len; pos++) {
                candidate[pos] = charset[temp % charset_len];
                temp /= charset_len;
            }
            candidate[len] = '\0';

            uint8_t hash[20];
            sha1(candidate, len, hash);

            if (hash_matches(hash, target_hash)) {
                int old_found;
                #pragma omp atomic capture
                { old_found = found; found = 1; }
                if (old_found == 0) {
                    found_index = i;
                }
            }
        }
    }
}
```

Аналогичная логика, но генерируем пароли по индексу. Для длины 4 это 26^4 = 456976 комбинаций — на GPU это быстро.

#### Проверка GPU
```cpp
int check_gpu() {
    int on_gpu = 0;
    #pragma omp target map(from:on_gpu)
    {
        if (omp_is_initial_device() == 0) on_gpu = 1;
    }
    return on_gpu;
}
```
Проверяем реально ли код запустился на GPU. `omp_is_initial_device()` возвращает 0 если мы на GPU.

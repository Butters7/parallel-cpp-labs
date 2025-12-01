# Лабораторная работа 4: CUDA

## Задача 1: Анализ переполнения буфера (buffer_overflow)

### Что конкретно сделал

#### Структура результата анализа
```cpp
typedef struct {
    int is_overflow;      // 1 = переполнение, 0 = норма
    int string_length;    // фактическая длина строки
    int buffer_limit;     // лимит буфера
    int overflow_amount;  // на сколько превышен лимит
} AnalysisResult;
```
Простая структура для хранения результата

#### CUDA Kernel для анализа
```cpp
__global__ void analyzeBufferOverflow(const char* strings, AnalysisResult* results,
                                       const int* limits, int n, int string_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const char* str = &strings[idx * string_size];
    int limit = limits[idx];

    // Вычисляем длину строки
    int len = 0;
    while (str[len] != '\0' && len < string_size) {
        len++;
    }

    // Анализ на переполнение
    results[idx].string_length = len;
    results[idx].buffer_limit = limit;

    if (len > limit) {
        results[idx].is_overflow = 1;
        results[idx].overflow_amount = len - limit;
    } else {
        results[idx].is_overflow = 0;
        results[idx].overflow_amount = 0;
    }
}
```

- **`__global__`** — это kernel, функция которая запускается на GPU
- **`blockIdx.x * blockDim.x + threadIdx.x`** — вычисляем глобальный индекс потока. В CUDA потоки группируются в блоки, поэтому формула такая
- **`if (idx >= n) return;`** — проверка границ, потоков может быть больше чем данных
- Каждый поток обрабатывает одну строку — массовый параллелизм

#### Выделение памяти и копирование
```cpp
// Выделение памяти на GPU
char* d_strings;
int* d_limits;
AnalysisResult* d_results;

cudaMalloc(&d_strings, strings_size);
cudaMalloc(&d_limits, limits_size);
cudaMalloc(&d_results, results_size);

// Копирование Host -> Device
cudaMemcpy(d_strings, h_strings, strings_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_limits, h_limits, limits_size, cudaMemcpyHostToDevice);
```
**`cudaMalloc`** — выделяем память на GPU (device).
**`cudaMemcpy`** — копируем данные. `cudaMemcpyHostToDevice` = с CPU на GPU.

Префиксы: `h_` — host (CPU), `d_` — device (GPU). Так удобно различать.

#### Запуск kernel
```cpp
int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

cudaEventRecord(start);
analyzeBufferOverflow<<<numBlocks, BLOCK_SIZE>>>(d_strings, d_results, d_limits, n, string_size);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
```
**`<<<numBlocks, BLOCK_SIZE>>>`** — это специальный синтаксис CUDA для запуска kernel. Первое число — сколько блоков, второе — сколько потоков в блоке.

Формула `(n + BLOCK_SIZE - 1) / BLOCK_SIZE` — это округление вверх, чтобы хватило потоков для всех элементов.

**`cudaEventRecord/cudaEventSynchronize`** — замер времени выполнения на GPU.

#### Сравнение CPU vs GPU
```cpp
float speedup = cpu_ms / total_gpu_ms;
float kernel_speedup = cpu_ms / kernel_ms;

printf("Speedup (total):       %.2fx\n", speedup);
printf("Speedup (kernel only): %.2fx\n", kernel_speedup);
```

## Задача 2: Обфускация строк (obfuscation)

### Что конкретно сделал

#### Device-функция для бинарного поиска
```cpp
__device__ int binarySearchStringIndex(const int* offsets, int num_strings, int char_idx) {
    int left = 0;
    int right = num_strings - 1;
    int result = 0;

    while (left <= right) {
        int mid = (left + right) / 2;
        if (offsets[mid] <= char_idx) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}
```
**`__device__`** — функция которая вызывается с GPU и выполняется на GPU. В отличие от `__global__` её нельзя вызвать с CPU.

Cтроки разной длины лежат подряд в одном массиве. Чтобы понять какой строке принадлежит символ, используем бинарный поиск по массиву смещений.

#### Kernel для обфускации
```cpp
__global__ void obfuscateKernel(char* data, const int* offsets, const int* keys,
                                 int num_strings, int total_chars) {
    int char_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (char_idx >= total_chars) return;

    // Бинарный поиск: определяем какой строке принадлежит символ
    int string_idx = binarySearchStringIndex(offsets, num_strings, char_idx);

    // Позиция символа внутри строки
    int pos_in_string = char_idx - offsets[string_idx];

    // Ключ для этой строки
    int key = keys[string_idx];

    // XOR с ключом, модифицированным позицией
    unsigned char modified_key = (unsigned char)((key + pos_in_string) % 256);
    data[char_idx] ^= modified_key;
}
```
Каждый поток обрабатывает один символ. Находим какой строке принадлежит символ, берём ключ этой строки, XOR'им.

Фишка XOR: если применить два раза — получим исходное значение. Поэтому функция работает и для шифрования, и для дешифрования.

#### Проверка обратимости XOR
```cpp
// Применяем обфускацию ещё раз
cudaMemcpy(d_data, h_data, total_chars, cudaMemcpyHostToDevice);
obfuscateKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_offsets, d_keys, num_strings, total_chars);
cudaMemcpy(h_data, d_data, total_chars, cudaMemcpyDeviceToHost);

// Сравниваем с оригиналом
int mismatches = 0;
for (int i = 0; i < total_chars; i++) {
    if (h_data[i] != h_original[i]) {
        mismatches++;
    }
}

if (mismatches == 0) {
    printf("XOR reversibility: PASSED (double XOR restores original)\n");
}
```
Тест — применяем обфускацию дважды и проверяем что получили исходные данные.

## Задача 3: Детектор руткитов (rootkit)

### Что конкретно сделал

#### Constant memory для сигнатур (строки 34-39)
```cpp
__constant__ char d_rootkit_signatures[20][32] = {
    "necurs", "zeroaccess", "tdss", "rustock", "mebroot",
    "bootkit", "stuxnet", "flame", "duqu", "regin",
    "turla", "equation", "careto", "darkhotel", "grayfish",
    "finfisher", "hacking", "team", "uroburos", "snake"
};
```
**`__constant__`** — константная память на GPU. Она кэшируется и очень быстрая для чтения. Идеально для данных которые не меняются (сигнатуры).

#### Device-функции для работы со строками 
```cpp
__device__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0' && len < MAX_PROCESS_NAME) len++;
    return len;
}

__device__ void d_tolower_copy(char* dst, const char* src, int max_len) {
    int i = 0;
    while (src[i] != '\0' && i < max_len - 1) {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') c = c + 32;
        dst[i] = c;
        i++;
    }
    dst[i] = '\0';
}

__device__ int d_strstr_check(const char* haystack, const char* needle) {
    // ... поиск подстроки
}
```
На GPU нет стандартной библиотеки C, поэтому пишем свои функции для работы со строками. Все помечены `__device__`.

#### Kernel 1: Обнаружение SSDT hooks
```cpp
__global__ void detectSSDTHooks(const SSDTEntry* ssdt, RootkitAnalysis* results,
                                 int num_entries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    unsigned long long orig = ssdt[idx].original_address;
    unsigned long long curr = ssdt[idx].current_address;

    if (orig != curr) {
        results[idx].ssdt_hook_detected = 1;
        results[idx].total_score += 50;
    } else {
        results[idx].ssdt_hook_detected = 0;
    }
}
```
SSDT hook — когда руткит подменяет адрес системного вызова. Сравниваем оригинальный адрес с текущим.

#### Kernel 2: Сигнатурный анализ
```cpp
__global__ void signatureAnalysis(const ProcessInfo* processes, RootkitAnalysis* results,
                                   int num_processes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_processes) return;

    char lower_name[MAX_PROCESS_NAME];
    d_tolower_copy(lower_name, processes[idx].name, MAX_PROCESS_NAME);

    int match_found = 0;

    // Проверка по всем известным сигнатурам
    for (int s = 0; s < 20; s++) {
        if (d_strstr_check(lower_name, d_rootkit_signatures[s])) {
            match_found = 1;
            break;
        }
    }

    if (match_found) {
        results[idx].signature_match = 1;
        results[idx].total_score += 40;
    }
}
```
Проверяем имя процесса по базе известных руткитов. Обращаемся к `d_rootkit_signatures` — это constant memory, быстро.

#### Kernel 3: Проверка дерева процессов
```cpp
__global__ void processTreeVerification(const ProcessInfo* processes, RootkitAnalysis* results,
                                          int num_processes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_processes) return;

    int ppid = processes[idx].ppid;
    int pid = processes[idx].pid;

    if (ppid == 0 && pid != 1 && pid != 0) {
        results[idx].orphan_process = 1;
        results[idx].total_score += 30;
    }
}
```

#### Kernel 4: Эвристический анализ
```cpp
__global__ void heuristicAnalysis(const ProcessInfo* processes, RootkitAnalysis* results,
                                   int num_processes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_processes) return;

    int score = 0;

    // Скрытый процесс
    if (processes[idx].is_hidden) {
        score += 25;
    }

    // Подозрительный стартовый адрес
    unsigned long long addr = processes[idx].start_address;
    if (addr < 0x10000 || addr > 0x7FFFFFFFFFFF) {
        score += 15;
    }

    // Короткое имя
    int name_len = d_strlen(processes[idx].name);
    if (name_len < 3) {
        score += 10;
    }

    results[idx].heuristic_score = score;
    results[idx].total_score += score;
}
```
Набор эвристик — скрытые процессы, подозрительные адреса, короткие имена. Каждый признак добавляет очки риска.

#### Запуск нескольких kernel
```cpp
// Kernel 1: SSDT Hook Detection
detectSSDTHooks<<<ssdt_blocks, BLOCK_SIZE>>>(d_ssdt, d_results_ssdt, num_ssdt_entries);

// Kernel 2: Signature Analysis
signatureAnalysis<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);

// Kernel 3: Process Tree Verification
processTreeVerification<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);

// Kernel 4: Heuristic Analysis
heuristicAnalysis<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);
```
Запускаем 4 разных kernel последовательно. Каждый делает свой тип анализа. Kernel'ы 2-4 работают с одними данными процессов и накапливают score в `results`.

## Задача 4: Анализатор веб-уязвимостей (web_analyze)

### Что конкретно сделал

#### Device-функции для поиска подстрок (строки 24-74)
```cpp
__device__ int d_strstr(const char* haystack, const char* needle, int haystack_len) {
    int needle_len = 0;
    while (needle[needle_len] != '\0') needle_len++;

    if (needle_len == 0) return 1;
    if (haystack_len < needle_len) return 0;

    for (int i = 0; i <= haystack_len - needle_len; i++) {
        int match = 1;
        for (int j = 0; j < needle_len; j++) {
            if (haystack[i + j] != needle[j]) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

// Case-insensitive версия
__device__ int d_stristr(const char* haystack, const char* needle, int haystack_len) {
    // ... то же самое, но с приведением к нижнему регистру
}
```
Две версии поиска — обычная и case-insensitive. Для XSS нужна нечувствительная к регистру (тег `<SCRIPT>` тоже опасен).

#### Главный kernel для детектирования
```cpp
__global__ void detectVulnerabilities(const char* inputs, int* results, int n, int input_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const char* str = &inputs[idx * input_len];
    int len = d_strlen(str);
    int vuln = VULN_NONE;

    // === Path Traversal ===
    if (d_strstr(str, "../", len) ||
        d_strstr(str, "..\\", len) ||
        d_strstr(str, "%2e%2e%2f", len) ||
        d_strstr(str, "/etc/passwd", len) ||
        d_strstr(str, "c:\\windows", len)) {
        vuln = VULN_PATH_TRAVERSAL;
    }

    // === XSS ===
    else if (d_stristr(str, "<script", len) ||
             d_stristr(str, "javascript:", len) ||
             d_stristr(str, "onerror=", len) ||
             d_stristr(str, "onclick=", len) ||
             d_stristr(str, "alert(", len) ||
             d_stristr(str, "document.cookie", len)) {
        vuln = VULN_XSS;
    }

    // === SQL Injection ===
    else if (d_stristr(str, "' or '", len) ||
             d_stristr(str, "' or 1=1", len) ||
             d_stristr(str, "union select", len) ||
             d_stristr(str, "drop table", len) ||
             d_stristr(str, "'; drop", len)) {
        vuln = VULN_SQL_INJECTION;
    }

    // === Command Injection ===
    else if (d_strstr(str, "; ls", len) ||
             d_strstr(str, "| cat", len) ||
             d_strstr(str, "$(", len) ||
             d_strstr(str, "`", len) ||
             d_strstr(str, "/bin/sh", len)) {
        vuln = VULN_CMD_INJECTION;
    }

    // ... ещё LDAP, XXE, SSRF

    results[idx] = vuln;
}
```
Большой kernel с кучей проверок. Каждый поток проверяет одну строку по всем паттернам. Приоритет проверок важен — если нашли Path Traversal, уже не проверяем XSS.

#### Тестовые данные с expected результатами
```cpp
TestCase testCases[] = {
    // Clean inputs
    {"hello world", VULN_NONE},
    {"user@example.com", VULN_NONE},

    // Path Traversal
    {"../../etc/passwd", VULN_PATH_TRAVERSAL},
    {"%2e%2e%2f%2e%2e%2fetc/passwd", VULN_PATH_TRAVERSAL},

    // XSS
    {"<script>alert('XSS')</script>", VULN_XSS},
    {"<img src=x onerror=alert(1)>", VULN_XSS},

    // SQL Injection
    {"' OR '1'='1", VULN_SQL_INJECTION},
    {"1; DROP TABLE users--", VULN_SQL_INJECTION},

    // Command Injection
    {"; ls -la", VULN_CMD_INJECTION},
    {"$(whoami)", VULN_CMD_INJECTION},

    // ... много тестов
};
```
Хороший набор тестов — и чистые входы, и разные типы атак. Проверяем что детектор не false positive'ит на нормальных данных.

#### Подсчёт метрик качества
```cpp
int tp = 0, tn = 0, fp = 0, fn = 0;

for (int i = 0; i < n; i++) {
    int expected = h_expected[i];
    int detected = h_results[i];

    int is_expected_vuln = (expected != VULN_NONE);
    int is_detected_vuln = (detected != VULN_NONE);

    if (is_expected_vuln && is_detected_vuln) tp++;
    else if (!is_expected_vuln && !is_detected_vuln) tn++;
    else if (!is_expected_vuln && is_detected_vuln) fp++;
    else if (is_expected_vuln && !is_detected_vuln) fn++;
}

double precision = (double)tp / (tp + fp);
double recall = (double)tp / (tp + fn);
double f1 = 2 * precision * recall / (precision + recall);
```
Считаем стандартные метрики классификации:
- **Precision** — из того что нашли, сколько реально уязвимости
- **Recall** — из всех уязвимостей, сколько нашли
- **F1** — гармоническое среднее

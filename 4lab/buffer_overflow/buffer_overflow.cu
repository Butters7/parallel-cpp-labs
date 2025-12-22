#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_STRING_LEN 256
#define BLOCK_SIZE 256

// Результат анализа
typedef struct {
    int is_overflow;      // 1 = переполнение, 0 = норма
    int string_length;    // фактическая длина строки
    int buffer_limit;     // лимит буфера
    int overflow_amount;  // на сколько превышен лимит
} AnalysisResult;

__global__ void analyzeBufferOverflow(const char* strings, AnalysisResult* results,
                                       const int* limits, int n, int string_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const char* str = &strings[idx * string_size];
    int limit = limits[idx];

    // Вычисляем длину строки с гарантированными границами
    int len = 0;
    // Явная проверка границ для предотвращения переполнения
    while (len < string_size) {
        if (str[len] == '\0') break;
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

void analyzeBufferOverflowCPU(const char* strings, AnalysisResult* results,
                               const int* limits, int n, int string_size) {
    for (int idx = 0; idx < n; idx++) {
        const char* str = &strings[idx * string_size];
        int limit = limits[idx];

        // Вычисляем длину строки с гарантированными границами
        int len = 0;
        while (len < string_size) {
            if (str[len] == '\0') break;
            len++;
        }

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
}

void generateTestData(char* strings, int* limits, int n, int string_size,
                      int min_limit, int max_limit, float overflow_ratio) {
    srand(42); // Фиксированный seed для воспроизводимости

    for (int i = 0; i < n; i++) {
        // Генерируем случайный лимит буфера
        int limit = min_limit + rand() % (max_limit - min_limit + 1);
        limits[i] = limit;

        // Определяем длину строки
        int str_len;
        if ((float)rand() / RAND_MAX < overflow_ratio) {
            // Создаём переполнение: длина > лимита
            str_len = limit + 1 + rand() % 50;
        } else {
            // Норма: длина <= лимита
            str_len = rand() % (limit + 1);
        }

        // Ограничиваем максимальной длиной
        if (str_len >= string_size) {
            str_len = string_size - 1;
        }

        // Заполняем строку случайными символами
        char* str = &strings[i * string_size];
        for (int j = 0; j < str_len; j++) {
            str[j] = 'a' + rand() % 26;
        }
        str[str_len] = '\0';

        // Заполняем остаток нулями
        for (int j = str_len + 1; j < string_size; j++) {
            str[j] = '\0';
        }
    }
}

void printResults(const char* strings, AnalysisResult* results, int n,
                  int string_size, int max_display) {
    printf("\n%-6s | %-8s | %-6s | %-10s | %-30s\n",
           "Index", "Status", "Limit", "Length", "String (truncated)");
    printf("-------+----------+--------+------------+--------------------------------\n");

    int overflows_shown = 0;
    int normal_shown = 0;

    // Сначала показываем переполнения
    for (int i = 0; i < n && overflows_shown < max_display / 2; i++) {
        if (results[i].is_overflow) {
            const char* str = &strings[i * string_size];
            printf("%-6d | %-8s | %-6d | %-10d | %.30s%s\n",
                   i,
                   "OVERFLOW",
                   results[i].buffer_limit,
                   results[i].string_length,
                   str,
                   results[i].string_length > 30 ? "..." : "");
            overflows_shown++;
        }
    }

    // Затем нормальные
    for (int i = 0; i < n && normal_shown < max_display / 2; i++) {
        if (!results[i].is_overflow) {
            const char* str = &strings[i * string_size];
            printf("%-6d | %-8s | %-6d | %-10d | %.30s%s\n",
                   i,
                   "OK",
                   results[i].buffer_limit,
                   results[i].string_length,
                   str,
                   results[i].string_length > 30 ? "..." : "");
            normal_shown++;
        }
    }
}

void printStatistics(AnalysisResult* results, int n) {
    int total_overflows = 0;
    int max_overflow = 0;
    int min_overflow = INT_MAX;
    long long total_overflow_amount = 0;

    for (int i = 0; i < n; i++) {
        if (results[i].is_overflow) {
            total_overflows++;
            total_overflow_amount += results[i].overflow_amount;
            if (results[i].overflow_amount > max_overflow) {
                max_overflow = results[i].overflow_amount;
            }
            if (results[i].overflow_amount < min_overflow) {
                min_overflow = results[i].overflow_amount;
            }
        }
    }

    printf("\n=== Statistics ===\n");
    printf("Total strings analyzed:    %d\n", n);
    printf("Buffer overflows detected: %d (%.2f%%)\n",
           total_overflows, (float)total_overflows / n * 100);
    printf("Safe strings:              %d (%.2f%%)\n",
           n - total_overflows, (float)(n - total_overflows) / n * 100);

    if (total_overflows > 0) {
        printf("\nOverflow details:\n");
        printf("  Min overflow: %d bytes\n", min_overflow);
        printf("  Max overflow: %d bytes\n", max_overflow);
        printf("  Avg overflow: %.2f bytes\n",
               (float)total_overflow_amount / total_overflows);
    }
}

int main() {
    printf("==============================================\n");
    printf("  Buffer Overflow Analyzer (CUDA)\n");
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
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Block Size: %d threads\n\n", BLOCK_SIZE);

    // Параметры теста
    const int n = 1000000;  // 1 миллион строк
    const int string_size = MAX_STRING_LEN;
    const int min_limit = 16;
    const int max_limit = 128;
    const float overflow_ratio = 0.3f;  // 30% строк с переполнением

    printf("Test parameters:\n");
    printf("  Strings count:    %d\n", n);
    printf("  Buffer limits:    %d - %d bytes\n", min_limit, max_limit);
    printf("  Overflow ratio:   %.0f%%\n\n", overflow_ratio * 100);

    // Размеры данных
    size_t strings_size = (size_t)n * string_size;
    size_t results_size = n * sizeof(AnalysisResult);
    size_t limits_size = n * sizeof(int);

    printf("Memory allocation:\n");
    printf("  Strings:  %.2f MB\n", strings_size / (1024.0 * 1024.0));
    printf("  Results:  %.2f MB\n", results_size / (1024.0 * 1024.0));
    printf("  Limits:   %.2f MB\n\n", limits_size / (1024.0 * 1024.0));

    // Выделение Host памяти
    char* h_strings = (char*)malloc(strings_size);
    int* h_limits = (int*)malloc(limits_size);
    AnalysisResult* h_results_gpu = (AnalysisResult*)malloc(results_size);
    AnalysisResult* h_results_cpu = (AnalysisResult*)malloc(results_size);

    if (!h_strings || !h_limits || !h_results_gpu || !h_results_cpu) {
        printf("ERROR: Host memory allocation failed!\n");
        if (h_strings) free(h_strings);
        if (h_limits) free(h_limits);
        if (h_results_gpu) free(h_results_gpu);
        if (h_results_cpu) free(h_results_cpu);
        return 1;
    }

    // Генерация тестовых данных
    printf("Generating test data...\n");
    generateTestData(h_strings, h_limits, n, string_size, min_limit, max_limit, overflow_ratio);
    printf("Test data generated.\n\n");

    // Выделение Device памяти
    char* d_strings;
    int* d_limits;
    AnalysisResult* d_results;

    cudaError_t err;
    err = cudaMalloc(&d_strings, strings_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_strings: %s\n", cudaGetErrorString(err));
        free(h_strings); free(h_limits); free(h_results_gpu); free(h_results_cpu);
        return 1;
    }
    err = cudaMalloc(&d_limits, limits_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_limits: %s\n", cudaGetErrorString(err));
        cudaFree(d_strings);
        free(h_strings); free(h_limits); free(h_results_gpu); free(h_results_cpu);
        return 1;
    }
    err = cudaMalloc(&d_results, results_size);
    if (err != cudaSuccess) {
        printf("ERROR: cudaMalloc failed for d_results: %s\n", cudaGetErrorString(err));
        cudaFree(d_strings); cudaFree(d_limits);
        free(h_strings); free(h_limits); free(h_results_gpu); free(h_results_cpu);
        return 1;
    }

    // GPU анализ
    printf("=== GPU Analysis ===\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Замер времени копирования Host -> Device
    cudaEventRecord(start);
    cudaMemcpy(d_strings, h_strings, strings_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_limits, h_limits, limits_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copy_to_device_ms;
    cudaEventElapsedTime(&copy_to_device_ms, start, stop);

    // Запуск kernel
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    analyzeBufferOverflow<<<numBlocks, BLOCK_SIZE>>>(d_strings, d_results, d_limits, n, string_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Проверка на ошибки выполнения kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_strings); cudaFree(d_limits); cudaFree(d_results);
        free(h_strings); free(h_limits); free(h_results_gpu); free(h_results_cpu);
        return 1;
    }

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Копирование результатов Device -> Host
    cudaEventRecord(start);
    cudaMemcpy(h_results_gpu, d_results, results_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copy_to_host_ms;
    cudaEventElapsedTime(&copy_to_host_ms, start, stop);

    float total_gpu_ms = copy_to_device_ms + kernel_ms + copy_to_host_ms;

    printf("  Copy to GPU:   %.3f ms\n", copy_to_device_ms);
    printf("  Kernel exec:   %.3f ms\n", kernel_ms);
    printf("  Copy to CPU:   %.3f ms\n", copy_to_host_ms);
    printf("  Total GPU:     %.3f ms\n", total_gpu_ms);
    printf("  Throughput:    %.2f million strings/sec\n\n", n / (total_gpu_ms / 1000.0) / 1000000.0);

    // CPU анализ (для сравнения)
    printf("=== CPU Analysis ===\n");

    clock_t cpu_start = clock();
    analyzeBufferOverflowCPU(h_strings, h_results_cpu, h_limits, n, string_size);
    clock_t cpu_end = clock();

    float cpu_ms = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f;

    printf("  Total CPU:     %.3f ms\n", cpu_ms);
    printf("  Throughput:    %.2f million strings/sec\n\n", n / (cpu_ms / 1000.0) / 1000000.0);

    // Сравнение производительности
    float speedup = cpu_ms / total_gpu_ms;
    float kernel_speedup = cpu_ms / kernel_ms;

    printf("==============================================\n");
    printf("  Performance Comparison\n");
    printf("==============================================\n");
    printf("CPU time:              %.3f ms\n", cpu_ms);
    printf("GPU time (total):      %.3f ms\n", total_gpu_ms);
    printf("GPU time (kernel):     %.3f ms\n", kernel_ms);
    printf("\n");
    printf("Speedup (total):       %.2fx\n", speedup);
    printf("Speedup (kernel only): %.2fx\n", kernel_speedup);
    printf("\n");
    printf("Target speedup (100x): %s\n", speedup >= 100 ? "PASS" :
           (kernel_speedup >= 100 ? "PASS (kernel)" : "FAIL"));

    // Верификация результатов
    printf("\n=== Verification ===\n");
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (h_results_gpu[i].is_overflow != h_results_cpu[i].is_overflow ||
            h_results_gpu[i].string_length != h_results_cpu[i].string_length ||
            h_results_gpu[i].overflow_amount != h_results_cpu[i].overflow_amount) {
            mismatches++;
            if (mismatches <= 5) {
                printf("Mismatch at index %d:\n", i);
                printf("  GPU: overflow=%d, len=%d, amount=%d\n",
                       h_results_gpu[i].is_overflow, h_results_gpu[i].string_length,
                       h_results_gpu[i].overflow_amount);
                printf("  CPU: overflow=%d, len=%d, amount=%d\n",
                       h_results_cpu[i].is_overflow, h_results_cpu[i].string_length,
                       h_results_cpu[i].overflow_amount);
            }
        }
    }

    if (mismatches == 0) {
        printf("All results match! Verification PASSED.\n");
    } else {
        printf("Found %d mismatches. Verification FAILED.\n", mismatches);
    }

    // Вывод статистики и примеров
    printStatistics(h_results_gpu, n);
    printResults(h_strings, h_results_gpu, n, string_size, 10);

    printf("\n=== Batch Processing Demo ===\n");

    int batch_sizes[] = {10000, 100000, 500000, 1000000};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("%-12s | %-12s | %-15s | %-10s\n",
           "Batch Size", "GPU Time", "Throughput", "Speedup");
    printf("-------------+--------------+-----------------+------------\n");

    for (int b = 0; b < num_batches; b++) {
        int batch_n = batch_sizes[b];
        if (batch_n > n) batch_n = n;

        int batch_blocks = (batch_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // GPU
        cudaEventRecord(start);
        analyzeBufferOverflow<<<batch_blocks, BLOCK_SIZE>>>(d_strings, d_results, d_limits, batch_n, string_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float batch_gpu_ms;
        cudaEventElapsedTime(&batch_gpu_ms, start, stop);

        // CPU
        clock_t batch_cpu_start = clock();
        analyzeBufferOverflowCPU(h_strings, h_results_cpu, h_limits, batch_n, string_size);
        clock_t batch_cpu_end = clock();
        float batch_cpu_ms = (float)(batch_cpu_end - batch_cpu_start) / CLOCKS_PER_SEC * 1000.0f;

        float batch_speedup = batch_cpu_ms / batch_gpu_ms;
        float batch_throughput = batch_n / (batch_gpu_ms / 1000.0) / 1000000.0;

        printf("%-12d | %-12.3f | %-15.2f | %-10.2fx\n",
               batch_n, batch_gpu_ms, batch_throughput, batch_speedup);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_strings);
    cudaFree(d_limits);
    cudaFree(d_results);

    free(h_strings);
    free(h_limits);
    free(h_results_gpu);
    free(h_results_cpu);

    printf("\n==============================================\n");
    printf("  Analysis Complete\n");
    printf("==============================================\n");

    return 0;
}

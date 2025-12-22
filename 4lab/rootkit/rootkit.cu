#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_PROCESS_NAME 64
#define MAX_FUNCTION_NAME 32

// Структуры данных
typedef struct {
    int pid;
    int ppid;
    char name[MAX_PROCESS_NAME];
    unsigned long long start_address;
    int is_hidden;
} ProcessInfo;

typedef struct {
    char function_name[MAX_FUNCTION_NAME];
    unsigned long long original_address;
    unsigned long long current_address;
} SSDTEntry;

typedef struct {
    int ssdt_hook_detected;
    int signature_match;
    int orphan_process;
    int heuristic_score;
    int total_score;
} RootkitAnalysis;

// Известные сигнатуры руткитов
// Выравнивание для оптимального доступа из constant memory
__constant__ __align__(16) char d_rootkit_signatures[20][32] = {
    "necurs", "zeroaccess", "tdss", "rustock", "mebroot",
    "bootkit", "stuxnet", "flame", "duqu", "regin",
    "turla", "equation", "careto", "darkhotel", "grayfish",
    "finfisher", "hacking", "team", "uroburos", "snake"
};

// Device функции
__device__ int d_strlen(const char* str) {
    if (!str) return 0;
    int len = 0;
    while (len < MAX_PROCESS_NAME && str[len] != '\0') len++;
    return len;
}

__device__ void d_tolower_copy(char* dst, const char* src, int max_len) {
    if (!dst || !src || max_len <= 0) return;
    int i = 0;
    while (i < max_len - 1 && src[i] != '\0') {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') c = c + 32;
        dst[i] = c;
        i++;
    }
    if (i < max_len) dst[i] = '\0';
}

__device__ int d_strstr_check(const char* haystack, const char* needle) {
    if (!haystack || !needle) return 0;

    int h_len = d_strlen(haystack);
    int n_len = 0;
    while (n_len < 32 && needle[n_len] != '\0') n_len++;

    if (n_len == 0 || n_len > h_len) return 0;

    for (int i = 0; i <= h_len - n_len; i++) {
        int match = 1;
        for (int j = 0; j < n_len; j++) {
            if (i + j >= MAX_PROCESS_NAME) {
                match = 0;
                break;
            }
            if (haystack[i + j] != needle[j]) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

// Kernel 1: Обнаружение SSDT hooks
__global__ void detectSSDTHooks(const SSDTEntry* ssdt, RootkitAnalysis* results,
                                 int num_entries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    // Проверка: отличается ли текущий адрес от оригинального
    unsigned long long orig = ssdt[idx].original_address;
    unsigned long long curr = ssdt[idx].current_address;

    if (orig != curr) {
        results[idx].ssdt_hook_detected = 1;
        atomicAdd(&results[idx].total_score, 50);  // Атомарное добавление
    } else {
        results[idx].ssdt_hook_detected = 0;
    }
}

// Kernel 2: Сигнатурный анализ имён процессов
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
        atomicAdd(&results[idx].total_score, 40);  // Атомарное добавление
    } else {
        results[idx].signature_match = 0;
    }
}

// Kernel 3: Проверка дерева процессов (orphan процессы с ppid=0)
__global__ void processTreeVerification(const ProcessInfo* processes, RootkitAnalysis* results,
                                          int num_processes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_processes) return;

    int ppid = processes[idx].ppid;
    int pid = processes[idx].pid;

    // Процесс с ppid=0, но не init (pid=1) - подозрительный
    if (ppid == 0 && pid != 1 && pid != 0) {
        results[idx].orphan_process = 1;
        atomicAdd(&results[idx].total_score, 30);  // Атомарное добавление
    } else {
        results[idx].orphan_process = 0;
    }
}

// Kernel 4: Эвристический анализ
__global__ void heuristicAnalysis(const ProcessInfo* processes, RootkitAnalysis* results,
                                   int num_processes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_processes) return;

    int score = 0;

    // Скрытый процесс
    if (processes[idx].is_hidden) {
        score += 25;
    }

    // Подозрительный стартовый адрес (вне нормального диапазона)
    unsigned long long addr = processes[idx].start_address;
    if (addr < 0x10000 || addr > 0x7FFFFFFFFFFF) {
        score += 15;
    }

    // Короткое или пустое имя
    int name_len = d_strlen(processes[idx].name);
    if (name_len < 3) {
        score += 10;
    }

    // PID в подозрительном диапазоне (очень низкий или очень высокий)
    int pid = processes[idx].pid;
    if (pid > 65000 || (pid < 100 && pid > 10)) {
        score += 5;
    }

    results[idx].heuristic_score = score;
    atomicAdd(&results[idx].total_score, score);  // Атомарное добавление
}

void generateProcessData(ProcessInfo* processes, int n) {
    srand(42);

    const char* normal_names[] = {
        "systemd", "sshd", "bash", "python", "nginx", "apache2",
        "mysql", "postgres", "redis", "node", "java", "docker"
    };

    const char* suspicious_names[] = {
        "necurs_agent", "zeroaccess_srv", "hidden_tdss", "rootkit_test",
        "mebroot_loader", "snake_backdoor"
    };

    for (int i = 0; i < n; i++) {
        processes[i].pid = 100 + i;
        processes[i].ppid = (i == 0) ? 0 : (rand() % i + 1);
        processes[i].start_address = 0x400000 + rand() % 0x100000;
        processes[i].is_hidden = 0;

        // 90% нормальные процессы
        if (rand() % 100 < 90) {
            strcpy(processes[i].name, normal_names[rand() % 12]);
        } else {
            strcpy(processes[i].name, suspicious_names[rand() % 6]);
        }

        // 5% скрытых процессов
        if (rand() % 100 < 5) {
            processes[i].is_hidden = 1;
        }

        // 3% orphan процессов
        if (rand() % 100 < 3 && i > 10) {
            processes[i].ppid = 0;
        }

        // 2% с подозрительным адресом
        if (rand() % 100 < 2) {
            processes[i].start_address = rand() % 0x1000;
        }
    }
}

void generateSSDTData(SSDTEntry* ssdt, int n) {
    srand(42);

    const char* syscall_names[] = {
        "NtOpenProcess", "NtReadVirtualMemory", "NtWriteVirtualMemory",
        "NtQuerySystemInformation", "NtTerminateProcess", "NtCreateFile",
        "NtOpenFile", "NtQueryDirectoryFile", "NtDeviceIoControlFile",
        "NtCreateProcess", "NtCreateThread", "NtLoadDriver"
    };

    for (int i = 0; i < n; i++) {
        strcpy(ssdt[i].function_name, syscall_names[i % 12]);
        ssdt[i].original_address = 0xFFFFF80000000000ULL + i * 0x100;

        // 10% hooked entries
        if (rand() % 100 < 10) {
            ssdt[i].current_address = 0xDEADBEEF00000000ULL + rand() % 0x10000;
        } else {
            ssdt[i].current_address = ssdt[i].original_address;
        }
    }
}

int main() {
    printf("==============================================\n");
    printf("  Rootkit Detector (CUDA)\n");
    printf("  SSDT Hooks | Signatures | Process Tree\n");
    printf("==============================================\n\n");

    // GPU Info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Block Size: %d threads\n\n", BLOCK_SIZE);

    // Параметры теста
    const int num_processes = 100000;
    const int num_ssdt_entries = 1000;

    printf("Test parameters:\n");
    printf("  Processes:     %d\n", num_processes);
    printf("  SSDT entries:  %d\n\n", num_ssdt_entries);

    // Выделение памяти на хосте
    size_t proc_size = num_processes * sizeof(ProcessInfo);
    size_t ssdt_size = num_ssdt_entries * sizeof(SSDTEntry);
    size_t result_proc_size = num_processes * sizeof(RootkitAnalysis);
    size_t result_ssdt_size = num_ssdt_entries * sizeof(RootkitAnalysis);

    ProcessInfo* h_processes = (ProcessInfo*)malloc(proc_size);
    SSDTEntry* h_ssdt = (SSDTEntry*)malloc(ssdt_size);
    RootkitAnalysis* h_results_proc = (RootkitAnalysis*)malloc(result_proc_size);
    RootkitAnalysis* h_results_ssdt = (RootkitAnalysis*)malloc(result_ssdt_size);

    // Генерация данных
    printf("Generating test data...\n");
    generateProcessData(h_processes, num_processes);
    generateSSDTData(h_ssdt, num_ssdt_entries);
    printf("  Data generated\n\n");

    // Выделение памяти на GPU
    ProcessInfo* d_processes;
    SSDTEntry* d_ssdt;
    RootkitAnalysis* d_results_proc;
    RootkitAnalysis* d_results_ssdt;

    cudaMalloc(&d_processes, proc_size);
    cudaMalloc(&d_ssdt, ssdt_size);
    cudaMalloc(&d_results_proc, result_proc_size);
    cudaMalloc(&d_results_ssdt, result_ssdt_size);

    // Инициализация результатов нулями
    cudaMemset(d_results_proc, 0, result_proc_size);
    cudaMemset(d_results_ssdt, 0, result_ssdt_size);

    // CUDA Events для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Копирование данных на GPU
    printf("=== GPU Analysis ===\n");

    cudaEventRecord(start);
    cudaMemcpy(d_processes, h_processes, proc_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssdt, h_ssdt, ssdt_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copy_to_ms;
    cudaEventElapsedTime(&copy_to_ms, start, stop);

    // Динамическое распределение блоков
    int proc_blocks = (num_processes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int ssdt_blocks = (num_ssdt_entries + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Dynamic block distribution:\n");
    printf("  Process kernels: %d blocks x %d threads\n", proc_blocks, BLOCK_SIZE);
    printf("  SSDT kernel:     %d blocks x %d threads\n\n", ssdt_blocks, BLOCK_SIZE);

    float total_kernel_ms = 0;

    // Kernel 1: SSDT Hook Detection
    cudaEventRecord(start);
    detectSSDTHooks<<<ssdt_blocks, BLOCK_SIZE>>>(d_ssdt, d_results_ssdt, num_ssdt_entries);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ssdt_ms;
    cudaEventElapsedTime(&ssdt_ms, start, stop);
    total_kernel_ms += ssdt_ms;
    printf("SSDT Hook Detection:    %.3f ms\n", ssdt_ms);

    // Kernel 2: Signature Analysis
    cudaEventRecord(start);
    signatureAnalysis<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sig_ms;
    cudaEventElapsedTime(&sig_ms, start, stop);
    total_kernel_ms += sig_ms;
    printf("Signature Analysis:     %.3f ms\n", sig_ms);

    // Kernel 3: Process Tree Verification
    cudaEventRecord(start);
    processTreeVerification<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tree_ms;
    cudaEventElapsedTime(&tree_ms, start, stop);
    total_kernel_ms += tree_ms;
    printf("Process Tree Check:     %.3f ms\n", tree_ms);

    // Kernel 4: Heuristic Analysis
    cudaEventRecord(start);
    heuristicAnalysis<<<proc_blocks, BLOCK_SIZE>>>(d_processes, d_results_proc, num_processes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float heur_ms;
    cudaEventElapsedTime(&heur_ms, start, stop);
    total_kernel_ms += heur_ms;
    printf("Heuristic Analysis:     %.3f ms\n", heur_ms);

    // Копирование результатов обратно
    cudaEventRecord(start);
    cudaMemcpy(h_results_proc, d_results_proc, result_proc_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results_ssdt, d_results_ssdt, result_ssdt_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copy_from_ms;
    cudaEventElapsedTime(&copy_from_ms, start, stop);

    float total_ms = copy_to_ms + total_kernel_ms + copy_from_ms;

    printf("\n=== Timing Summary ===\n");
    printf("Copy to GPU:    %.3f ms\n", copy_to_ms);
    printf("All kernels:    %.3f ms\n", total_kernel_ms);
    printf("Copy to CPU:    %.3f ms\n", copy_from_ms);
    printf("Total:          %.3f ms\n", total_ms);
    printf("Throughput:     %.2f million checks/sec\n",
           (num_processes * 3 + num_ssdt_entries) / (total_ms / 1000.0) / 1000000.0);

    // Подсчёт статистики
    printf("\n=== Detection Results ===\n");

    int ssdt_hooks = 0;
    for (int i = 0; i < num_ssdt_entries; i++) {
        if (h_results_ssdt[i].ssdt_hook_detected) ssdt_hooks++;
    }

    int sig_matches = 0, orphans = 0, high_risk = 0;
    for (int i = 0; i < num_processes; i++) {
        if (h_results_proc[i].signature_match) sig_matches++;
        if (h_results_proc[i].orphan_process) orphans++;
        if (h_results_proc[i].total_score >= 50) high_risk++;
    }

    printf("SSDT Hooks detected:     %d / %d\n", ssdt_hooks, num_ssdt_entries);
    printf("Signature matches:       %d / %d\n", sig_matches, num_processes);
    printf("Orphan processes:        %d / %d\n", orphans, num_processes);
    printf("High-risk processes:     %d (score >= 50)\n", high_risk);

    // Примеры обнаружений
    printf("\n=== Detection Examples ===\n");
    printf("%-6s | %-20s | %-8s | %-8s | %-8s | %-6s\n",
           "PID", "Name", "Sig", "Orphan", "Heur", "Score");
    printf("-------+----------------------+----------+----------+----------+--------\n");

    int shown = 0;
    for (int i = 0; i < num_processes && shown < 10; i++) {
        if (h_results_proc[i].total_score > 0) {
            printf("%-6d | %-20s | %-8s | %-8s | %-8d | %-6d\n",
                   h_processes[i].pid,
                   h_processes[i].name,
                   h_results_proc[i].signature_match ? "YES" : "NO",
                   h_results_proc[i].orphan_process ? "YES" : "NO",
                   h_results_proc[i].heuristic_score,
                   h_results_proc[i].total_score);
            shown++;
        }
    }

    // SSDT hooks примеры
    printf("\n=== SSDT Hook Examples ===\n");
    printf("%-25s | %-18s | %-18s | %-6s\n",
           "Function", "Original", "Current", "Hooked");
    printf("--------------------------+--------------------+--------------------+--------\n");

    shown = 0;
    for (int i = 0; i < num_ssdt_entries && shown < 5; i++) {
        if (h_results_ssdt[i].ssdt_hook_detected) {
            printf("%-25s | 0x%016llX | 0x%016llX | YES\n",
                   h_ssdt[i].function_name,
                   h_ssdt[i].original_address,
                   h_ssdt[i].current_address);
            shown++;
        }
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_processes);
    cudaFree(d_ssdt);
    cudaFree(d_results_proc);
    cudaFree(d_results_ssdt);

    free(h_processes);
    free(h_ssdt);
    free(h_results_proc);
    free(h_results_ssdt);

    printf("\n==============================================\n");
    printf("  Rootkit Detection Complete\n");
    printf("==============================================\n");

    return 0;
}

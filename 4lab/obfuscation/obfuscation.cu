#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Бинарный поиск для определения принадлежности символа к строке
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

void generateTestData(char** data, int** offsets, int** keys, int** lengths,
                      int num_strings, int min_len, int max_len, int* total_chars) {
    srand(42);

    *offsets = (int*)malloc(num_strings * sizeof(int));
    *keys = (int*)malloc(num_strings * sizeof(int));
    *lengths = (int*)malloc(num_strings * sizeof(int));

    int offset = 0;
    for (int i = 0; i < num_strings; i++) {
        (*lengths)[i] = min_len + rand() % (max_len - min_len + 1);
        (*offsets)[i] = offset;
        (*keys)[i] = rand() % 256;
        offset += (*lengths)[i];
    }
    *total_chars = offset;

    *data = (char*)malloc(*total_chars);

    for (int i = 0; i < *total_chars; i++) {
        (*data)[i] = 32 + rand() % 95;
    }
}

int main() {
    printf("==============================================\n");
    printf("  String Obfuscation (CUDA)\n");
    printf("  Algorithm: XOR with position-modified key\n");
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
    printf("Block Size: %d threads\n\n", BLOCK_SIZE);

    // Параметры теста
    const int num_strings = 100000;
    const int min_len = 50;
    const int max_len = 200;

    printf("Test parameters:\n");
    printf("  Strings:    %d\n", num_strings);
    printf("  Length:     %d - %d chars\n", min_len, max_len);

    // Генерация данных
    char* h_data;
    int* h_offsets;
    int* h_keys;
    int* h_lengths;
    int total_chars;

    printf("\nGenerating test data...\n");
    generateTestData(&h_data, &h_offsets, &h_keys, &h_lengths,
                     num_strings, min_len, max_len, &total_chars);

    printf("  Total characters: %d\n", total_chars);
    printf("  Data size: %.2f MB\n\n", total_chars / (1024.0 * 1024.0));

    // Сохраняем копию оригинала
    char* h_original = (char*)malloc(total_chars);
    memcpy(h_original, h_data, total_chars);

    // Device память
    char* d_data;
    int* d_offsets;
    int* d_keys;

    cudaMalloc(&d_data, total_chars);
    cudaMalloc(&d_offsets, num_strings * sizeof(int));
    cudaMalloc(&d_keys, num_strings * sizeof(int));

    // GPU Obfuscation
    printf("=== GPU Obfuscation ===\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy to device
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, num_strings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, h_keys, num_strings * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copy_to_device_ms;
    cudaEventElapsedTime(&copy_to_device_ms, start, stop);

    // Kernel
    int numBlocks = (total_chars + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    obfuscateKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_offsets, d_keys, num_strings, total_chars);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Copy back
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, total_chars, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copy_to_host_ms;
    cudaEventElapsedTime(&copy_to_host_ms, start, stop);

    float total_gpu_ms = copy_to_device_ms + kernel_ms + copy_to_host_ms;

    printf("  Copy to GPU:   %.3f ms\n", copy_to_device_ms);
    printf("  Kernel exec:   %.3f ms\n", kernel_ms);
    printf("  Copy to CPU:   %.3f ms\n", copy_to_host_ms);
    printf("  Total:         %.3f ms\n", total_gpu_ms);
    printf("  Throughput:    %.2f million chars/sec\n",
           total_chars / (total_gpu_ms / 1000.0) / 1000000.0);

    // Проверка XOR: obfuscate(obfuscate(x)) == x
    printf("\n=== XOR Reversibility Test ===\n");

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
    } else {
        printf("XOR reversibility: FAILED (%d mismatches)\n", mismatches);
    }

    // Восстанавливаем и обфусцируем для отображения
    memcpy(h_data, h_original, total_chars);
    cudaMemcpy(d_data, h_data, total_chars, cudaMemcpyHostToDevice);
    obfuscateKernel<<<numBlocks, BLOCK_SIZE>>>(d_data, d_offsets, d_keys, num_strings, total_chars);
    cudaMemcpy(h_data, d_data, total_chars, cudaMemcpyDeviceToHost);

    printf("\n=== Obfuscation Examples ===\n");
    printf("%-6s | %-20s | %-30s | %-6s\n", "Index", "Original", "Obfuscated (hex)", "Key");
    printf("-------+----------------------+--------------------------------+--------\n");

    for (int i = 0; i < 5 && i < num_strings; i++) {
        char orig_buf[21], obf_buf[61];
        int len = (h_lengths[i] < 10) ? h_lengths[i] : 10;

        memcpy(orig_buf, &h_original[h_offsets[i]], len);
        orig_buf[len] = '\0';

        obf_buf[0] = '\0';
        for (int j = 0; j < len; j++) {
            char hex[4];
            sprintf(hex, "%02X ", (unsigned char)h_data[h_offsets[i] + j]);
            strcat(obf_buf, hex);
        }

        printf("%-6d | %-20s | %-30s | %-6d\n", i, orig_buf, obf_buf, h_keys[i]);
    }

    printf("\n=== Binary Search Demo ===\n");
    printf("Character position -> String index mapping:\n");
    printf("%-15s | %-12s | %-15s\n", "Char Position", "String Index", "Pos in String");
    printf("----------------+--------------+-----------------\n");

    int test_positions[] = {0, 100, 1000, 5000, total_chars / 2, total_chars - 1};
    int num_tests = sizeof(test_positions) / sizeof(test_positions[0]);

    for (int t = 0; t < num_tests; t++) {
        int char_idx = test_positions[t];
        if (char_idx >= total_chars) char_idx = total_chars - 1;

        // Линейный поиск для проверки
        int string_idx = 0;
        for (int i = 0; i < num_strings; i++) {
            if (h_offsets[i] <= char_idx) {
                string_idx = i;
            } else {
                break;
            }
        }

        int pos_in_string = char_idx - h_offsets[string_idx];
        printf("%-15d | %-12d | %-15d\n", char_idx, string_idx, pos_in_string);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_keys);

    free(h_data);
    free(h_original);
    free(h_offsets);
    free(h_keys);
    free(h_lengths);

    printf("\n==============================================\n");
    printf("  Obfuscation Complete\n");
    printf("==============================================\n");

    return 0;
}

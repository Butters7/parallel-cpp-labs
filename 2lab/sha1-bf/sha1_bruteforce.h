#ifndef SHA1_BRUTEFORCE_H
#define SHA1_BRUTEFORCE_H

#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>

#define MAX_PASSWORD_LEN 256
#define HASH_SIZE 20

// SHA1 функция
void sha1(const char* message, size_t length, uint8_t* digest);

// Сравнение хешей
int hash_matches(uint8_t* hash1, uint8_t* hash2);

// Вывод хеша
void print_hash(uint8_t* hash);

// Копирование строки
char* strcpy_compact(char* dest, const char* src);

// Конвертация hex в байты
void hex_to_bytes(const char* hex, uint8_t* bytes);

// Загрузка словаря
std::vector<std::string> load_dictionary(const char* filename);

// Атака по словарю на GPU
// schedule(static) - статическое планирование (nvc++ GPU requirement)
// #pragma omp atomic - атомарная операция (nvc++ GPU requirement)
void dictionary_attack(uint8_t* target_hash, const char* dict_filename, size_t start = 0, size_t count = 0);

// Прямой перебор на GPU
// schedule(static) - статическое планирование (nvc++ GPU requirement)
// #pragma omp atomic - атомарная операция (nvc++ GPU requirement)
void brute_force_attack(uint8_t* target_hash, int min_length, int max_length);

// Проверка GPU
int check_gpu();

#endif

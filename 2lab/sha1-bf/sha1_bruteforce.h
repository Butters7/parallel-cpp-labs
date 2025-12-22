#ifndef SHA1_BRUTEFORCE_H
#define SHA1_BRUTEFORCE_H

#include <string.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#define MAX_PASSWORD_LEN 256
#define HASH_SIZE 20

void sha1(const char* message, size_t length, uint8_t* digest);
int hash_matches(uint8_t* hash1, uint8_t* hash2);
void print_hash(uint8_t* hash);
bool strcpy_compact(char* dest, const char* src, size_t dest_size);
bool hex_to_bytes(const char* hex, uint8_t* bytes);
std::vector<std::string> load_dictionary(const char* filename);

// Атака по словарю на GPU
// start - начальный индекс, count - количество (0 = все)
void dictionary_attack(uint8_t* target_hash, const char* dict_filename, size_t start = 0, size_t count = 0);

void brute_force_attack(uint8_t* target_hash, int min_length, int max_length);
int check_gpu();

#endif

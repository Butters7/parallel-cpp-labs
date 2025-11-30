#include "sha1_bruteforce.h"

// =============================================================================
// SHA1 функция
// =============================================================================
void sha1(const char* message, size_t length, uint8_t* digest) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    size_t original_length = length;
    size_t new_length = ((((length + 8) / 64) + 1) * 64);
    uint8_t* msg = new uint8_t[new_length];
    memcpy(msg, message, length);
    msg[length] = 0x80;

    for (size_t i = length + 1; i < new_length - 8; i++) {
        msg[i] = 0;
    }

    uint64_t bit_length = original_length * 8;
    for (int i = 0; i < 8; i++) {
        msg[new_length - 1 - i] = (bit_length >> (i * 8)) & 0xFF;
    }

    for (size_t i = 0; i < new_length; i += 64) {
        uint32_t w[80];

        for (int j = 0; j < 16; j++) {
            w[j] = (msg[i + j * 4] << 24) | (msg[i + j * 4 + 1] << 16) |
                   (msg[i + j * 4 + 2] << 8) | (msg[i + j * 4 + 3]);
        }

        for (int j = 16; j < 80; j++) {
            w[j] = w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16];
            w[j] = (w[j] << 1) | (w[j] >> 31);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;

        for (int j = 0; j < 80; j++) {
            uint32_t f, k;

            if (j < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (j < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (j < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k + w[j];
            e = d;
            d = c;
            c = (b << 30) | (b >> 2);
            b = a;
            a = temp;
        }

        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    delete[] msg;

    digest[0] = (h0 >> 24) & 0xFF; digest[1] = (h0 >> 16) & 0xFF;
    digest[2] = (h0 >> 8) & 0xFF; digest[3] = h0 & 0xFF;
    digest[4] = (h1 >> 24) & 0xFF; digest[5] = (h1 >> 16) & 0xFF;
    digest[6] = (h1 >> 8) & 0xFF; digest[7] = h1 & 0xFF;
    digest[8] = (h2 >> 24) & 0xFF; digest[9] = (h2 >> 16) & 0xFF;
    digest[10] = (h2 >> 8) & 0xFF; digest[11] = h2 & 0xFF;
    digest[12] = (h3 >> 24) & 0xFF; digest[13] = (h3 >> 16) & 0xFF;
    digest[14] = (h3 >> 8) & 0xFF; digest[15] = h3 & 0xFF;
    digest[16] = (h4 >> 24) & 0xFF; digest[17] = (h4 >> 16) & 0xFF;
    digest[18] = (h4 >> 8) & 0xFF; digest[19] = h4 & 0xFF;
}

int hash_matches(uint8_t* hash1, uint8_t* hash2) {
    return memcmp(hash1, hash2, 20) == 0;
}

void print_hash(uint8_t* hash) {
    for (int i = 0; i < 20; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

char* strcpy_compact(char* dest, const char* src) {
    char* ptr = dest;
    while ((*dest++ = *src++) != '\0');
    return ptr;
}

void hex_to_bytes(const char* hex, uint8_t* bytes) {
    for (int i = 0; i < 20; i++) {
        char byte_str[3] = {hex[i*2], hex[i*2+1], '\0'};
        bytes[i] = (uint8_t)strtol(byte_str, NULL, 16);
    }
}

// =============================================================================
// Загрузка словаря
// =============================================================================
std::vector<std::string> load_dictionary(const char* filename) {
    std::vector<std::string> dictionary;
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Ошибка: не удалось открыть файл %s\n", filename);
        return dictionary;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), file)) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') buffer[--len] = '\0';
        if (len > 0 && buffer[len-1] == '\r') buffer[--len] = '\0';
        if (strlen(buffer) > 0) dictionary.push_back(buffer);
    }
    fclose(file);
    return dictionary;
}

// =============================================================================
// Атака по словарю на GPU
// schedule(static) - статическое планирование (требование nvc++ для GPU)
// #pragma omp atomic - атомарная операция для безопасного обновления на GPU
// start - начальный индекс в словаре
// count - количество слов для обработки (0 = все)
// =============================================================================
void dictionary_attack(uint8_t* target_hash, const char* dict_filename, size_t start, size_t count) {
    printf("Загружаем словарь: %s\n", dict_filename);
    std::vector<std::string> dictionary = load_dictionary(dict_filename);

    if (dictionary.empty()) {
        printf("Словарь пуст\n");
        return;
    }

    printf("Загружено %zu слов\n", dictionary.size());

    // Обрезаем по start и count
    if (start >= dictionary.size()) {
        printf("Start индекс за пределами словаря\n");
        return;
    }
    size_t end = (count == 0) ? dictionary.size() : std::min(start + count, dictionary.size());
    size_t dict_size = end - start;

    printf("Обработка слов %zu - %zu (%zu слов)\n", start, end - 1, dict_size);

    int found = 0;
    size_t found_index = 0;

    char* dict_words = new char[dict_size * MAX_PASSWORD_LEN];
    size_t* word_lengths = new size_t[dict_size];

    for (size_t i = 0; i < dict_size; i++) {
        strcpy_compact(dict_words + i * MAX_PASSWORD_LEN, dictionary[start + i].c_str());
        word_lengths[i] = dictionary[start + i].length();
    }

    double start_time = omp_get_wtime();

    // OpenMP GPU offload со статическим планированием
    #pragma omp target teams distribute parallel for \
                schedule(static) \
                map(to: target_hash[0:20], dict_words[0:dict_size*MAX_PASSWORD_LEN], word_lengths[0:dict_size], dict_size) \
                map(tofrom: found, found_index)
    for (size_t i = 0; i < dict_size; i++) {
        // Early termination - прерывание при нахождении
        if (found) continue;

        const char* candidate = dict_words + i * MAX_PASSWORD_LEN;
        size_t length = word_lengths[i];

        uint8_t hash[20];
        sha1(candidate, length, hash);

        if (hash_matches(hash, target_hash)) {
            // Атомарная операция для безопасного обновления на GPU
            int old_found;
            #pragma omp atomic capture
            { old_found = found; found = 1; }
            if (old_found == 0) {
                found_index = i;
            }
        }
    }

    double elapsed = omp_get_wtime() - start_time;

    printf("Время: %.3f сек (%.0f hash/sec)\n", elapsed, dict_size / elapsed);

    if (found) {
        const char* found_password = dict_words + found_index * MAX_PASSWORD_LEN;
        printf("*** Пароль найден: %s ***\n", found_password);
        printf("Проверка SHA1: ");
        uint8_t check[20];
        sha1(found_password, strlen(found_password), check);
        print_hash(check);
    } else {
        printf("Пароль не найден в словаре\n");
    }

    delete[] dict_words;
    delete[] word_lengths;
}

// =============================================================================
// Прямой перебор на GPU
// schedule(static) - статическое планирование (требование nvc++ для GPU)
// #pragma omp atomic - атомарная операция для безопасного обновления на GPU
// Поддержка min/max длины пароля
// =============================================================================
void brute_force_attack(uint8_t* target_hash, int min_length, int max_length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    int charset_len = sizeof(charset) - 1;
    int found = 0;
    long long found_index = -1;
    int found_len = 0;

    printf("Перебор (длина %d-%d)...\n", min_length, max_length);

    double total_start = omp_get_wtime();
    long long total_attempts = 0;

    for (int len = min_length; len <= max_length && !found; len++) {
        long long max_i = 1;
        for (int j = 0; j < len; j++) max_i *= charset_len;

        printf("Длина %d: %lld комбинаций\n", len, max_i);

        double start = omp_get_wtime();

        // OpenMP GPU offload со статическим планированием
        #pragma omp target teams distribute parallel for \
                    schedule(static) \
                    map(to: target_hash[0:20], charset[0:charset_len+1], len, max_i, charset_len) \
                    map(tofrom: found, found_index)
        for (long long i = 0; i < max_i; i++) {
            // Early termination
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
                // Атомарная операция для безопасного обновления на GPU
                int old_found;
                #pragma omp atomic capture
                { old_found = found; found = 1; }
                if (old_found == 0) {
                    found_index = i;
                }
            }
        }

        double elapsed = omp_get_wtime() - start;
        total_attempts += max_i;
        printf("  %.3f сек (%.0f hash/sec)\n", elapsed, max_i / elapsed);

        if (found) found_len = len;
    }

    double total_elapsed = omp_get_wtime() - total_start;
    printf("Всего: %lld попыток за %.3f сек\n", total_attempts, total_elapsed);

    if (found && found_index >= 0) {
        char found_password[32];
        long long temp = found_index;
        for (int pos = 0; pos < found_len; pos++) {
            found_password[pos] = charset[temp % charset_len];
            temp /= charset_len;
        }
        found_password[found_len] = '\0';
        printf("*** Пароль найден: %s ***\n", found_password);
    } else {
        printf("Пароль не найден\n");
    }
}

// =============================================================================
// Проверка GPU
// =============================================================================
int check_gpu() {
    int on_gpu = 0;
    #pragma omp target map(from:on_gpu)
    {
        if (omp_is_initial_device() == 0) on_gpu = 1;
    }
    return on_gpu;
}

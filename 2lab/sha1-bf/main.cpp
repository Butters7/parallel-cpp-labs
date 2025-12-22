#include "sha1_bruteforce.h"

int main(int argc, char* argv[]) {
    if (check_gpu()) { printf("### GPU доступен ###\n\n"); }
    else { printf("### GPU недоступен ###\n\n"); }

    const char* target_hashes[] = {
        "4f43015fb43b7e308ec2b95e2b00134175cb4417",
        "95b28b679cfd57b33ec0d5507984806c5a6f7a89"
    };

    printf("========================================\n");
    printf("SHA1 Brute Force (OpenMP GPU)\n");
    printf("========================================\n\n");

    // Аргументы: ./sha1_bruteforce [словарь] [start] [count]
    const char* dict_file = (argc > 1) ? argv[1] : "rockyou.txt";
    size_t start = (argc > 2) ? atoll(argv[2]) : 0;
    size_t count = (argc > 3) ? atoll(argv[3]) : 0;

    for (int h = 0; h < 2; h++) {
        printf("--- Хеш #%d: %s ---\n", h+1, target_hashes[h]);
        uint8_t target[20];

        // ИСПРАВЛЕНО: проверяем результат hex_to_bytes
        if (!hex_to_bytes(target_hashes[h], target)) {
            printf("Ошибка: некорректный hex формат хеша\n");
            continue;
        }

        dictionary_attack(target, dict_file, start, count);

        if (count == 0) {
            printf("\nПрямой перебор:\n");
            brute_force_attack(target, 1, 6);
        }
        printf("\n");
    }
    return 0;
}

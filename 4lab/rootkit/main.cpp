#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_PROCESSES 10000
#define MAX_MODULES 500

typedef struct {
    unsigned int pid;
    char name[256];
    unsigned int parent_pid;
    bool hidden;
    bool hooked;
} ProcessInfo;

typedef struct {
    char name[256];
    unsigned long base_address;
    unsigned long size;
    bool suspicious;
} ModuleInfo;

void analyze_process_hiding(ProcessInfo* processes, int count, bool* results) {
    for (int idx = 0; idx < count; idx++) {
        bool is_hidden = false;
        
        // Проверка на скрытые процессы (PID reuse, parent-child inconsistency)
        if (processes[idx].pid == 0 || processes[idx].pid == 65535) {
            is_hidden = true;
        }
        
        // Проверка несоответствия родительского PID
        if (processes[idx].parent_pid == 0 && processes[idx].pid != 0) {
            is_hidden = true;
        }
        
        // Проверка подозрительных имен процессов
        const char* suspicious_processes[] = {
            "rootkit", "stealth", "hidden", "hook", "inject",
            "beast", "hxdef", "fu", "adore", "diamonds"
        };
        
        for (int i = 0; i < 10; i++) {
            if (strstr(processes[idx].name, suspicious_processes[i]) != NULL) {
                is_hidden = true;
                break;
            }
        }
        
        results[idx] = is_hidden;
        processes[idx].hidden = is_hidden;
    }
}

void analyze_module_integrity(ModuleInfo* modules, int count, bool* results) {
    for (int idx = 0; idx < count; idx++) {
        bool is_suspicious = false;
        
        // Проверка базовых адресов (необычные расположения)
        if (modules[idx].base_address < 0x400000 || modules[idx].base_address > 0x7FFFFFFF0000) {
            is_suspicious = true;
        }
        
        // Проверка размера (слишком маленькие или большие модули)
        if (modules[idx].size < 0x1000 || modules[idx].size > 0x1000000) {
            is_suspicious = true;
        }
        
        // Проверка имен (известные имена руткитов)
        const char* suspicious_names[] = {
            "rootkit", "stealth", "hidden", "hook", "inject",
            "beast", "hxdef", "fu", "adore", "diamonds",
            "driver", "sys", "hidden", "secret"
        };
        
        for (int i = 0; i < 14; i++) {
            if (strstr(modules[idx].name, suspicious_names[i]) != NULL) {
                is_suspicious = true;
                break;
            }
        }
        
        results[idx] = is_suspicious;
        modules[idx].suspicious = is_suspicious;
    }
}

void ssdt_hook_detector(unsigned long* ssdt_entries, unsigned long* original_ssdt, 
                       int count, bool* hooks_detected) {
    for (int idx = 0; idx < count; idx++) {
        // Сравнение текущих SSDT записей с оригинальными
        if (ssdt_entries[idx] != original_ssdt[idx]) {
            hooks_detected[idx] = true;
            printf("[DETAIL] SSDT индекс %d изменен: 0x%lX -> 0x%lX\n", 
                   idx, original_ssdt[idx], ssdt_entries[idx]);
        } else {
            hooks_detected[idx] = false;
        }
    }
}

void analyze_processes(ProcessInfo* processes, int count) {
    bool* results = (bool*)malloc(count * sizeof(bool));
    
    analyze_process_hiding(processes, count, results);
    
    printf("\n=== АНАЛИЗ СКРЫТЫХ ПРОЦЕССОВ ===\n");
    int hidden_count = 0;
    for (int i = 0; i < count; i++) {
        if (results[i]) {
            printf("[ВНИМАНИЕ] Подозрительный процесс: PID=%u, Name=%s, ParentPID=%u\n", 
                   processes[i].pid, processes[i].name, processes[i].parent_pid);
            hidden_count++;
        }
    }
    
    if (hidden_count == 0) {
        printf("[ИНФО] Скрытые процессы не обнаружены\n");
    } else {
        printf("[ИНФО] Найдено скрытых процессов: %d\n", hidden_count);
    }
    
    free(results);
}

void analyze_modules(ModuleInfo* modules, int count) {
    bool* results = (bool*)malloc(count * sizeof(bool));
    
    analyze_module_integrity(modules, count, results);
    
    printf("\n=== АНАЛИЗ ПОДОЗРИТЕЛЬНЫХ МОДУЛЕЙ ===\n");
    int suspicious_count = 0;
    for (int i = 0; i < count; i++) {
        if (results[i]) {
            printf("[ВНИМАНИЕ] Подозрительный модуль: %s (Base: 0x%lX, Size: 0x%lX)\n", 
                   modules[i].name, modules[i].base_address, modules[i].size);
            suspicious_count++;
        }
    }
    
    if (suspicious_count == 0) {
        printf("[ИНФО] Подозрительные модули не обнаружены\n");
    } else {
        printf("[ИНФО] Найдено подозрительных модулей: %d\n", suspicious_count);
    }
    
    free(results);
}

void detect_ssdt_hooks(unsigned long* current_ssdt, unsigned long* original_ssdt, int count) {
    bool* hooks_detected = (bool*)malloc(count * sizeof(bool));
    
    ssdt_hook_detector(current_ssdt, original_ssdt, count, hooks_detected);
    
    printf("\n=== ДЕТЕКЦИЯ SSDT ХУКОВ ===\n");
    bool hooks_found = false;
    int hook_count = 0;
    
    for (int i = 0; i < count; i++) {
        if (hooks_detected[i]) {
            printf("[ВНИМАНИЕ] Обнаружен SSDT хук в индексе %d\n", i);
            hooks_found = true;
            hook_count++;
        }
    }
    
    if (!hooks_found) {
        printf("[ИНФО] SSDT хуки не обнаружены\n");
    } else {
        printf("[ИНФО] Найдено SSDT хуков: %d\n", hook_count);
    }
    
    free(hooks_detected);
}

// Дополнительные проверки
void perform_integrity_checks(ProcessInfo* processes, int proc_count, 
                             ModuleInfo* modules, int mod_count) {
    printf("\n=== ПРОВЕРКА ЦЕЛОСТНОСТИ СИСТЕМЫ ===\n");
    
    // Проверка двойных процессов (одинаковые PID)
    printf("\n[ПРОВЕРКА] Поиск дубликатов PID...\n");
    for (int i = 0; i < proc_count; i++) {
        for (int j = i + 1; j < proc_count; j++) {
            if (processes[i].pid == processes[j].pid) {
                printf("[ВНИМАНИЕ] Обнаружены процессы с одинаковым PID %u: %s и %s\n",
                       processes[i].pid, processes[i].name, processes[j].name);
            }
        }
    }
    
    // Проверка несуществующих родительских процессов
    printf("\n[ПРОВЕРКА] Проверка родительских процессов...\n");
    for (int i = 0; i < proc_count; i++) {
        if (processes[i].parent_pid != 0) {
            bool parent_exists = false;
            for (int j = 0; j < proc_count; j++) {
                if (processes[j].pid == processes[i].parent_pid) {
                    parent_exists = true;
                    break;
                }
            }
            if (!parent_exists && processes[i].parent_pid != 0) {
                printf("[ВНИМАНИЕ] Процесс %s (PID: %u) имеет несуществующего родителя (PID: %u)\n",
                       processes[i].name, processes[i].pid, processes[i].parent_pid);
            }
        }
    }
}
void generate_test_data(ProcessInfo** processes, int* proc_count, 
                       ModuleInfo** modules, int* mod_count,
                       unsigned long** ssdt_current, unsigned long** ssdt_original, int* ssdt_count) {
    
    *proc_count = 6;
    *processes = (ProcessInfo*)malloc(*proc_count * sizeof(ProcessInfo));
    
    // Нормальные процессы
    strcpy((*processes)[0].name, "System");
    (*processes)[0].pid = 4;
    (*processes)[0].parent_pid = 0;
    
    strcpy((*processes)[1].name, "explorer.exe");
    (*processes)[1].pid = 1024;
    (*processes)[1].parent_pid = 512;
    
    strcpy((*processes)[2].name, "svchost.exe");
    (*processes)[2].pid = 1048;
    (*processes)[2].parent_pid = 512;
    
    // Подозрительные процессы
    strcpy((*processes)[3].name, "rootkit_driver.sys");
    (*processes)[3].pid = 1337;
    (*processes)[3].parent_pid = 0; // Подозрительный родительский PID
    
    strcpy((*processes)[4].name, "stealth_process.exe");
    (*processes)[4].pid = 65535; // Подозрительный PID
    (*processes)[4].parent_pid = 1024;
    
    strcpy((*processes)[5].name, "hidden_service");
    (*processes)[5].pid = 31337;
    (*processes)[5].parent_pid = 1; // Несуществующий родитель
    
    // Генерируем тестовые модули
    *mod_count = 5;
    *modules = (ModuleInfo*)malloc(*mod_count * sizeof(ModuleInfo));
    
    // Нормальные модули
    strcpy((*modules)[0].name, "ntoskrnl.exe");
    (*modules)[0].base_address = 0x00400000;
    (*modules)[0].size = 0x200000;
    
    strcpy((*modules)[1].name, "win32k.sys");
    (*modules)[1].base_address = 0x00800000;
    (*modules)[1].size = 0x300000;
    
    // Подозрительные модули
    strcpy((*modules)[2].name, "hook_driver.sys");
    (*modules)[2].base_address = 0x00001000; // Подозрительный адрес
    (*modules)[2].size = 0x500;
    
    strcpy((*modules)[3].name, "rootkit_module.dll");
    (*modules)[3].base_address = 0x800000000000; // Очень высокий адрес
    (*modules)[3].size = 0x100;
    
    strcpy((*modules)[4].name, "stealth_injector.sys");
    (*modules)[4].base_address = 0x00410000;
    (*modules)[4].size = 0x50; // Слишком маленький размер
    
    // Генерируем тестовые SSDT данные
    *ssdt_count = 15;
    *ssdt_current = (unsigned long*)malloc(*ssdt_count * sizeof(unsigned long));
    *ssdt_original = (unsigned long*)malloc(*ssdt_count * sizeof(unsigned long));
    
    // Инициализируем оригинальные и текущие SSDT
    for (int i = 0; i < *ssdt_count; i++) {
        (*ssdt_original)[i] = 0x1000 + i * 0x100;
        (*ssdt_current)[i] = 0x1000 + i * 0x100;
    }
    
    // Добавляем хуки для демонстрации
    (*ssdt_current)[3] = 0xDEADBEEF;  // Подмененный адрес
    (*ssdt_current)[7] = 0xCAFEBABE;  // Еще один подмененный адрес
    (*ssdt_current)[12] = 0x13371337; // Третий подмененный адрес
}

void print_system_summary(ProcessInfo* processes, int proc_count, 
                         ModuleInfo* modules, int mod_count) {
    printf("\n=== СВОДКА СИСТЕМЫ ===\n");
    printf("Обнаружено процессов: %d\n", proc_count);
    printf("Обнаружено модулей: %d\n", mod_count);
    
    printf("\nСписок процессов:\n");
    for (int i = 0; i < proc_count; i++) {
        printf("  PID: %5u, Parent: %5u, Name: %s\n", 
               processes[i].pid, processes[i].parent_pid, processes[i].name);
    }
    
    printf("\nСписок модулей:\n");
    for (int i = 0; i < mod_count; i++) {
        printf("  Module: %-25s Base: 0x%-12lX Size: 0x%-8lX\n",
               modules[i].name, modules[i].base_address, modules[i].size);
    }
}

int main() {
    printf("=== ROOTKIT DETECTOR ===\n");
    printf("Версия: 1.0 (Без CUDA)\n\n");
    
    ProcessInfo* processes;
    ModuleInfo* modules;
    unsigned long* ssdt_current, * ssdt_original;
    int proc_count, mod_count, ssdt_count;
    
    // Генерируем тестовые данные
    generate_test_data(&processes, &proc_count, &modules, &mod_count, 
                      &ssdt_current, &ssdt_original, &ssdt_count);
    
    // Выводим сводку системы
    print_system_summary(processes, proc_count, modules, mod_count);
    
    // Запускаем анализ
    analyze_processes(processes, proc_count);
    analyze_modules(modules, mod_count);
    detect_ssdt_hooks(ssdt_current, ssdt_original, ssdt_count);
    perform_integrity_checks(processes, proc_count, modules, mod_count);
    
    printf("\n=== АНАЛИЗ ЗАВЕРШЕН ===\n");
    
    // Очистка
    free(processes);
    free(modules);
    free(ssdt_current);
    free(ssdt_original);
    
    return 0;
}
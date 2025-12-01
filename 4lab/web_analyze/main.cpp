#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <stdio.h>

char* _strstr(const char* str, const char* pattern) {
    if (*pattern == '\0') {
        return (char*)str;
    }
    while (*str != '\0') {
        const char* s = str;
        const char* p = pattern;
        while (*s != '\0' && *p != '\0' && *s == *p) {
            s++;
            p++;
        }
        if (*p == '\0') {
            return (char*)str;
        }
        str++;
    }
    
    return NULL;
}

void findVulnerabilities(char *inputs, int *results, int n) {
    for(int idx = 0; idx < n; idx++) {
        char *str = &inputs[idx * 50];
        int vuln = 0;
        
        if(_strstr(str, "../")) vuln = 1;        // Path traversal
        if(_strstr(str, "<script>")) vuln = 2;   // XSS
        if(_strstr(str, "' OR '1'='1")) vuln = 3; // SQL injection
        
        results[idx] = vuln;
    }
}

int main() {
    const int n = 100;
    char *h_inputs;
    int *h_results;
    
    h_inputs = (char*)malloc(n * 50);
    h_results = (int*)malloc(n * sizeof(int));
    
    memset(h_inputs, 0, n * 50);
    memset(h_results, 0, n * sizeof(int));
    
    strcpy(&h_inputs[0 * 50], "normal input");
    strcpy(&h_inputs[1 * 50], "../../etc/passwd");
    strcpy(&h_inputs[2 * 50], "<script>alert('xss')</script>");
    
    // Вызываем CPU-версию функции поиска уязвимостей
    findVulnerabilities(h_inputs, h_results, n);
    
    const char* vuln_types[] = {"None", "Path traversal", "XSS", "SQL Injection"};
    for(int i = 0; i < 3; i++) {
        printf("Input %d: %s - %s\n", i, &h_inputs[i * 50], vuln_types[h_results[i]]);
    }
    
    free(h_inputs);
    free(h_results);
    return 0;
}

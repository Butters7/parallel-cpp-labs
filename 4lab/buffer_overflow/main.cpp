#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void checkBufferOverflow(char *inputs, int *overflow, int n, int max_len) {
    for(int idx = 0; idx < n; idx++) {
        char *str = &inputs[idx * 20];
        int len = 0;
        while(len < 20 && str[len] != '\0') len++;
        overflow[idx] = (len >= max_len) ? 1 : 0;
    }
}

int main() {
    const int n = 100;
    const int max_len = 15;
    char *h_inputs;
    int *h_overflow;
    
    h_inputs = (char*)malloc(n * 20);
    h_overflow = (int*)malloc(n * sizeof(int));
    
    memset(h_inputs, 0, n * 20);
    memset(h_overflow, 0, n * sizeof(int));
    
    for(int i = 0; i < n; i++) {
        if(i == 5) {
            strcpy(&h_inputs[i * 20], "very_long_string_here");
        } else {
            sprintf(&h_inputs[i * 20], "str%d", i);
        }
    }
    
    checkBufferOverflow(h_inputs, h_overflow, n, max_len);
    
    printf("Potential buffer overflows at indices: ");
    for(int i = 0; i < n; i++) {
        if(h_overflow[i] == 1) printf("%d ", i);
    }
    printf("\n");
    
    free(h_inputs);
    free(h_overflow);
    return 0;
}

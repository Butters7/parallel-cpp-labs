### Необходимо внести изменения
1. CUDA-зависимости (#include <cuda_runtime.h>)
2. Добавить директиву__global__ для функцию findVulnerabilities 
3. Добавить CUDA-specific вызовы (cudaMalloc, cudaMemcpy, cudaFree) 
4. Добавить kernel launch <<<...>>> вместо обычного вызова функции 
5. Добавить расчет индексов для CUDA потоков, используем обычный цикл
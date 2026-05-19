// gpu_batch/gpu_batch.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include "cgbn/cgbn.h"

typedef unsigned long long ull;

#define BITS 1024
#define TPI  32
#define TPB  128   // threads per block, must be multiple of TPI

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS>            mem_t;

#define CUDA_CHECK(x)                                      \
do {                                                       \
    cudaError_t err = (x);                                 \
    if(err != cudaSuccess){                                \
        printf("CUDA Error at %s:%d\n%s\n",                \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while(0)


struct Task {
    mem_t base;
    mem_t exp;
    mem_t mod;
};


// store a 64-bit value into the low two limbs of a mem_t
static void store64(mem_t *m, ull val) {
    memset(m, 0, sizeof(mem_t));
    m->_limbs[0] = (uint32_t)(val & 0xFFFFFFFF);
    m->_limbs[1] = (uint32_t)(val >> 32);
}


// read a 64-bit value back from the low two limbs of a mem_t
static ull load64(const mem_t *m) {
    return (ull)m->_limbs[0] | ((ull)m->_limbs[1] << 32);
}


__global__ void batch_kernel(
    Task  *tasks,
    mem_t *results,
    int    n
){
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;

    if(instance >= n) return;

    context_t ctx(cgbn_no_checks);
    env_t     env(ctx);

    env_t::cgbn_t base, exp, mod, result;

    cgbn_load(env, base, &tasks[instance].base);
    cgbn_load(env, exp,  &tasks[instance].exp);
    cgbn_load(env, mod,  &tasks[instance].mod);

    cgbn_modular_power(env, result, base, exp, mod);

    cgbn_store(env, &results[instance], result);
}


int main(){

    // load dataset
    FILE *f = fopen("../data/dataset_64bit.txt", "r");
    if(!f){ perror("Error opening dataset"); return 1; }

    std::vector<Task> h_tasks;
    h_tasks.reserve(100000);

    char line[256];

    while(fgets(line, sizeof(line), f)){
        if(line[0] == '#' || line[0] == '\n') continue;

        ull b, e, m;
        if(sscanf(line, "%llu %llu %llu", &b, &e, &m) != 3){
            printf("Skipping malformed line:\n%s", line);
            continue;
        }
        if(m == 0){ printf("Skipping modulus=0\n"); continue; }

        // cgbn_modular_power requires odd modulus
        if(m % 2 == 0){ printf("Skipping even modulus\n"); continue; }

        Task t;
        store64(&t.base, b);
        store64(&t.exp,  e);
        store64(&t.mod,  m);
        h_tasks.push_back(t);
    }

    fclose(f);

    int n = (int)h_tasks.size();
    printf("Loaded %d cases\n", n);
    if(n == 0){ printf("No valid data\n"); return 1; }


    // allocate GPU memory
    Task  *d_tasks;
    mem_t *d_results;

    CUDA_CHECK(cudaMalloc(&d_tasks,   n * sizeof(Task)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(mem_t)));


    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_tasks, h_tasks.data(),
        n * sizeof(Task), cudaMemcpyHostToDevice));


    // launch kernel
    int blocks = (n * TPI + TPB - 1) / TPB;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    batch_kernel<<<blocks, TPB>>>(d_tasks, d_results, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Batch time: %.4f ms\n", ms);


    // GPU -> CPU
    std::vector<mem_t> h_results(n);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
        n * sizeof(mem_t), cudaMemcpyDeviceToHost));


    // verify and write output
    FILE *seq = fopen("../data/sequential_results.txt", "r");
    if(!seq){ perror("Error opening sequential file"); return 1; }

    FILE *out = fopen("../data/gpubatch_results.txt", "w");
    if(!out){ perror("Error opening output file"); fclose(seq); return 1; }

    int errors = 0;

    for(int i = 0; i < n; i++){
        ull gpu_val  = load64(&h_results[i]);
        ull expected;

        if(fscanf(seq, "%llu", &expected) != 1){
            printf("Could not read expected result %d\n", i);
            break;
        }

        fprintf(out, "%llu\n", gpu_val);

        if(gpu_val != expected){
            printf("Mismatch %d\nGPU: %llu\nCPU: %llu\n", i, gpu_val, expected);
            errors++;
        }
    }

    fclose(seq);
    fclose(out);

    if(errors == 0)
        printf("All %d results correct!\n", n);
    else
        printf("Found %d mismatches\n", errors);


    // cleanup
    cudaFree(d_tasks);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

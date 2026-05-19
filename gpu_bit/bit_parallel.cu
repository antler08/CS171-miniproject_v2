// gpu_bit/bit_parallel.cu
//
// Approach C: Bit-Parallel Modular Exponentiation
//
// Phase 1 (parallel): one CGBN instance per bit of the exponent.
//   Instance i computes base^(2^i) mod n by squaring base i times.
//   All instances run fully independently -- no communication.
//
// Phase 2 (combine): one CGBN instance per task.
//   Multiplies together the Phase 1 results where the corresponding
//   exponent bit is 1.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cgbn/cgbn.h"

typedef unsigned long long ull;

#define BITS  1024   // supports up to 512-bit inputs (1024-bit intermediates)
#define TPI   32
#define TPB   128    // must be multiple of TPI
#define EBITS 64     // exponent bit width -- raise to 512 for larger datasets

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS>            mem_t;

#define CUDA_CHECK(x)                                          \
do {                                                           \
    cudaError_t err = (x);                                     \
    if(err != cudaSuccess){                                    \
        printf("CUDA Error %s:%d : %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(1);                                               \
    }                                                          \
} while(0)


struct Task {
    mem_t base;
    mem_t exp;
    mem_t mod;
};


static void store64(mem_t *m, ull val) {
    memset(m, 0, sizeof(mem_t));
    m->_limbs[0] = (uint32_t)(val & 0xFFFFFFFF);
    m->_limbs[1] = (uint32_t)(val >> 32);
}


static ull load64(const mem_t *m) {
    return (ull)m->_limbs[0] | ((ull)m->_limbs[1] << 32);
}


// Phase 1: one CGBN instance per (task, bit_position) pair.
// powers[task * EBITS + i] = base^(2^i) mod n
__global__ void phase1_kernel(
    Task  *tasks,
    mem_t *powers,
    int    n
){
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if(instance >= n * EBITS) return;

    int task_id = instance / EBITS;
    int bit_idx = instance % EBITS;

    context_t ctx(cgbn_no_checks);
    env_t     env(ctx);

    env_t::cgbn_t      base, mod, result;
    env_t::cgbn_wide_t wide;

    cgbn_load(env, base, &tasks[task_id].base);
    cgbn_load(env, mod,  &tasks[task_id].mod);

    // base^(2^bit_idx) = square base bit_idx times
    cgbn_set(env, result, base);

    for(int s = 0; s < bit_idx; s++){
        cgbn_mul_wide(env, wide, result, result);
        cgbn_rem_wide(env, result, wide, mod);
    }

    cgbn_store(env, &powers[instance], result);
}


// Phase 2: one CGBN instance per task.
// Accumulate product of powers[i] where bit i of exp is set.
__global__ void phase2_kernel(
    Task  *tasks,
    mem_t *powers,
    mem_t *results,
    int    n
){
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if(instance >= n) return;

    context_t ctx(cgbn_no_checks);
    env_t     env(ctx);

    env_t::cgbn_t      exp, mod, acc, power;
    env_t::cgbn_wide_t wide;

    cgbn_load(env, exp, &tasks[instance].exp);
    cgbn_load(env, mod, &tasks[instance].mod);
    cgbn_set_ui32(env, acc, 1);

    for(int i = 0; i < EBITS; i++){
        // all 32 threads in this instance read the same exp,
        // so no warp divergence here
        uint32_t bit = cgbn_extract_bits_ui32(env, exp, i, 1);
        if(bit){
            cgbn_load(env, power, &powers[instance * EBITS + i]);
            cgbn_mul_wide(env, wide, acc, power);
            cgbn_rem_wide(env, acc, wide, mod);
        }
    }

    cgbn_store(env, &results[instance], acc);
}


int main(){

    // sanity: 3^11 mod 17 = 7
    {
        Task  h;
        store64(&h.base, 3);
        store64(&h.exp,  11);
        store64(&h.mod,  17);

        Task  *d_task;
        mem_t *d_powers, *d_result;
        mem_t  result;

        CUDA_CHECK(cudaMalloc(&d_task,   sizeof(Task)));
        CUDA_CHECK(cudaMalloc(&d_powers, EBITS * sizeof(mem_t)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(mem_t)));
        CUDA_CHECK(cudaMemcpy(d_task, &h, sizeof(Task), cudaMemcpyHostToDevice));

        phase1_kernel<<<(EBITS * TPI + TPB - 1) / TPB, TPB>>>(d_task, d_powers, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        phase2_kernel<<<1, TPB>>>(d_task, d_powers, d_result, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(mem_t), cudaMemcpyDeviceToHost));

        ull val = load64(&result);
        printf("Sanity: 3^11 mod 17 = %llu (expected 7)\n", val);
        if(val != 7){ printf("Kernel bug\n"); return 1; }

        cudaFree(d_task);
        cudaFree(d_powers);
        cudaFree(d_result);
    }


    // open files
    FILE *fin     = fopen("../data/dataset_64bit.txt",      "r");
    FILE *seq     = fopen("../data/sequential_results.txt", "r");
    FILE *out     = fopen("../data/bitpar_results.txt",     "w");
    FILE *timelog = fopen("../data/runtime_results.txt",    "a");

    if(!fin || !seq || !out || !timelog){
        printf("File open failed\n");
        return 1;
    }


    // load tasks
    char line[256];
    int  count = 0;
    Task hostInput[100000];

    while(fgets(line, sizeof(line), fin)){
        if(line[0] == '#' || line[0] == '\n') continue;
        ull b, e, m;
        if(sscanf(line, "%llu %llu %llu", &b, &e, &m) != 3) continue;
        if(m == 0 || m % 2 == 0) continue;
        store64(&hostInput[count].base, b);
        store64(&hostInput[count].exp,  e);
        store64(&hostInput[count].mod,  m);
        count++;
    }

    fclose(fin);
    printf("Loaded %d tasks\n", count);


    // GPU memory
    Task  *d_tasks;
    mem_t *d_powers, *d_results;

    CUDA_CHECK(cudaMalloc(&d_tasks,   count * sizeof(Task)));
    CUDA_CHECK(cudaMalloc(&d_powers,  count * EBITS * sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc(&d_results, count * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_tasks, hostInput, count * sizeof(Task), cudaMemcpyHostToDevice));


    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // Phase 1
    int p1_instances = count * EBITS;
    int p1_blocks    = (p1_instances * TPI + TPB - 1) / TPB;
    phase1_kernel<<<p1_blocks, TPB>>>(d_tasks, d_powers, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase 2
    int p2_blocks = (count * TPI + TPB - 1) / TPB;
    phase2_kernel<<<p2_blocks, TPB>>>(d_tasks, d_powers, d_results, count);
    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);


    // copy back and verify
    mem_t *gpuResults = (mem_t*)malloc(count * sizeof(mem_t));
    CUDA_CHECK(cudaMemcpy(gpuResults, d_results, count * sizeof(mem_t), cudaMemcpyDeviceToHost));

    int errors = 0;

    for(int i = 0; i < count; i++){
        ull gpu_val = load64(&gpuResults[i]);
        ull expected;

        fprintf(out, "%llu\n", gpu_val);

        if(fscanf(seq, "%llu", &expected) != 1) break;

        if(gpu_val != expected){
            errors++;
            printf("Mismatch case %d: got %llu expected %llu\n", i, gpu_val, expected);
        }
    }

    printf("\n%d/%d correct | GPU %.4f ms (phase1 + phase2)\n",
        count - errors, count, gpu_ms);

    fprintf(timelog,
        "bit_parallel | cases=%d | correct=%d/%d | gpu_ms=%.4f\n",
        count, count - errors, count, gpu_ms);


    // cleanup
    fclose(seq);
    fclose(out);
    fclose(timelog);

    free(gpuResults);
    cudaFree(d_tasks);
    cudaFree(d_powers);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// gpu_bit/bit_parallel.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cgbn/cgbn.h"

typedef unsigned long long ull;

#define BITS 1024
#define TPI  32
#define TPB  128   // threads per block, must be multiple of TPI

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS>            mem_t;

#define CUDA_CHECK(x)                                     \
do{                                                       \
    cudaError_t err=(x);                                  \
    if(err!=cudaSuccess){                                 \
        printf("CUDA Error %s:%d : %s\n",                 \
               __FILE__,__LINE__,cudaGetErrorString(err));\
        exit(1);                                          \
    }                                                     \
}while(0)


struct ModExpTask {
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


__global__ void batched_modexp(
    ModExpTask *tasks,
    mem_t      *results,
    int         n
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

    // sanity: 3^11 mod 17 = 7
    {
        ModExpTask h;
        store64(&h.base, 3);
        store64(&h.exp,  11);
        store64(&h.mod,  17);

        ModExpTask *dIn;
        mem_t      *dOut;
        mem_t       result;

        CUDA_CHECK(cudaMalloc(&dIn,  sizeof(ModExpTask)));
        CUDA_CHECK(cudaMalloc(&dOut, sizeof(mem_t)));
        CUDA_CHECK(cudaMemcpy(dIn, &h, sizeof(ModExpTask), cudaMemcpyHostToDevice));

        batched_modexp<<<1, TPB>>>(dIn, dOut, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&result, dOut, sizeof(mem_t), cudaMemcpyDeviceToHost));

        ull val = load64(&result);
        printf("Sanity: %llu (expected 7)\n", val);

        if(val != 7){ printf("Kernel bug\n"); return 1; }

        cudaFree(dIn);
        cudaFree(dOut);
    }


    // open files
    FILE *fin     = fopen("../data/dataset_64bit.txt",       "r");
    FILE *seq     = fopen("../data/sequential_results.txt",  "r");
    FILE *out     = fopen("../data/warp_results.txt",        "w");
    FILE *timelog = fopen("../data/runtime_results.txt",     "a");

    if(!fin || !seq || !out || !timelog){
        printf("File open failed\n");
        return 1;
    }


    // count valid lines
    char line[256];
    int count = 0;

    while(fgets(line, sizeof(line), fin)){
        if(line[0] == '#' || line[0] == '\n') continue;
        ull b, e, m;
        if(sscanf(line, "%llu %llu %llu", &b, &e, &m) != 3) continue;
        if(m == 0 || m % 2 == 0) continue;
        count++;
    }

    rewind(fin);


    // load tasks
    ModExpTask *hostInput = (ModExpTask*)malloc(count * sizeof(ModExpTask));
    int idx = 0;

    while(fgets(line, sizeof(line), fin)){
        if(line[0] == '#' || line[0] == '\n') continue;

        ull b, e, m;
        if(sscanf(line, "%llu %llu %llu", &b, &e, &m) != 3) continue;
        if(m == 0 || m % 2 == 0) continue;

        store64(&hostInput[idx].base, b);
        store64(&hostInput[idx].exp,  e);
        store64(&hostInput[idx].mod,  m);
        idx++;
    }

    count = idx;


    // allocate GPU memory
    ModExpTask *dInput;
    mem_t      *dOutput;

    CUDA_CHECK(cudaMalloc(&dInput,  count * sizeof(ModExpTask)));
    CUDA_CHECK(cudaMalloc(&dOutput, count * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(dInput, hostInput, count * sizeof(ModExpTask), cudaMemcpyHostToDevice));


    // launch kernel
    int blocks = (count * TPI + TPB - 1) / TPB;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    batched_modexp<<<blocks, TPB>>>(dInput, dOutput, count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);


    // GPU -> CPU
    mem_t *gpuResults = (mem_t*)malloc(count * sizeof(mem_t));
    CUDA_CHECK(cudaMemcpy(gpuResults, dOutput, count * sizeof(mem_t), cudaMemcpyDeviceToHost));


    // verify and write output
    int errors = 0;

    for(int i = 0; i < count; i++){
        ull gpu_val  = load64(&gpuResults[i]);
        ull expected;

        fprintf(out, "%llu\n", gpu_val);

        if(fscanf(seq, "%llu", &expected) != 1) break;

        if(gpu_val != expected){
            errors++;
            printf("Mismatch case %d\n", i);
        }
    }

    printf("\n%d/%d correct | GPU %.4f ms\n", count - errors, count, gpu_ms);

    fprintf(timelog,
        "cgbn_warp | cases=%d | correct=%d/%d | gpu_ms=%.4f\n",
        count, count - errors, count, gpu_ms);


    // cleanup
    fclose(fin);
    fclose(seq);
    fclose(out);
    fclose(timelog);

    free(hostInput);
    free(gpuResults);

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

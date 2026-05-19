// gpu_batch/gpu_batch.cu
// Compile:
//   nvcc -O2 -o gpu_batch gpu_batch.cu \
//        -I./CGBN/include               \
//        -Xlinker -lgmp
//
// Usage:
//   ./gpu_batch input_file [output_file]
// Example:
//   ./gpu_batch ../data/dataset_1024bit.txt
//   ./gpu_batch ../data/dataset_1024bit.txt ../data/out/gpubatch_results.txt

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

#define BITS      1024
#define TPI       32
#define TPB       128          // threads per block, must be multiple of TPI
#define LIMBS     (BITS / 32)  // 32 limbs of uint32_t

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS>            mem_t;

#define CUDA_CHECK(x)                                          \
do {                                                           \
    cudaError_t err = (x);                                     \
    if (err != cudaSuccess) {                                  \
        printf("CUDA Error at %s:%d\n%s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while (0)


struct Task {
    mem_t base;
    mem_t exp;
    mem_t mod;
};


// ---------------------------------------------------------------------------
// mpz_t --> mem_t   (little-endian byte export -> 32-bit limbs)
// ---------------------------------------------------------------------------
static void mpz_to_mem(mem_t *m, const mpz_t z) {
    memset(m, 0, sizeof(mem_t));

    uint8_t buf[BITS / 8] = {0};
    size_t  count = 0;

    mpz_export(buf, &count, -1, 1, -1, 0, z);

    for (int i = 0; i < LIMBS; i++) {
        uint32_t limb = 0;
        for (int b = 0; b < 4; b++) {
            size_t byte_idx = (size_t)i * 4 + b;
            if (byte_idx < count)
                limb |= (uint32_t)buf[byte_idx] << (b * 8);
        }
        m->_limbs[i] = limb;
    }
}


// ---------------------------------------------------------------------------
// mem_t --> mpz_t   (32-bit limbs -> little-endian byte import)
// ---------------------------------------------------------------------------
static void mem_to_mpz(mpz_t z, const mem_t *m) {
    uint8_t buf[BITS / 8];

    for (int i = 0; i < LIMBS; i++) {
        uint32_t limb = m->_limbs[i];
        buf[i * 4 + 0] = (uint8_t)(limb         & 0xFF);
        buf[i * 4 + 1] = (uint8_t)((limb >>  8)  & 0xFF);
        buf[i * 4 + 2] = (uint8_t)((limb >> 16)  & 0xFF);
        buf[i * 4 + 3] = (uint8_t)((limb >> 24)  & 0xFF);
    }

    mpz_import(z, BITS / 8, -1, 1, -1, 0, buf);
}


// ---------------------------------------------------------------------------
// CGBN kernel
// ---------------------------------------------------------------------------
__global__ void batch_kernel(Task *tasks, mem_t *results, int n) {
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (instance >= n) return;

    context_t ctx(cgbn_no_checks);
    env_t     env(ctx);

    env_t::cgbn_t base, exp, mod, result;

    cgbn_load(env, base, &tasks[instance].base);
    cgbn_load(env, exp,  &tasks[instance].exp);
    cgbn_load(env, mod,  &tasks[instance].mod);

    cgbn_modular_power(env, result, base, exp, mod);

    cgbn_store(env, &results[instance], result);
}


int main(int argc, char *argv[]) {

    // -----------------------------------------------------------------------
    // Sanity check: 3^11 mod 17 = 7
    // -----------------------------------------------------------------------
    {
        mpz_t b, e, m, r;
        mpz_inits(b, e, m, r, NULL);
        mpz_set_ui(b, 3);
        mpz_set_ui(e, 11);
        mpz_set_ui(m, 17);

        Task   h_task;
        mem_t  h_result;
        Task  *d_task;
        mem_t *d_result;

        mpz_to_mem(&h_task.base, b);
        mpz_to_mem(&h_task.exp,  e);
        mpz_to_mem(&h_task.mod,  m);

        CUDA_CHECK(cudaMalloc(&d_task,   sizeof(Task)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(mem_t)));
        CUDA_CHECK(cudaMemcpy(d_task, &h_task, sizeof(Task), cudaMemcpyHostToDevice));

        batch_kernel<<<1, TPB>>>(d_task, d_result, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(mem_t), cudaMemcpyDeviceToHost));

        mem_to_mpz(r, &h_result);
        unsigned long val = mpz_get_ui(r);
        printf("Sanity: 3^11 mod 17 = %lu (expected 7)\n", val);

        if (val != 7) {
            fprintf(stderr, "Sanity check FAILED\n");
            return 1;
        }

        cudaFree(d_task);
        cudaFree(d_result);
        mpz_clears(b, e, m, r, NULL);
    }

    // -----------------------------------------------------------------------
    // Arguments
    // -----------------------------------------------------------------------
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "    %s input_file [output_file]\n\n"
            "Example:\n"
            "    %s ../data/dataset_1024bit.txt\n"
            "    %s ../data/dataset_1024bit.txt ../data/out/gpubatch_results.txt\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *inputFile  = argv[1];
    const char *outputFile = (argc >= 3) ? argv[2]
                                         : "../data/out/gpubatch_results.txt";

    // -----------------------------------------------------------------------
    // Open input file
    // -----------------------------------------------------------------------
    FILE *fin = fopen(inputFile, "r");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    // -----------------------------------------------------------------------
    // Parse dataset into tasks
    // -----------------------------------------------------------------------
    std::vector<Task> h_tasks;
    h_tasks.reserve(100000);

    mpz_t base, exp, mod;
    mpz_inits(base, exp, mod, NULL);

    char line[4096];
    int  skipped = 0;

    while (fgets(line, sizeof(line), fin)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        line[strcspn(line, "\r\n")] = '\0';

        char *tok_b = strtok(line, " \t");
        char *tok_e = strtok(NULL, " \t");
        char *tok_m = strtok(NULL, " \t");

        if (!tok_b || !tok_e || !tok_m) {
            fprintf(stderr, "Skipping malformed line (need 3 tokens)\n");
            skipped++;
            continue;
        }

        if (mpz_set_str(base, tok_b, 0) != 0 ||
            mpz_set_str(exp,  tok_e, 0) != 0 ||
            mpz_set_str(mod,  tok_m, 0) != 0) {
            fprintf(stderr, "Skipping unparseable line\n");
            skipped++;
            continue;
        }

        if (mpz_sgn(mod) == 0) {
            fprintf(stderr, "Skipping modulus=0\n");
            skipped++;
            continue;
        }

        if (mpz_even_p(mod)) {
            fprintf(stderr, "Skipping even modulus\n");
            skipped++;
            continue;
        }

        Task t;
        mpz_to_mem(&t.base, base);
        mpz_to_mem(&t.exp,  exp);
        mpz_to_mem(&t.mod,  mod);
        h_tasks.push_back(t);
    }

    fclose(fin);
    mpz_clears(base, exp, mod, NULL);

    int n = (int)h_tasks.size();
    printf("Input file  : %s\n", inputFile);
    printf("Output file : %s\n", outputFile);
    printf("Processed   : %d cases (%d skipped)\n", n, skipped);

    if (n == 0) { printf("No valid data\n"); return 1; }

    // -----------------------------------------------------------------------
    // GPU memory
    // -----------------------------------------------------------------------
    Task  *d_tasks;
    mem_t *d_results;

    CUDA_CHECK(cudaMalloc(&d_tasks,   n * sizeof(Task)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(mem_t)));
    CUDA_CHECK(cudaMemcpy(d_tasks, h_tasks.data(),
        n * sizeof(Task), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Launch kernel
    // -----------------------------------------------------------------------
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

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    printf("Total time  : %.4f ms\n", gpu_ms);

    // -----------------------------------------------------------------------
    // GPU -> CPU
    // -----------------------------------------------------------------------
    std::vector<mem_t> h_results(n);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
        n * sizeof(mem_t), cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Write output + verify against sequential results if present
    // -----------------------------------------------------------------------
    FILE *fout = fopen(outputFile, "w");
    if (!fout) {
        perror("Error opening output file");
        return 1;
    }

    // sequential results are optional — skip verification if not found
    FILE *seq = fopen("../data/out/sequential_results.txt", "r");

    mpz_t gpu_val, expected;
    mpz_inits(gpu_val, expected, NULL);

    int errors   = 0;
    int verified = 0;

    for (int i = 0; i < n; i++) {
        mem_to_mpz(gpu_val, &h_results[i]);

        char *res_str = mpz_get_str(NULL, 10, gpu_val);
        fprintf(fout, "%s\n", res_str);
        free(res_str);

        if (seq) {
            if (mpz_inp_str(expected, seq, 10) == 0) {
                printf("Sequential file shorter than dataset — stopping verification at %d\n", i);
                fclose(seq);
                seq = NULL;
            } else {
                verified++;
                if (mpz_cmp(gpu_val, expected) != 0) {
                    errors++;
                    char *gs = mpz_get_str(NULL, 10, gpu_val);
                    char *es = mpz_get_str(NULL, 10, expected);
                    printf("Mismatch case %d:\n  GPU: %s\n  CPU: %s\n", i, gs, es);
                    free(gs);
                    free(es);
                }
            }
        }
    }

    if (verified > 0)
        printf("%d/%d correct\n", verified - errors, verified);
    else
        printf("(No sequential_results.txt found — skipping verification)\n");

    // -----------------------------------------------------------------------
    // Timelog
    // -----------------------------------------------------------------------
    FILE *timelog = fopen("../data/runtime/runtime_results.txt", "a");
    if (timelog) {
        fprintf(timelog,
            "cgbn_batch | input=%s | cases=%d | correct=%d/%d | gpu_ms=%.4f\n",
            inputFile, n, verified - errors, verified, gpu_ms);
        fclose(timelog);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    fclose(fout);
    if (seq) fclose(seq);

    mpz_clears(gpu_val, expected, NULL);

    cudaFree(d_tasks);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

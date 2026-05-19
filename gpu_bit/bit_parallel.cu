// gpu_bit/bit_parallel.cu
//
// Approach C — Bit-Parallel Modular Exponentiation
//
// Phase 1 (GPU): One CGBN instance per bit of the exponent.
//   Thread i computes base^(2^i) mod n independently — zero inter-thread
//   communication.  All BITS threads run simultaneously.
//
// Phase 2 (CPU/GMP): Collect the partial results where the corresponding
//   exponent bit is 1, then multiply them together mod n.
//   This is the sequential bottleneck described by Amdahl's Law.
//
// Compile:
//   nvcc -O2 -o bit_parallel bit_parallel.cu \
        -I./CGBN/include                      \
        -Xlinker -lgmp
//
// Usage:
//   ./bit_parallel input_file [output_file]
// Example:
//   ./bit_parallel ../data/dataset_1024bit.txt
//   ./bit_parallel ../data/dataset_1024bit.txt ../data/out/bitparallel_results.txt

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <gmp.h>
#include "cgbn/cgbn.h"

#define BITS   1024
#define TPI    32
#define TPB    128          // threads per block, must be multiple of TPI
#define LIMBS  (BITS / 32)  // 32 × uint32_t limbs per value

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS>            mem_t;

#define CUDA_CHECK(x)                                           \
do {                                                            \
    cudaError_t _e = (x);                                       \
    if (_e != cudaSuccess) {                                    \
        printf("CUDA Error at %s:%d\n%s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e));        \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while (0)


// ---------------------------------------------------------------------------
// mpz_t <--> mem_t   (portable via little-endian byte array)
// ---------------------------------------------------------------------------
static void mpz_to_mem(mem_t *m, const mpz_t z) {
    memset(m, 0, sizeof(mem_t));
    uint8_t buf[BITS / 8] = {0};
    size_t  count = 0;
    mpz_export(buf, &count, -1, 1, -1, 0, z);          // little-endian bytes
    for (int i = 0; i < LIMBS; i++) {
        uint32_t limb = 0;
        for (int b = 0; b < 4; b++) {
            size_t idx = (size_t)i * 4 + b;
            if (idx < count) limb |= (uint32_t)buf[idx] << (b * 8);
        }
        m->_limbs[i] = limb;
    }
}

static void mem_to_mpz(mpz_t z, const mem_t *m) {
    uint8_t buf[BITS / 8];
    for (int i = 0; i < LIMBS; i++) {
        uint32_t limb = m->_limbs[i];
        buf[i*4+0] = (uint8_t)(limb        & 0xFF);
        buf[i*4+1] = (uint8_t)((limb >>  8) & 0xFF);
        buf[i*4+2] = (uint8_t)((limb >> 16) & 0xFF);
        buf[i*4+3] = (uint8_t)((limb >> 24) & 0xFF);
    }
    mpz_import(z, BITS / 8, -1, 1, -1, 0, buf);        // little-endian bytes
}


// ---------------------------------------------------------------------------
// Phase 1 kernel
//
// Layout: one CGBN instance per (problem, bit) pair.
//   global_instance = prob_idx * BITS + bit_idx
//
// Each instance i independently computes:
//   partials[prob_idx * BITS + bit_idx] = base^(2^bit_idx) mod n
//
// No inter-thread communication whatsoever.
// ---------------------------------------------------------------------------
__global__ void phase1_kernel(
    const mem_t * __restrict__ d_bases,     // [n_problems]
    const mem_t * __restrict__ d_mods,      // [n_problems]
          mem_t * __restrict__ d_partials,  // [n_problems * BITS]
    int n_problems
){
    int instance  = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    int total     = n_problems * BITS;
    if (instance >= total) return;

    int prob_idx  = instance / BITS;
    int bit_idx   = instance % BITS;

    context_t ctx(cgbn_no_checks);
    env_t     env(ctx);

    env_t::cgbn_t base, mod, exponent, result;

    cgbn_load(env, base, &d_bases[prob_idx]);
    cgbn_load(env, mod,  &d_mods[prob_idx]);

    // Build exponent = 2^bit_idx  (shift 1 left by bit_idx positions)
    cgbn_set_ui32(env, exponent, 1);
    cgbn_shift_left(env, exponent, exponent, bit_idx);

    // Compute base^(2^bit_idx) mod n
    cgbn_modular_power(env, result, base, exponent, mod);

    cgbn_store(env, &d_partials[instance], result);
}


// ---------------------------------------------------------------------------
// Phase 2 (CPU + GMP)
//
// For one problem: multiply together all partial[i] where bit i of exp is 1.
// This is the sequential combine step limited by Amdahl's Law.
// ---------------------------------------------------------------------------
static void phase2_combine(
    mpz_t        result_out,
    const mem_t *partials,      // BITS partial results for this problem
    const mpz_t  exp,
    const mpz_t  mod
){
    mpz_t acc, term;
    mpz_inits(acc, term, NULL);
    mpz_set_ui(acc, 1);

    for (int i = 0; i < BITS; i++) {
        if (mpz_tstbit(exp, i)) {                      // bit i of exp is set
            mem_to_mpz(term, &partials[i]);
            mpz_mul(acc, acc, term);
            mpz_mod(acc, acc, mod);
        }
    }

    mpz_set(result_out, acc);
    mpz_clears(acc, term, NULL);
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {

    // -----------------------------------------------------------------------
    // Sanity check: 3^11 mod 17 = 7
    // Mirrors the worked example in the pitch deck.
    // -----------------------------------------------------------------------
    {
        mpz_t b, e, m, r;
        mpz_inits(b, e, m, r, NULL);
        mpz_set_ui(b, 3);
        mpz_set_ui(e, 11);
        mpz_set_ui(m, 17);

        // Phase 1: launch BITS instances for one problem
        mem_t h_base, h_mod;
        mpz_to_mem(&h_base, b);
        mpz_to_mem(&h_mod,  m);

        mem_t *d_bases, *d_mods, *d_partials;
        CUDA_CHECK(cudaMalloc(&d_bases,    sizeof(mem_t)));
        CUDA_CHECK(cudaMalloc(&d_mods,     sizeof(mem_t)));
        CUDA_CHECK(cudaMalloc(&d_partials, BITS * sizeof(mem_t)));

        CUDA_CHECK(cudaMemcpy(d_bases, &h_base, sizeof(mem_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mods,  &h_mod,  sizeof(mem_t), cudaMemcpyHostToDevice));

        int blocks = (1 * BITS * TPI + TPB - 1) / TPB;
        phase1_kernel<<<blocks, TPB>>>(d_bases, d_mods, d_partials, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<mem_t> h_partials(BITS);
        CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials,
            BITS * sizeof(mem_t), cudaMemcpyDeviceToHost));

        // Phase 2
        phase2_combine(r, h_partials.data(), e, m);

        unsigned long val = mpz_get_ui(r);
        printf("Sanity: 3^11 mod 17 = %lu (expected 7)\n", val);

        if (val != 7) {
            fprintf(stderr, "Sanity check FAILED\n");
            return 1;
        }

        cudaFree(d_bases);
        cudaFree(d_mods);
        cudaFree(d_partials);
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
            "    %s ../data/dataset_1024bit.txt ../data/out/bitparallel_results.txt\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *inputFile  = argv[1];
    const char *outputFile = (argc >= 3) ? argv[2]
                                         : "../data/out/bitparallel_results.txt";

    // -----------------------------------------------------------------------
    // Open input file
    // -----------------------------------------------------------------------
    FILE *fin = fopen(inputFile, "r");
    if (!fin) { perror("Error opening input file"); return 1; }

    // -----------------------------------------------------------------------
    // Parse dataset
    // -----------------------------------------------------------------------
    struct Problem { mpz_t base, exp, mod; };

    std::vector<mem_t>  h_bases, h_mods;
    std::vector<Problem> problems;   // keep exp + mod for Phase 2

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
            skipped++; continue;
        }

        if (mpz_set_str(base, tok_b, 0) != 0 ||
            mpz_set_str(exp,  tok_e, 0) != 0 ||
            mpz_set_str(mod,  tok_m, 0) != 0) {
            fprintf(stderr, "Skipping unparseable line\n");
            skipped++; continue;
        }

        if (mpz_sgn(mod) == 0) {
            fprintf(stderr, "Skipping modulus=0\n");
            skipped++; continue;
        }

        if (mpz_even_p(mod)) {
            fprintf(stderr, "Skipping even modulus (cgbn_modular_power requires odd)\n");
            skipped++; continue;
        }

        // store base and mod for Phase 1 (GPU)
        mem_t mb, mm;
        mpz_to_mem(&mb, base);
        mpz_to_mem(&mm, mod);
        h_bases.push_back(mb);
        h_mods.push_back(mm);

        // store exp and mod for Phase 2 (CPU)
        Problem p;
        mpz_init_set(p.base, base);
        mpz_init_set(p.exp,  exp);
        mpz_init_set(p.mod,  mod);
        problems.push_back(p);
    }

    fclose(fin);
    mpz_clears(base, exp, mod, NULL);

    int n = (int)problems.size();
    printf("Input file  : %s\n", inputFile);
    printf("Output file : %s\n", outputFile);
    printf("Processed   : %d cases (%d skipped)\n", n, skipped);

    if (n == 0) { printf("No valid data\n"); return 1; }

    // -----------------------------------------------------------------------
    // GPU memory  (Phase 1 needs n*BITS partial result slots)
    // -----------------------------------------------------------------------
    mem_t *d_bases, *d_mods, *d_partials;

    CUDA_CHECK(cudaMalloc(&d_bases,    n * sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc(&d_mods,     n * sizeof(mem_t)));
    CUDA_CHECK(cudaMalloc(&d_partials, (size_t)n * BITS * sizeof(mem_t)));

    CUDA_CHECK(cudaMemcpy(d_bases, h_bases.data(), n * sizeof(mem_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mods,  h_mods.data(),  n * sizeof(mem_t), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Phase 1 — GPU
    // One CGBN instance per (problem × bit).  No inter-thread communication.
    // -----------------------------------------------------------------------
    int total_instances = n * BITS;
    int blocks          = (total_instances * TPI + TPB - 1) / TPB;

    cudaEvent_t p1_start, p1_stop;
    cudaEventCreate(&p1_start);
    cudaEventCreate(&p1_stop);
    cudaEventRecord(p1_start);

    phase1_kernel<<<blocks, TPB>>>(d_bases, d_mods, d_partials, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(p1_stop);
    cudaEventSynchronize(p1_stop);

    float phase1_ms;
    cudaEventElapsedTime(&phase1_ms, p1_start, p1_stop);
    printf("Phase 1 (GPU parallel)  : %.4f ms\n", phase1_ms);

    // -----------------------------------------------------------------------
    // GPU -> CPU
    // -----------------------------------------------------------------------
    std::vector<mem_t> h_partials((size_t)n * BITS);
    CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials,
        (size_t)n * BITS * sizeof(mem_t), cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Phase 2 — CPU (sequential combine, Amdahl bottleneck)
    // -----------------------------------------------------------------------
    struct timespec p2_ts, p2_te;
    clock_gettime(CLOCK_MONOTONIC, &p2_ts);

    std::vector<mpz_t> h_results(n);
    for (int i = 0; i < n; i++) {
        mpz_init(h_results[i]);
        phase2_combine(h_results[i],
                       &h_partials[(size_t)i * BITS],
                       problems[i].exp,
                       problems[i].mod);
    }

    clock_gettime(CLOCK_MONOTONIC, &p2_te);
    double phase2_ms = (p2_te.tv_sec  - p2_ts.tv_sec)  * 1000.0
                     + (p2_te.tv_nsec - p2_ts.tv_nsec) / 1e6;
    printf("Phase 2 (CPU combine)   : %.4f ms\n", phase2_ms);
    printf("Total time              : %.4f ms\n", phase1_ms + (float)phase2_ms);

    // -----------------------------------------------------------------------
    // Write output + verify against sequential results if present
    // -----------------------------------------------------------------------
    FILE *fout = fopen(outputFile, "w");
    if (!fout) { perror("Error opening output file"); return 1; }

    FILE *seq = fopen("../data/out/sequential_results.txt", "r");

    mpz_t expected;
    mpz_init(expected);

    int errors   = 0;
    int verified = 0;

    for (int i = 0; i < n; i++) {
        char *res_str = mpz_get_str(NULL, 10, h_results[i]);
        fprintf(fout, "%s\n", res_str);
        free(res_str);

        if (seq) {
            if (mpz_inp_str(expected, seq, 10) == 0) {
                printf("Sequential file shorter than dataset — stopping at %d\n", i);
                fclose(seq);
                seq = NULL;
            } else {
                verified++;
                if (mpz_cmp(h_results[i], expected) != 0) {
                    errors++;
                    char *gs = mpz_get_str(NULL, 10, h_results[i]);
                    char *es = mpz_get_str(NULL, 10, expected);
                    printf("Mismatch case %d:\n  GPU: %s\n  CPU: %s\n", i, gs, es);
                    free(gs); free(es);
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
            "bit_parallel | input=%s | cases=%d | correct=%d/%d"
            " | phase1_ms=%.4f | phase2_ms=%.4f | total_ms=%.4f\n",
            inputFile, n, verified - errors, verified,
            phase1_ms, (float)phase2_ms, phase1_ms + (float)phase2_ms);
        fclose(timelog);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    fclose(fout);
    if (seq) fclose(seq);

    mpz_clear(expected);
    for (int i = 0; i < n; i++) {
        mpz_clears(problems[i].base, problems[i].exp, problems[i].mod, NULL);
        mpz_clear(h_results[i]);
    }

    cudaFree(d_bases);
    cudaFree(d_mods);
    cudaFree(d_partials);
    cudaEventDestroy(p1_start);
    cudaEventDestroy(p1_stop);

    return 0;
}

// sequential/sequential.c
// Supports arbitrary-precision (up to 1024-bit) modular exponentiation via GMP.
// Compile: gcc -O2 -o sequential sequential.c -lgmp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gmp.h>

// ---------------------------------------------------------------------------
// mod_exp_gmp: result = base^exp mod m   (all values as hex or decimal strings)
// ---------------------------------------------------------------------------
static void mod_exp_gmp(mpz_t result, const mpz_t base, const mpz_t exp, const mpz_t mod) {
    mpz_powm(result, base, exp, mod);
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

        mod_exp_gmp(r, b, e, m);

        unsigned long val = mpz_get_ui(r);
        printf("3^11 mod 17 = %lu (expected 7)\n", val);

        if (val != 7) {
            fprintf(stderr, "BUG: mod_exp_gmp failed sanity check\n");
            mpz_clears(b, e, m, r, NULL);
            return 1;
        }

        mpz_clears(b, e, m, r, NULL);
    }

    // -----------------------------------------------------------------------
    // Check arguments
    // -----------------------------------------------------------------------
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "    %s input_file [output_file]\n\n"
            "Example:\n"
            "    %s ../data/dataset_1024bit.txt\n"
            "    %s ../data/dataset_1024bit.txt results.txt\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *inputFile  = argv[1];
    const char *outputFile = (argc >= 3) ? argv[2]
                                         : "../data/out/sequential_results.txt";

    // -----------------------------------------------------------------------
    // Open files
    // -----------------------------------------------------------------------
    FILE *fin = fopen(inputFile, "r");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    FILE *fout = fopen(outputFile, "w");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Process lines
    //
    // Supported input formats (one triplet per line):
    //   decimal:     <base_dec> <exp_dec> <mod_dec>
    //   hex:         0x<base> 0x<exp> 0x<mod>
    //   mixed:       any combination of the two
    //
    // Lines starting with '#' or blank lines are skipped.
    // Results are written as decimal strings, one per line.
    // -----------------------------------------------------------------------
    mpz_t base, exp, mod, result;
    mpz_inits(base, exp, mod, result, NULL);

    char line[4096];   // wide enough for three 1024-bit hex numbers
    int  count  = 0;
    int  skipped = 0;

    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    while (fgets(line, sizeof(line), fin)) {

        // skip comments and blank lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        // strip trailing newline so strtok works cleanly
        line[strcspn(line, "\r\n")] = '\0';

        // tokenise on whitespace
        char *tok_b = strtok(line,  " \t");
        char *tok_e = strtok(NULL,  " \t");
        char *tok_m = strtok(NULL,  " \t");

        if (!tok_b || !tok_e || !tok_m) {
            fprintf(stderr, "Skipping malformed line (need 3 tokens)\n");
            skipped++;
            continue;
        }

        // parse each token: leading "0x"/"0X" → base 16, else base 10
        int ok = 1;
        ok &= (mpz_set_str(base, tok_b, 0) == 0);
        ok &= (mpz_set_str(exp,  tok_e, 0) == 0);
        ok &= (mpz_set_str(mod,  tok_m, 0) == 0);

        if (!ok) {
            fprintf(stderr, "Skipping unparseable line\n");
            skipped++;
            continue;
        }

        if (mpz_sgn(mod) == 0) {
            fprintf(stderr, "Skipping modulus = 0\n");
            skipped++;
            continue;
        }

        mod_exp_gmp(result, base, exp, mod);

        // write result as decimal string
        char *res_str = mpz_get_str(NULL, 10, result);
        fprintf(fout, "%s\n", res_str);
        free(res_str);

        count++;
    }

    clock_gettime(CLOCK_MONOTONIC, &te);

    double ms = (te.tv_sec  - ts.tv_sec)  * 1000.0
              + (te.tv_nsec - ts.tv_nsec) / 1e6;

    printf("Input  file : %s\n",  inputFile);
    printf("Output file : %s\n",  outputFile);
    printf("Processed   : %d cases\n", count);
    if (skipped)
        printf("Skipped     : %d lines\n", skipped);
    printf("Total time  : %.4f ms\n", ms);

    mpz_clears(base, exp, mod, result, NULL);
    fclose(fin);
    fclose(fout);

    return 0;
}
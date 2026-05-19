// sequential/sequential.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned long long ull;
typedef __uint128_t u128;

ull mod_exp(ull base, ull exp, ull mod) {
    if(mod == 0){
        fprintf(stderr, "Error: modulus cannot be 0\n");
        exit(1);
    }

    ull result = 1;
    base %= mod;

    while(exp > 0){
        if(exp & 1)
            result = ((u128)result * base) % mod;

        exp >>= 1;
        base = ((u128)base * base) % mod;
    }

    return result;
}


int main(int argc, char *argv[]) {

    // sanity: 3^11 mod 17 = 7
    ull test = mod_exp(3,11,17);

    printf(
        "3^11 mod 17 = %llu (expected 7)\n",
        test
    );

    if(test != 7){
        fprintf(stderr,
                "BUG: mod_exp failed sanity check\n");
        return 1;
    }


    // --------------------------------------------
    // Check arguments
    // --------------------------------------------

    if(argc < 2){
        fprintf(stderr,
            "Usage:\n"
            "    %s input_file [output_file]\n\n"
            "Example:\n"
            "    %s ../data/dataset_64bit.txt\n"
            "    %s ../data/dataset_64bit.txt results.txt\n",
            argv[0], argv[0], argv[0]);

        return 1;
    }

    char *inputFile = argv[1];

    // optional output argument
    char *outputFile =
        (argc >= 3)
        ? argv[2]
        : "../data/out/sequential_results.txt";


    // --------------------------------------------
    // Open files
    // --------------------------------------------

    FILE *fin = fopen(inputFile, "r");

    if(!fin){
        perror("Error opening input file");
        return 1;
    }

    FILE *fout = fopen(outputFile, "w");

    if(!fout){
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }


    char line[256];
    ull b,e,m;

    int count = 0;

    struct timespec ts,te;

    clock_gettime(CLOCK_MONOTONIC,&ts);


    while(fgets(line,sizeof(line),fin)){

        if(line[0]=='#' || line[0]=='\n')
            continue;

        if(sscanf(
            line,
            "%llu %llu %llu",
            &b,
            &e,
            &m
        ) != 3){

            fprintf(
                stderr,
                "Skipping malformed line: %s",
                line
            );

            continue;
        }

        if(m==0){
            fprintf(
                stderr,
                "Skipping modulus=0\n"
            );
            continue;
        }

        ull result = mod_exp(b,e,m);

        fprintf(
            fout,
            "%llu\n",
            result
        );

        count++;
    }


    clock_gettime(CLOCK_MONOTONIC,&te);


    double ms =
        (te.tv_sec-ts.tv_sec)*1000.0 +
        (te.tv_nsec-ts.tv_nsec)/1000000.0;


    printf("Input file: %s\n",inputFile);
    printf("Output file: %s\n",outputFile);
    printf("Processed %d cases\n",count);
    printf("Total time: %.4f ms\n",ms);

    fclose(fin);
    fclose(fout);

    return 0;
}
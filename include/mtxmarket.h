#ifndef MATMARKET_H
#define MATMARKET_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

// Parse the matrix market .mat file and store the results in a dense matrix
bool parse_alloc_mat(const char *filename, int *nrows, int *ncols, double **A) {
    const static size_t buffer_size = 2048;
    FILE *mat = fopen(filename, "r");
    if (!mat) { 
        fprintf(stderr, "No file named %s", filename);
        return false;
    }
    char line[buffer_size];
    while(fgets(line, buffer_size, mat) && !strncmp(line, "%", 1)) { continue; } // Skip header
    // Dimensions
    char *tok = strtok(line, " ");
    *nrows = atoi(tok);
    tok = strtok(NULL, " ");
    *ncols = atoi(tok);
    if (*ncols == 0 || *nrows == 0) { 
        fprintf(stderr, "Invalid dims\n");
        return false;
    }
    // Allocate
    if (!*A) { 
        *A = calloc(*nrows * *ncols, sizeof(double)); 
        if (!*A) {
            fprintf(stderr, "calloc failure\n");
            return false;
        }
    }
    else { 
        double *tmp = realloc(A, *nrows * *ncols * sizeof(double)); 
        if (!tmp) {
            fprintf(stderr, "realloc failure\n");
            free(*A);
            return false;
        }
        memset(tmp, 0., *nrows * *ncols * sizeof(double));
        *A = tmp;
    }
    // Store mat
    while(fgets(line, buffer_size, mat)) {
        tok = strtok(line, " "); 
        int i = atoi(tok) - 1;
        tok = strtok(NULL, " ");
        int j = atoi(tok) - 1;
        tok = strtok(NULL, " ");
        (*A)[i * *ncols + j] = atof(tok);
    }
    fclose(mat);
    printf("%d x %d matrix stored successfully\n", *nrows, *ncols);
    return true;
}

// Parse the matrix market .mat file and store the results in a dense matrix
bool parse_store_mat(const char *filename, const int nrows, const int ncols, double *A) {
    const static size_t buffer_size = 2048;
    FILE *mat = fopen(filename, "r");
    if (!mat) { 
        fprintf(stderr, "No file named %s", filename);
        return false;
    }
    char line[buffer_size];
    // Store mat
    while(fgets(line, buffer_size, mat)) {
        if (!strncmp(line, "%", 1)) { continue; }
        char *tok = strtok(line, " "); 
        int i = atoi(tok) - 1;
        tok = strtok(NULL, " ");
        int j = atoi(tok) - 1;
        tok = strtok(NULL, " ");
        A[i * ncols + j] = atof(tok);
    }
    fclose(mat);
    printf("%d x %d matrix stored successfully\n", nrows, ncols);
    return true;
}

#endif
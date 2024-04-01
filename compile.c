#include <stdio.h>

#include "linearize.h"
#include "compile.h"

void compile_linearized_to_c(const char *filename, linearized_t *linearized) {
    FILE *f = fopen(filename, "w");
    char l[] = "what da haaaa\n";
    fwrite(l, 1, sizeof(l) - 1, f);
    fclose(f);
}

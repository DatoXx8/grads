set -xe
clang main.c linearize.c tensor.c nn.c compile.c -o grad -Ofast -mavx2 -mbmi2 -lm -lOpenCL
./grad

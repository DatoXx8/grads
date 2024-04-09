set -xe
clang main.c linearize.c tensor.c nn.c compile.c -o grad -ggdb -lm -lOpenCL -Wall -Wextra -pedantic -fsanitize=address
./grad

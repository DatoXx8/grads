set -xe
zig cc ./tensor.c ./tests/simulate-ops-cpu.c -o simulate-ops-cpu -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-ops-cpu
zig cc ./tensor.c ./tests/simulate-tree.c -o simulate-tree -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-tree 1000 10 1000
zig cc ./tensor.c ./compile.c ./runtimes/cl.c ./tests/simulate-compiler.c -o simulate-compiler -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-compiler 1000 10 10

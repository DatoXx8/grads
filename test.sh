zig cc ./tensor.c ./tests/simulate-ops-cpu.c -o simulate-ops-cpu -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-ops-cpu
zig cc ./tensor.c ./tests/simulate-linear.c -o simulate-linear -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-linear 1000 10
zig cc ./tensor.c ./compile.c ./runtimes/cl.c ./tests/simulate-compiler.c -o simulate-compiler -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-compiler 1000 10

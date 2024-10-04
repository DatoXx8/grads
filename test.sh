zig cc ./tensor.c ./tests/simulate-ops-cpu.c -o simulate-ops-cpu -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-ops-cpu
zig cc ./tensor.c ./tests/simulate-linearized.c -o simulate-linearized -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-linearized 1000 10
zig cc ./tensor.c ./compiler/codegen.c ./compiler/compile.c ./runtimes/cl.c ./tests/simulate-compiler.c -o simulate-compiler -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
./simulate-compiler

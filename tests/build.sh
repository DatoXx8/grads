set -xe
zig cc simulate-linearizer.c ../tensor.c ../linearize.c -o simulate-linearizer -lm -lasan -lOpenCL -Wall -Wextra -pedantic -fsanitize=null,address,undefined,leak,pointer-compare,pointer-subtract -ggdb
zig cc unit-ops-cpu.c ../tensor.c ../linearize.c -o unit-ops-cpu -lm -lasan -lOpenCL -Wall -Wextra -pedantic -fsanitize=address,undefined,leak,pointer-compare,pointer-subtract -ggdb
zig cc simulate-compiler.c ../tensor.c ../linearize.c ../compile.c ../runtimes/cl.c -o simulate-compiler -lm -lOpenCL -Wall -Wextra -pedantic

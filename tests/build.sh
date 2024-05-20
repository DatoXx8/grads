set -xe
zig cc ./simulate-ops-cpu.c ../tensor.c -o simulate-ops-cpu -lm -lasan -lOpenCL -Wall -Wextra -pedantic -fsanitize=address,undefined,leak,pointer-compare,pointer-subtract -ggdb
zig cc ./simulate-compiler.c ../tensor.c ../compile.c ../runtimes/cl.c -o simulate-compiler -lm -lOpenCL -Wall -Wextra -pedantic -ggdb
# zig cc ./simulate-tree.c ../tensor.c -o simulate-tree -lasan -lm -lOpenCL -Wall -Wextra -pedantic -fsanitize=address,undefined,leak,pointer-compare,pointer-subtract -ggdb
zig cc ./simulate-tree.c ../tensor.c -o simulate-tree -lm -lOpenCL -Wall -Wextra -pedantic -ggdb

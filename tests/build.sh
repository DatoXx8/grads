clang simulate_linearizer.c ../tensor.c ../linearize.c -o simulate_linearizer -lm -Wall -Wextra -pedantic -fsanitize=address,undefined,leak,pointer-compare,pointer-subtract -ggdb
clang unit_ops_cpu.c ../tensor.c ../linearize.c -o unit_ops_cpu -lm -Wall -Wextra -pedantic -fsanitize=address,undefined,leak,pointer-compare,pointer-subtract -ggdb

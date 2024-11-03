# Grads

Grads is a deep learning framework written in C. It is mainly a recreational project, though you can use it if you like.
It is essentialy an optimizing transpiler to OpenCL.

> [!IMPORTANT]
> In the future this will probably be ported to Zig and the C version will likely be abandoned at that point.

## Dependencies

Right now, there are only 3 dependencies:
- a C compiler
- the C standard library
- OpenCL runtime & headers

## Installation

1. Copy this repository as a subdirectory of your main project. You can do this by using the following command in the directory of your project:
``` sh
git clone https://github.com/DatoXx8/grads.git
```
2. Add all the `.c` files to your compile step, which could look as following:
``` sh
clang main.c [your files] ./grads/tensor.c ./grads/nn.c ./grads/compiler/codegen.c ./grads/compiler/compile.c ./grads/runtimes/cl.c -o grad -O3 -lm -lOpenCL -Wall -Wextra -pedantic
```

## Testing

This project has a big focus on testing, but be aware that despite that there will always be bugs in any sufficiently complex program including this one.

We test:
- All atomic ops give the expected results. (Unit testing)
- Our linearization of atomic ops gives the same results as calling the ops on their own. (Randomized simulation testing)
- Compiling any linearized ops gives the same results as calling the linearized ops with `linearized_run()`, within some small margin of error for hardware differences, at any optimization level. (Randomized simulation testing)

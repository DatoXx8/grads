# C Grad

C Grad is a deep learning framework written in C. It is mainly a recreational project, though you can use it if you like.

## Dependencies

Right now, there are only 2 (3 if you want OpenCl) dependencies:
- a C compiler
- the C standard library
- OpenCL runtime & headers

## Installation

1. Copy this repository as a subdirectory of your main project. You can do this by using the following command in the directory of your project:
``` sh
git clone https://github.com/DatoXx8/cgrad.git
```
2. Add all the .c files to your compile step, which could look as following:
``` sh
clang main.c [your files]  ./cgrad/tensor.c ./cgrad/nn.c ./cgrad/compile.c ./cgrad/runtimes/cl.c -o grad -O3 -lm -lOpenCL
```

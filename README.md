# C Grad

C Grad is a deep learning framework written in C. It is mainly a recreational project, though you can use it if you like.

## Dependencies

Right now, there are only 2  dependencies:
- a C compiler
- the C standard library

In the not so distant future, OpenCL will also be an optional dependency.

## Installation

1. Copy this repository as a subdirectory of your main project. You can do this by using the following command in the directory of your project:
``` sh
git clone https://github.com/DatoXx8/cgrad.git
```
2. Add all the .c files to your compile step, which could look as following:
    If you don't want OpenCL:
    ``` sh
    clang main.c <your files> ./cgrad/linearize.c ./cgrad/runtime.c ./cgrad/tensor.c ./cgrad/nn.c -o grad -Ofast -lm
    ```
    If you do want OpenCL:
    ``` sh
    clang main.c <your files> ./cgrad/linearize.c ./cgrad/runtime.c ./cgrad/tensor.c ./cgrad/nn.c -o grad -Ofast -lm -lOpenCL
    ```

## Usage

TODO: Add examples folder

First, since this is written in C and not a more fleshed out language, it can't use a lot of the more recent fancy features. Despite this, I still think C Grad is quite simple to use. You can look in the /cgrad/examples/ directory for several examples for programs i.e. for MNIST classification.

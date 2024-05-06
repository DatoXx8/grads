#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>

#include "cl.h"

cl_device_id cl_device_get(void) {
    cl_platform_id platform;
    cl_device_id dev;
    int err = 0;
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) { ERROR("Couldn't identify a OpenCL platform!\nError %d\n", err); }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if(err == CL_DEVICE_NOT_FOUND) { err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL); }
    if(err < 0) { ERROR("Couldn't access any devices!\nError %d\n", err); }
    return dev;
}
cl_program cl_program_build(cl_context context, cl_device_id device, const char *source, int64_t source_size) {
    uint64_t log_size;
    int err;
    char *program_log;
    cl_program program =
        clCreateProgramWithSource(context, 1, (const char **) &source, (const size_t *) &source_size, &err);
    if(err < 0) { ERROR("Couldn't create the program!\nError %d\n", err); }
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = calloc(log_size + 1, sizeof(char));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        ERROR("Could not build OpenCL program!\nError %d\n", err);
    }
    return program;
}
static void program_build(program_t *program) {
    int err;
    program->cl_command_queue =
        clCreateCommandQueueWithProperties(program->cl_context, program->cl_device_id, NULL, &err);
    if(err < 0) { ERROR("Could not create OpenCL command queue!\nError %d\n", err); }
    program->cl_program = calloc(1, sizeof(cl_program));
    *program->cl_program =
        cl_program_build(program->cl_context, program->cl_device_id, program->source, program->source_len);
}
/* NOTE: Compiles the program if it wasn't already. */
void program_run(program_t *program) {
    int err;
    if(!program->cl_program) {
        program_build(program);
        for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
            program->kernel->cl_kernel = calloc(1, sizeof(cl_kernel));
            *program->kernel->cl_kernel = clCreateKernel(*program->cl_program, program->kernel[kernel_idx].name, &err);
            if(err < 0) { ERROR("Could not create OpenCL kernel at index %lu\nError %d\n", kernel_idx, err); }
        }
    }
    for(int64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        if(kernel_idx) { printf("\n"); }
        for(int64_t arg_idx = 0; arg_idx < program->kernel[kernel_idx].arg_num; arg_idx++) {
            printf("%s\n", program->kernel[kernel_idx].args_name[arg_idx]);
            // clSetKernelArg(*program->kernel[kernel_idx].cl_kernel, arg_idx, sizeof(cl_mem),
            //                program->kernel[kernel_idx].args_mem[arg_idx]);
        }
        printf("%s\n", program->kernel[kernel_idx].source);
    }
}

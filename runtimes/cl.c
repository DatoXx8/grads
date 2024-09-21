#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../compiler/compile.h"
#include "../utils.h"
#include "cl.h"

/*
 * Nightmares - A poem about OpenCL
 *
 * In the dim lit room of coder’s plight,
 * OpenCL brings endless night.
 * Syntax snarls, errors rage,
 * Trapping minds in frustration’s cage.
 *
 * Segfaults haunt the silent code,
 * Memory leaks, burdens bestowed.
 * Threads in chaos, race in flight,
 * Cryptic messages, vague as night.
 *
 * Compiler's whispers, terse and cold,
 * Stories of torment, quietly told.
 * In this maze, hope’s thread is thin,
 * OpenCL, where battles begin.
 */

cl_device_id cl_device_get(void) {
    cl_platform_id platform;
    cl_device_id device;
    int err = 0;
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        ERROR("Couldn't identify a OpenCL platform!\nError %d\n", err);
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(err < 0) {
        ERROR("Couldn't access any devices!\nError %d\n", err);
    }
    return device;
}
cl_program cl_program_build(cl_context context, cl_device_id device, const char *source, uint64_t source_size) {
    uint64_t log_size;
    int err = 0;
    char *program_log;
    cl_program program =
        clCreateProgramWithSource(context, 1, (const char **) &source, (const size_t *) &source_size, &err);
    if(err < 0) {
        ERROR("Couldn't create the program!\nError %d\n", err);
    }
    /* TODO: This is really slow for large kernels. Maybe there is a way of splitting the kernels up in a smarter
     * fashion such that this builds faster */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        printf("%s\n", source);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = calloc(log_size + 1, sizeof(char));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        ERROR("Could not build OpenCL program!\nError %d\n", err);
    }
    return program;
}
/* Compiles the program if it wasn't already. All `cl_mem` fields need to be synced before and after this function to
 * make sure they are up to date. */
void program_run(program_t *program) {
    int err;
    for(uint64_t kernel_idx = 0; kernel_idx < program->kernel_num; kernel_idx++) {
        if(program->kernel[kernel_idx].cl_program == NULL) {
            program->kernel[kernel_idx].cl_program =
                cl_program_build(*program->cl_context, *program->cl_device_id, program->kernel[kernel_idx].source,
                                 program->kernel[kernel_idx].source_cap);
            program->kernel[kernel_idx].cl_kernel =
                clCreateKernel(program->kernel[kernel_idx].cl_program, KERNEL_NAME, &err);
            if(err < 0) {
                ERROR("Could not create OpenCL kernel\nError %d\n", err);
            }
            for(uint64_t arg_idx = 0; arg_idx < program->kernel[kernel_idx].arg_num; arg_idx++) {
                clSetKernelArg(program->kernel[kernel_idx].cl_kernel, arg_idx, sizeof(cl_mem),
                               &program->kernel[kernel_idx].arg_mem[arg_idx]);
            }
            clFinish(*program->cl_command_queue);
        }
        clEnqueueNDRangeKernel(*program->cl_command_queue, program->kernel[kernel_idx].cl_kernel, 1, NULL,
                               (size_t *) &program->global_size, (size_t *) &program->local_size, 0, NULL, NULL);
        clFinish(*program->cl_command_queue);
    }
}

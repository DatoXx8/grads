#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>

#include "cl.h"

cl_device_id device_get(void) {
    cl_platform_id platform;
    cl_device_id dev;
    int err = 0;
    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) { ERROR("Couldn't identify a platform %d", err); }
    // GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    // if(err == CL_DEVICE_NOT_FOUND) {
    //    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    // }
    if(err < 0) { ERROR("Couldn't access any devices"); }
    return dev;
}
cl_program program_build(cl_context context, cl_device_id device, const char *source, int64_t source_size) {
    uint64_t log_size;
    int err;
    char *program_log;
    cl_program program =
        clCreateProgramWithSource(context, 1, (const char **) &source, (const size_t *) &source_size, &err);
    if(err < 0) { ERROR("Couldn't create the program"); }
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    return program;
}

void program_run(program_t *program) {}

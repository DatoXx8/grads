#include <CL/cl.h>
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
cl_program program_build(cl_context context, cl_device_id device, const char *filename) {
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        ERROR("Couldn't find the program file");
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    program = clCreateProgramWithSource(context, 1, (const char **) &program_buffer, &program_size, &err);
    if(err < 0) {
        ERROR("Couldn't create the program");
    }
    free(program_buffer);
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

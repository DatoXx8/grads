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
    if(err < 0) { ERROR("Couldn't access any devices!\nError %d\n", err); }
    return dev;
}
cl_program cl_program_build(cl_context context, cl_device_id device, const char *source, int64_t source_size) {
    uint64_t log_size;
    int err = 0;
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
    program->cl_program = calloc(1, sizeof(cl_program));
    *program->cl_program =
        cl_program_build(*program->cl_context, *program->cl_device_id, program->source, program->source_len);
}
/* Compiles the program if it wasn't already. All `cl_mem` fields need to be synced before and after this function to
 * make sure they are up to date. */
void program_run(program_t *program) {
    int err;
    if(!program->cl_program) {
        program_build(program);
        program->cl_kernel = clCreateKernel(*program->cl_program, KERNEL_NAME, &err);
        if(err < 0) { ERROR("Could not create OpenCL kernel\nError %d\n", err); }
    }
    for(int64_t arg_idx = 0; arg_idx < program->arg_num; arg_idx++) {
        clSetKernelArg(program->cl_kernel, arg_idx, sizeof(cl_mem),
                       &program->arg_mem[arg_idx]);
    }
    clEnqueueNDRangeKernel(*program->cl_command_queue, program->cl_kernel, 1, NULL,
                           (size_t *) &program->global_size,
                           (size_t *) &program->local_size, 0, NULL, NULL);
    clFinish(*program->cl_command_queue);
}

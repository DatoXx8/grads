#ifndef CL_H_
#define CL_H_

#include <CL/cl.h>

#include "../linearize.h"
#include "../compile.h"

extern cl_device_id device_get(void);
extern cl_program program_build(cl_context context, cl_device_id device, const char *filename);

extern void program_run(program_t *program);

#endif

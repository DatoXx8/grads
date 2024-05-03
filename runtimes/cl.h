#ifndef CL_H_
#define CL_H_

#include <CL/cl.h>

#include "../compile.h"
#include "../linearize.h"

extern cl_device_id cl_device_get(void);
extern cl_program cl_program_build(cl_context context, cl_device_id device, const char *source, int64_t source_size);

extern void program_run(program_t *program);

#endif

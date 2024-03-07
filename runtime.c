#include "tensor.h"
#include "nn.h"
#include "linearize.h"
#include "runtime.h"

extern runtime_t runtime_allocate(enum runtime_e type, linearized_t *linearized);
extern void runtime_execute(runtime_t *runtime);
extern void runtime_free(runtime_t *runtime);

# Grads

Grads is a deep learning framework written in Zig. It is a recreational project, though you can use it for serious projects if you want to.
It is essentialy an optimizing transpiler to OpenCL.

Grads is currently licensed under the [Mozilla Public License](https://www.mozilla.org/en-US/MPL/2.0/).

## Dependencies

Right now, there are only 3 dependencies:
- The Zig compiler
- the Zig standard library
- OpenCL runtime & headers

I am not planning to add any dependencies, and if anything I will try to get rid of everything but the Zig compiler, though that would be a *LOT* of work. 
So don't count on that happening anytime soon.

## Installation

For now Grads only supports being directly included as source code in your projects.

Copy this repository to your project as a subdirectory. You can do this with the following command:
``` sh
git clone https://github.com/DatoXx8/grads.git
```
From there I recommend creating a Grads module which could look something like this:
```zig
const grads = b.addModule("grads", .{
    .root_source_file = b.path("grads/src/root.zig"),
});
```
And then you need to add the module to the executables where you want to use it:
```zig
exe.root_module.addImport("grads", grads);
```
Then you can import grads in a source file as follows:
```zig
const grads = @import("grads");
```

## Example

An example program will be uploaded soon.

## Testing

To run the unit tests for singular ops run `zig build test-op`.

To run the simulation tests for linearized ops run `zig build test-linearized`.

To run the simulation tests for the compiler run `zig build test-compiler`.

To profile the optimizer you can run `zig build profile-compiler`. (The optimizer does not yet deserve it's name as an optimizer)

In case a test fails, you can open an issue on the GitHub with the random seed and because the tests are deterministic that means me and others can then fix that bug.

## Coming up

I will not work on these in optimal order, because this just a hobby project.

| Step                      | Expected work | Expected gain                             |
| ------------------------- | ------------- | ----------------------------------------- |
| Compiler optimizations    | High          | Very high                                 |
| Get rid of stdlib         | Moderate      | Low                                       |
| Write OpenCl debug impl   | Very high     | High                                      |
| Autograd                  | Moderate      | None (Performance), Moderate (Dev time)   |
| Multi GPU                 | High (?)      | Very high                                 |
| Completely custom OpenCL  | Extreme       | High to extreme                           |

To provide some context on the custom OpenCL implementation: compiling a kernel in profile-compiler with O0 and op_num set to 10 takes about 1–2 seconds on average.
More than 99% of that time is spent by the Nvidia OpenCL compiler. 
I know that this is with O0, so the results are obviously very skewed, but still, being over 100 times longer is unacceptable.

Seeing this makes me worry that the actual computation might be hilariously slow as well, but I’m not really sure how to measure it.
This is why the expected gain has such a wide range.

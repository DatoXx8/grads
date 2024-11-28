# Grads

Grads is a deep learning framework written in Zig. It is mainly a recreational project, though you can use it for serious projects if you want to.
It is essentialy an optimizing transpiler to OpenCL.

Grads is currently licensed under the [MIT License](https://spdx.org/licenses/MIT.html) but it may be switched to the [Mozilla Public License](https://www.mozilla.org/en-US/MPL/2.0/)

> [!IMPORTANT]
> This is the Zig version and it is still very early in it's development.

## Dependencies

Right now, there are only 3 dependencies:
- The Zig compiler
- the Zig standard library
- OpenCL runtime & headers

## Installation

1. Copy this repository as a subdirectory of your main project. You can do this by using the following command in the directory of your project:
``` sh
git clone https://github.com/DatoXx8/grads.git
```
2. Compile the project, you can use something else than `ReleaseSafe` but that disables assertions, so be warned!
``` sh
zig build -Doptmize=ReleaseSafe
```

## Testing

To run the unit tests for singular ops run `zig build test-op`.
To run the simulation tests for linearized ops run `zig build test-linearized`. (Not yet implemented)
To run the simulation tests for the compiler run `zig build test-compiler`. (Not yet implemented)

In case a test fails, you can open an issue on the GitHub with the random seed and because the tests are deterministic that means me and others can then fix that bug.

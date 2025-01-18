# Grads

Grads is a deep learning framework written in Zig. It is mainly a recreational project, though you can use it for serious projects if you want to.
It is essentialy an optimizing transpiler to OpenCL.

Grads is currently licensed under the [Mozilla Public License](https://www.mozilla.org/en-US/MPL/2.0/).

## Dependencies

Right now, there are only 3 dependencies:
- The Zig compiler
- the Zig standard library
- OpenCL runtime & headers

## Installation

1. Copy this repository to your local machine. You can do this with the following command:
``` sh
git clone https://github.com/DatoXx8/grads.git
```
2. Compile the project, you can use something else than `ReleaseSafe` but that disables assertions, so be warned!
``` sh
zig build -Doptmize=ReleaseSafe
```
3. In the `zig-out/lib/` directory there will be a file called `libgrads.a`. You can then use the library as you wish from there!

## Example

An example progam will soon be uploaded.

## Testing

To run the unit tests for singular ops run `zig build test-op`.
To run the simulation tests for linearized ops run `zig build test-linearized`.
To run the simulation tests for the compiler run `zig build test-compiler`.

To profile the optimiser you can run `zig build profile-compiler`. (The optimiser does not yet deserve it's name as an optimiser)

In case a test fails, you can open an issue on the GitHub with the random seed and because the tests are deterministic that means me and others can then fix that bug.

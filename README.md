# dynamatic-utils

Personal utilities for hacking, prototyping, and benchmarking around [Dynamatic++](https://github.com/EPFL-LAP/dynamatic) (private repository for now) and [legacy Dynamatic](https://github.com/lana555/dynamatic) (public repository).  ***Nothing in this repository should be considered stable by any stretch of the imagination.*** Things *will* change (i.e., break) regularly without warning or excuse. Yet, this repository contains a number of useful scripts that can be very helpful in your day-to-day Dynamatic workflows. Over time, some things will become deprecated and/or graduate to being called ⭐*stable*⭐.

## Repository Map

The repository contains many scripts and utilities, which are not necessarily connected to each other. This section aims to provide some high-level guidance inside the irregularly-documented infrastructure and on what you can do with it.


### [`.env` file](.env)

The `.env` file located at the top-level contains, as is traditional, a number of environment variables (mostly, file paths) that many parts of the repository use. As such, it is advised to always source the file before doing anything in the repository.

```sh
$ source .env
```

As written inside the file, the only variables you should ever modify are the three defined at the top (all the ones below are defined based on these three):

- `LEGACY_DYNAMATIC_PATH`: path to [legacy Dynamatic](https://github.com/lana555/dynamatic). If installing the latter using the [dedicated script](setup_legacy_dynamatic.sh), the provided location matches the one where legacy Dynamatic will automatically be cloned.
- `LEGACY_DYNAMATIC_LLVM_PATH`: path to LLVM source used internally by legacy Dynamatic. Legacy Dynamatic's users normally build this as part of the [elastic-circuits build instructions](https://github.com/lana555/dynamatic/tree/master/elastic-circuits). Again, if installing legacy Dynamatic using the [dedicated script](setup_legacy_dynamatic.sh), the provided location matches the one where LLVM will automatically be cloned.
- `DYNAMATIC_PATH`: path to Dynamatic, wherever it is in your filesystem.

Note that the `realpath` shell command does not exist on all shells. If your shell does not support this, just write down the absolute paths manually. 


### [`setup_legacy_dynamatic.sh` script](setup_legacy_dynamatic.sh)

The `setup_legacy_dynamatic.sh` automatically clones legacy Dynamatic in the subdirectory `legacy-dynamatic/dhls/etc/dynamatic`. Then, it attempts to automatically build:
- *elastic-circuits* (elastic pass, converting LLVM IR into dataflow circuits in the DOT format)
- *buffers* (smart buffer placement that solves MILPs with Gurobi)
- *dot2vhdl* (converts DOTs to VHDL designs)
- *hlsVerifier* (VHDL-level verification)

This is largely untested, so your mileage *will* vary.

### [`runner.sh` script](runner.sh)

The `runner.sh` script is extremely useful to run a sequence of transformations or conversion steps on a range of benchmarks at the same time. At the current time, it supports automatic compilation with legacy Dynamatic and Dynamatic (no flags, done automatically), simulation using legacy Dynamatic's backend and Modelsim (`--simulate`), and synthesization using legacy Dynamatic's backend and Vivado HLS (`--synthesize`). The script is configured using command-line arguments (and internally uses environment variables defined in [.env](.env)). You can also run `./runner.sh --help` to see the command line interface and available options. 

The script runs benchmarks from [Dynamatic's integration-tests](https://github.com/EPFL-LAP/dynamatic/tree/main/integration-test). All benchmarks are ran by default. However, users may instead choose to only run specific benchmarks by providing their name as positional arguments to the script (see examples below).

Importantly, users should select a *flow* (`--flow <flow-name>`) to run when calling the script. *Flows* describe a sequence of transformation/analysis steps to apply on selected benchmarks. Current options are:
- `dynamatic`: full compilation with Dynamatic from souce to DOT, then converts to VHDL using legacy Dynamatic's toolchain.
- `legacy`: full compilation with legacy Dynamatic from source to VHDL.
- `bridge`: hybrid run using Dynamatic down to ubuffered DOT then bridging to legacy Dynamatic which performs buffer placement and VHDL export.

The script contains bash functions with identical names containing the sequence of steps followed by each flow.

Output files for each run are placed in a directory adjacent to the source directory of each benchmark and named identically to the flow that was executed. For example, running the `dynamatic` flow on the `fir` benchmark will create a directory `/path/to/dynamatic/integration-test/fir/dynamatic` to contain the run's results. 

Some examples:
```sh
# Run on all benchmarks using dynamatic flow (with simple buffers).
./runner.sh --flow dynamatic
# Run legacy flow with smart buffers. Simulate and synthesize each design. 
./runner.sh --flow legacy --smart-buffers --simulate --synthesize
# Run ONLY the fir and gaussian benchmarks (with simple buffers) with the bridge flow.
./runner.sh --flow bridge fir gaussian
```

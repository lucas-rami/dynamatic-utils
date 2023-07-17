# dynamatic-utils

Personal utilities for hacking, prototyping, and benchmarking around [Dynamatic++](https://github.com/EPFL-LAP/dynamatic) (private repository for now) and [legacy Dynamatic](https://github.com/lana555/dynamatic) (public repository).  ***Nothing in this repository should be considered stable by any stretch of the imagination.*** Things *will* change (i.e., break) regularly without warning or excuse. Yet, this repository contains a number of useful scripts that can be very helpful in your day-to-day Dynamatic workflows. Over time, some things will become deprecated and/or graduate to being called ⭐*stable*⭐.

## Repository Map

The repository contains many scripts and utilities, which are not necessarily connected to each other. This section aims to provide some high-level guidance inside the irregularly-documented infrastructure and on what you can do with it.


### [`.env` file](.env)

The `.env` file located at the top-level contains, as is traditional, a number of environment variables (mostly, file pathes) that many parts of the repository use. As such, it is advised to always source the file before doing anything in the repository.

```sh
$ source .env
```

As written inside the file, the only variables you should ever modify are the three defined at the top (all the ones below are defined based on these three):

- `LEGACY_DYNAMATIC_PATH`: path to [legacy Dynamatic](https://github.com/lana555/dynamatic). If installing the latter using the [dedicated script](setup_legacy_dynamatic.sh), the provided location matches the one where legacy Dynamatic will automatically be cloned.
- `LEGACY_DYNAMATIC_LLVM_PATH`: path to LLVM source used internally by legacy Dynamatic. Legacy Dynamatic's users normally build this as part of the [elastic-circuits build instructions](https://github.com/lana555/dynamatic/tree/master/elastic-circuits). Again, if installing legacy Dynamatic using the [dedicated script](setup_legacy_dynamatic.sh), the provided location matches the one where LLVM will automatically be cloned.
- `DYNAMATIC_PATH`: path to Dynamatic++, wherever it is in your filesystem.

Note that the `realpath` shell command does not exist on all shells. If your shell does not support this, just write down the absolute paths manually. 


### [`setup_legacy_dynamatic.sh` script](setup_legacy_dynamatic.sh)

The `setup_legacy_dynamatic.sh` automatically clones legacy Dynamatic in the subdirectory `legacy-dynamatic/dhls/etc/dynamatic`. Then, it attempts to automatically build:
- *elastic-circuits* (elastic pass, converting LLVM IR into dataflow circuits (DOTs))
- *buffers* (smart buffer placement using MILPs)
- *dot2vhdl* (converts DOTs to VHDL designs)
- *hlsVerifier* (VHDL-level verification)
This is largely untested, so your mileage *will* vary.

### [`runner.sh` script](runner.sh)

The `runner.sh` script is extremely useful to run a sequence of transformations/conversions on a range of benchmarks at the same time. At the current time, it supports automatic compilation with legacy Dynamatic and Dynamatic++ (no flags, done automatically), simulation using legacy Dynamatic's backend and Modelsim (`--simulate`), and synthesization using legacy Dynamatic's backend and Vivado HLS (`--synthesize`).

The script uses a couple core concepts to allow the user to personalize their runs, which are described below. The script is configured using command-line arguments (and internally uses environment variables defined in [.env](.env)). You can also run `./runner --help` to see the command line interface and available options.

- *Testsuites* (`--testsuite <testsuite-name>`) represent a list of benchmarks on which the transformations will be applied sequentially. Current options are:
    - `dynamatic`: legacy Dynamatic's regression tests, stored in legacy Dynamatic's repository
    - `fpl22`: benchmarks used in [this paper](https://ieeexplore.ieee.org/abstract/document/10035134) published at FPL22, stored in this repository under `benchmarks/FPL22`
- *Flows* (`--flow <flow-name>`) describe a sequence of transformation/analysis steps to apply on selected benchmarks. The script contains bash functions with identical names representing the sequence of transformations performed by each flow. Current options are:
    - `dynamatic`: Run using Dynamatic++, which at the moment can only compile from source to a netlist-level representation in MLIR. Simulation and synthesis are *not* currently available.
    - `legacy`: Run using legacy Dynamatic from source to VHDL. Simulation and synthesis are available.
    - `bridge`: Hybrid run using Dynamatic++ down to dataflow representation (Handshake-level IR) then bridging to legacy Dynamatic using DOT format. Goes from source to VHDL. Simulation and synthesis are available.

Output files for each run are placed in a directory adjacent to the source directory of each benchmark and named identically to the flow that was executed. For example, running the `dynamatic` flow on the `fpl22` benchmarks will create a directory `benchmarks/FPL22/<benchmark-name>/dynamatic` for each benchmark to contain the run's results. 

Some examples:
```sh
# Run the FPL22 benchmarks using dynamatic flow
./runner.sh --testsuite fpl22 --flow dynamatic
# Run the FPL22 benchmarks using legacy flow and smart buffers. Simulate and synthesize each design. 
./runner.sh --testsuite fpl22 --flow legacy --smart-buffers --simulate --synthesize
# Run ONLY the fir and gaussian benchmarks from the FPL22 testsuite using the bridge flow
./runner.sh --testsuite fpl22 --flow bridge fir gaussian
```

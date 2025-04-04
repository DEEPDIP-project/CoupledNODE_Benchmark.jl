# Benchmark script

This script is used to benchmark the performance of the different
implementations of the algorithm. Different configuration can be set in the
`configs/conf.yaml` file or as environment variable as ENV["CONF_FILE"], see
example [`configs/conf.yaml`](configs/conf.yaml) file for more details.

There are two main scripts in the `Benchmark` repository:
* `single_benchmark.jl": a script to train a single model (define in its config file), and produce its specific analysis for the model defined architecture.
* `benchmark.jl`: compares the performance of all the models that have been trained already, and produces some summary plots to quantify which model is the best.

## Installation and running

To install the dependecies and run the benchmark script, make sure you are in
the `Benchmark` directory and run the following command in julia:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
juila> Pkg.include("benchmark.jl")
```

To install the dependecies, you can also run the
following command in the terminal:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The `-e` flag allows you to execute the commands without opening the Julia REPL.

To run the script:

```bash
julia --project benchmark.jl
```

### Setting the number of CPU cores

To set the number of CPU cores to be used by julia, you can set the argument `-p`
followed by the number of cores to be used. For example:

```bash
julia --project -p 4 benchmark.jl
```

To set number of cores for the number of threads for multithreading, you can set
the argument `-t` followed by the number of threads to be used. For example,
`auto` will use all available threads:

```bash
julia --project -t auto benchmark.jl
```

## Development

To develop the benchmark script, you can use `Revise` package to automatically reload
the script when changes are made. To do so, run the following command in julia:

```julia
julia> using Revise
julia> includet("benchmark.jl")  # instead of include("benchmark.jl")
```

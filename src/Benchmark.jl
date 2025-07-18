"""
Utility functions for scripts.
"""
module Benchmark

using Accessors
using Adapt
using CairoMakie # for plotting
using ComponentArrays
using CoupledNODE
using CoupledNODE: loss_priori_lux, create_loss_post_lux
using CUDA
using Dates
using DifferentialEquations
using DocStringExtensions
using EnumX
using LinearAlgebra
using IncompressibleNavierStokes
using IncompressibleNavierStokes:
    momentum!, divergence!, project!, apply_bc_u!, kinetic_energy!, scalewithvolume!
using JLD2
using LoggingExtras
using Lux
using MLUtils
using NeuralClosure
using Observables
using Optimization
using Optimisers
using Random
using Statistics: mean
NS = Base.get_extension(CoupledNODE, :NavierStokes)

"Write output to file, as the default SLURM file is not updated often enough."
function setsnelliuslogger(logfile)
    filelogger = MinLevelLogger(FileLogger(logfile), Logging.Info)
    logger = TeeLogger(ConsoleLogger(), filelogger)
    oldlogger = global_logger(logger)
    @info """
    Logging to file: $logfile

    Starting at $(Dates.now()).

    Last commit:

    $(cd(() -> read(`git log -n 1`, String), @__DIR__))
    """
    oldlogger
end

# Inherit docstring templates
@template (MODULES, FUNCTIONS, METHODS, TYPES) = IncompressibleNavierStokes

"Load JLD2 file as named tuple (instead of dict)."
function namedtupleload(file)
    dict = load(file)
    k, v = keys(dict), values(dict)
    pairs = @. Symbol(k) => v
    (; pairs...)
end

"""
Make file name from parts.

```@example
julia> splatfileparts("toto", 23; haha = 1e3, hehe = "hihi")
"toto_23_haha=1000.0_hehe=hihi"
```
"""
function splatfileparts(args...; kwargs...)
    sargs = string.(args)
    skwargs = map((k, v) -> string(k) * "=" * string(v), keys(kwargs), values(kwargs))
    s = [sargs..., skwargs...]
    join(s, "_")
end

function getsetup(; params, nles)
    Setup(;
        x = ntuple(α -> range(params.lims..., nles + 1), params.D),
        params.Re,
        params.backend,
        params.bodyforce,
        params.issteadybodyforce,
    )
end

include("observe.jl")
include("rk.jl")
include("train.jl")
include("io.jl")
include("plots.jl")

export setsnelliuslogger
export namedtupleload, splatfileparts
export observe_u, observe_v
export ProjectOrder, RKProject
export getdatafile, createdata, getsetup
export trainprior, loadprior
export trainpost, loadpost
export trainsmagorinsky, loadsmagorinsky

export read_config, check_necessary_files
export _convert_to_single_index,
    plot_prior_traininghistory,
    plot_posteriori_traininghistory,
    plot_divergence,
    plot_energy_evolution,
    plot_energy_evolution_hist,
    plot_energy_spectra,
    plot_training_time,
    plot_training_comptime,
    plot_inference_time,
    plot_num_parameters,
    plot_error,
    plot_epost_vs_t,
    plot_dns_solution,
    plot_csv_comparison

export compute_eprior, compute_epost, compute_t_prior_inference
export reusepriorfile, reusepostfile
export load_data_set
export create_test_dns_proj

end # module Benchmark

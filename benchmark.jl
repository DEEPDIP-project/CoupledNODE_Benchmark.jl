# Here we compare all the models trained in the previous notebook

#! format: off
if false                      #src
    include("src/Benchmark.jl") #src
end                           #src

@info "Script started"
@info VERSION

using Pkg
@info Pkg.status()

#############################################
# Identify the models that have been trained
using Glob
filter_out = ["plots", "logs", "posttraining", "priortraining", "data"]
list_models = filter(x -> !any(pattern -> occursin(pattern, x), filter_out), glob("output/kolmogorov/*"))

list_confs = filter(x -> !any(pattern -> occursin(pattern, x), filter_out), glob("configs/*"))

#############################################
# Device
if CUDA.functional()
    ## For running on a CUDA compatible GPU
    @info "Running on CUDA"
    cuda_active = true
    backend = CUDABackend()
    CUDA.allowscalar(false)
    device = x -> adapt(CuArray, x)
    clean() = (GC.gc(); CUDA.reclaim())
else
    ## For running on CPU.
    ## Consider reducing the sizes of DNS, LES, and CNN layers if
    ## you want to test run on a laptop.
    @warn "Running on CPU"
    cuda_active = false
    backend = CPU()
    device = identity
    clean() = nothing
end
########################################################################## #src
@info "Loading packages"

if "CoupledNODE" in keys(Pkg.installed())
    @info "CoupledNODE already installed"
else
    Pkg.add(PackageSpec(rev = "main", url = "https://github.com/DEEPDIP-project/CoupledNODE.jl.git"))
end

using Accessors
using Adapt
using CairoMakie
using CoupledNODE: loss_priori_lux, create_loss_post_lux
using CUDA
using DifferentialEquations
using IncompressibleNavierStokes.RKMethods
using JLD2
using LaTeXStrings
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using ParameterSchedulers
using Random
using SparseArrays
using Benchmark
using Dates
using IncompressibleNavierStokes
using NeuralClosure
using CoupledNODE
NS = Base.get_extension(CoupledNODE, :NavierStokes)

##########################################################################
# Loop over the trained models

# Initialize the figure
fig = Figure(; size = (950, 600))
ax = Axis(
    fig[1, 1];
    title = "A-priori error for different configurations",
    xlabel = "Iteration",
    ylabel = "A-priori error",
)

for (i, conf_file) in enumerate(list_confs)
    @info "Reading configuration file $conf_file"
    conf = NS.read_config(conf_file)
    closure_name = conf["closure"]["name"]
    model_path = joinpath("output", "kolmogorov", closure_name)
    # Check if the model exists
    if !ispath(model_path)
        @error "Model $closure_name has not been trained yet"
        continue
    end
    conf["params"]["backend"] = deepcopy(backend)

    # Choose where to put output
    basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : @__DIR__
    outdir = joinpath(basedir, "output", "kolmogorov")
    outdir_model = joinpath(outdir, closure_name)

    # Load learned parameters and training times
    priortraining = loadprior(outdir, closure_name, params.nles, params.filters)
    θ_cnn_prior = map(p -> copyto!(copy(θ_start), p.θ), priortraining)
    @info "" θ_cnn_prior .|> extrema # Check that parameters are within reasonable bounds

    # Training times
    map(p -> p.comptime, priortraining)
    map(p -> p.comptime, priortraining) |> vec .|> x -> round(x; digits = 1)
    map(p -> p.comptime, priortraining) |> sum |> x -> x / 60 # Minutes

    # Add lines for each configuration
    for (ig, nles) in enumerate(params.nles)
        lines!(
            ax,
            priortraining[ig, 1].lhist_nomodel,
            label = "$closure_name (n = $nles, No closure)",
            linestyle = :dash,
        )
        for (ifil, Φ) in enumerate(params.filters)
            label = Φ isa FaceAverage ? "FA" : "VA"
            lines!(ax, priortraining[ig, ifil].lhist_val; label = "$closure_name (n = $nles, $label)")
        end
    end
end

# Add legend
axislegend(ax)

# Save and display the figure
figdir = joinpath(outdir, "comparison", "priortraining")
ispath(figdir) || mkpath(figdir)
save("$figdir/validationerror.pdf", fig)
display(fig)

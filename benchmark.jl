# Here we compare all the models trained in the previous notebook
#! format: off
if false
    include("src/Benchmark.jl") #src
end

@info "Script started"
@info VERSION

# Identify the models that have been trained
basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : @__DIR__
outdir = joinpath(basedir, "output", "kolmogorov")
confdir = joinpath(basedir, "configs")
compdir = joinpath(outdir, "comparison")
ispath(compdir) || mkpath(compdir)

# List configurations files
using Glob
list_confs = glob("*.yaml", confdir)

@info "Loading packages"

using Pkg
if "CoupledNODE" in keys(Pkg.installed())
    @info "CoupledNODE already installed"
else
    Pkg.add(PackageSpec(rev = "main", url = "https://github.com/DEEPDIP-project/CoupledNODE.jl.git"))
end

using CairoMakie # for plotting
using CUDA
using JLD2  # for saving plots
using Benchmark  # for loading post and prior
using IncompressibleNavierStokes  # for CPU()
using NeuralClosure  # for models
using CoupledNODE # for reading config

NS = Base.get_extension(CoupledNODE, :NavierStokes)

# Device
if CUDA.functional()
    ## For running on a CUDA compatible GPU
    @info "Running on CUDA"
    cuda_active = true
    backend = CUDABackend()
    CUDA.allowscalar(false)
else
    ## For running on CPU.
    ## Consider reducing the sizes of DNS, LES, and CNN layers if
    ## you want to test run on a laptop.
    @warn "Running on CPU"
    cuda_active = false
    backend = CPU()
end

# Define some functions
function read_config(config_file, backend)
    conf = NS.read_config(config_file)
    closure_name = conf["closure"]["name"]
    model_path = joinpath("output", "kolmogorov", closure_name)

    # Check if the model exists
    msg = """
    Model $closure_name has not been trained yet
    for configuration $config_file
    """
    if !ispath(model_path)
        @error msg
    end

    # Parameters
    conf["params"]["backend"] = deepcopy(backend)
    params = NS.load_params(conf)

    return closure_name, params, conf
end

function missing_label(ax, label)
    for plt in ax.scene.plots
        if :label in keys(plt.attributes)
            return false
        end
    end
    return true
end

function plot_prior(outdir, closure_name, params, ax, color)
    # Load learned parameters and training times
    priortraining = loadprior(outdir, closure_name, params.nles, params.filters)

    # Add lines
    for (ig, nles) in enumerate(params.nles)
        label = "No closure (n = $nles)"
        if missing_label(ax, label)  # add No closure only once
            lines!(
                ax,
                priortraining[ig, 1].lhist_nomodel,
                label = "No closure (n = $nles)",
                linestyle = :dash,
                color = color,
            )
        end
        for (ifil, Φ) in enumerate(params.filters)
            label = Φ isa FaceAverage ? "FA" : "VA"
            # TODO: if xtick should be checked
            lines!(
                ax,
                priortraining[ig, ifil].lhist_val;
                label = "$closure_name (n = $nles, $label)",
                color = color,
            )
        end
    end
end

function plot_posteriori(outdir, closure_name, projectorders, params, ax, color)
    # Load learned parameters
    posttraining = loadpost(outdir, closure_name, params.nles, params.filters, projectorders)

    # Add lines
    for (ig, nles) in enumerate(params.nles)
        for (ifil, Φ) in enumerate(params.filters)
            label = Φ isa FaceAverage ? "FA" : "VA"
            y = posttraining[ig, ifil].lhist_val
            # TODO check step between ticks
            ax.xticks = 1:length(y)  # because x is "iteration", it should be integer
            scatterlines!(
                ax,
                y;
                label = "$closure_name (n = $nles, $label)",
                linestyle = :dash,  # should not interpolate between points
                marker = :circle,
                color = color,
            )
        end
    end
end

function create_figure(title, xlabel, ylabel, size = (950, 600))
    fig = Figure(; size = size)
    ax = Axis(
        fig[1, 1];
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
    )
    return fig, ax
end

plot_labels = Dict(
    "prior_error" => Dict(
        "title" => "A-priori error for different configurations",
        "xlabel" => "Iteration",
        "ylabel" => "A-priori error",
    ),
    "posteriori_error" => Dict(
        "title" => "A-posteriori error for different configurations",
        "xlabel" => "Iteration",
        "ylabel" => "DCF",
    ),
)

# loop over plot types and configurations
for key in keys(plot_labels)
    @info "Plotting $key"
    # Create the figure
    title = plot_labels[key]["title"]
    xlabel = plot_labels[key]["xlabel"]
    ylabel = plot_labels[key]["ylabel"]
    fig, ax = create_figure(title, xlabel, ylabel)

    # Loop over the configurations
    for (i, conf_file) in enumerate(list_confs)
        @info "Reading configuration file $conf_file"
        closure_name, params, conf = read_config(conf_file, backend)
        @info "Plotting $closure_name"
        color = Cycled(i + 1)  # make sure each config has a consistent color

        if key == "prior_error"
            plot_prior(outdir, closure_name, params, ax, color)
        elseif key == "posteriori_error"
            projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
            plot_posteriori(outdir, closure_name, projectorders, params, ax, color)
        end
    end
    # Add legend
    axislegend(ax)

    # Display and save the figure
    save("$compdir/$(key)_validationerror.pdf", fig)
    @info "Saved $compdir/$(key)_validationerror.pdf"
    display(fig)
end

@info "Script ended"

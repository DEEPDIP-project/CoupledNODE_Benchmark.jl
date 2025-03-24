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

function _update_ax_limits(ax, x, y)
    # Get data limits with extra space for the legend
    xmin, xmax = extrema(x) .* (1, 1.5)
    ymin, ymax = extrema(y) .* (1, 1.5)

    # Get current axis limits
    current_xmin, current_xmax = something(ax.limits[][1], (xmin, xmax))
    current_ymin, current_ymax = something(ax.limits[][2], (ymin, ymax))

    # Update limits
    xlims!(ax, extrema((xmin, xmax, current_xmin, current_xmax)))
    ylims!(ax, extrema((ymin, ymax, current_ymin, current_ymax)))

    return ax
end

function missing_label(ax, label)
    for plt in ax.scene.plots
        if :label in keys(plt.attributes) && plt.attributes[:label][] == label
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
                label = label,
                linestyle = :dash,
                linewidth = 2,
                color = "black",
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
            ax.xticks = 1:length(y)  # because y is "iteration", it should be integer
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

function plot_divergence(outdir, closure_name, projectorders, params, ax, color)
    divergence_dir = joinpath(outdir, closure_name,  "history.jld2")
    divergencehistory = namedtupleload(divergence_dir).divergencehistory;

    for (igrid, nles) in enumerate(params.nles)
        for iorder in 1:length(projectorders),
            (ifil, Φ) in enumerate(params.filters)

            I = CartesianIndex(igrid, ifil, iorder)

            # add No closure only once
            label = "No closure (n = $nles)"
            if missing_label(ax, label)
                lines!(
                    ax,
                    divergencehistory.nomodel[I];
                    label = label,
                    linestyle = :dash,
                    linewidth = 2,
                    color = "black",
                    )
            end

            # add reference only once
            label = "Reference"
            if missing_label(ax, label)
                lines!(
                    ax,
                    divergencehistory.ref[I];
                    color = "black",
                    linestyle = :dot,
                    linewidth = 2,
                    label = label,
                )
            end

            label = Φ isa FaceAverage ? "FA" : "VA"
            lines!(
                ax,
                divergencehistory.cnn_prior[I];
                label = "$closure_name (prior) (n = $nles, $label)",
                linestyle = :dashdot,
                color = color,
            )
            lines!(
                ax,
                divergencehistory.cnn_post[I];
                label = "$closure_name (post) (n = $nles, $label)",
                color = color,
            )

            # update axis limits
            x_values = [point[1] for v in values(divergencehistory) for point in v[I]]
            y_values = [point[2] for v in values(divergencehistory) for point in v[I]]
            ax = _update_ax_limits(ax, x_values, y_values)

            ax.yscale = log10
        end
    end
end

function plot_energy_evolution(outdir, closure_name, projectorders, params, ax, color)
    energy_dir = joinpath(outdir, closure_name, "history.jld2")
    energyhistory = namedtupleload(energy_dir).energyhistory;

    for (igrid, nles) in enumerate(params.nles)
        for iorder in 1:length(projectorders),
            (ifil, Φ) in enumerate(params.filters)

            I = CartesianIndex(igrid, ifil, iorder)

            # add No closure only once
            label = "No closure (n = $nles)"
            if missing_label(ax, label)
                lines!(
                    ax,
                    energyhistory.nomodel[I];
                    label = label,
                    linestyle = :dash,
                    linewidth = 2,
                    color = "black",
                    )
            end

            # add reference only once
            label = "Reference"
            if missing_label(ax, label)
                lines!(
                    ax,
                    energyhistory.ref[I];
                    color = "black",
                    linestyle = :dot,
                    linewidth = 2,
                    label = label,
                )
            end

            label = Φ isa FaceAverage ? "FA" : "VA"
            lines!(
                ax,
                energyhistory.cnn_prior[I];
                label = "$closure_name (prior) (n = $nles, $label)",
                linestyle = :dashdot,
                color = color,
            )
            lines!(
                ax,
                energyhistory.cnn_post[I];
                label = "$closure_name (post) (n = $nles, $label)",
                color = color,
            )

            # update axis limits
            x_values = [point[1] for v in values(energyhistory) for point in v[I]]
            y_values = [point[2] for v in values(energyhistory) for point in v[I]]
            ax = _update_ax_limits(ax, x_values, y_values)
        end
    end
end

# Loop over plot types and configurations
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
    "divergence" => Dict(
        "title" => "Divergence for different configurations",
        "xlabel" => "t",
        "ylabel" => "Face-average",
    ),
    "energy_evolution" => Dict(
        "title" => "Energy evolution for different configurations",
        "xlabel" => "t",
        "ylabel" => "E(t)",
    ),
)

#TODO check for the colors of internal loops
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
        elseif key == "divergence"
            projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
            plot_divergence(outdir, closure_name, projectorders, params, ax, color)
        elseif key == "energy_evolution"
            projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
            plot_energy_evolution(outdir, closure_name, projectorders, params, ax, color)

        end
    end
    # Add legend
    axislegend(ax)

    # Display and save the figure
    save("$compdir/$(key).pdf", fig)
    @info "Saved $compdir/$(key).pdf"
    display(fig)
end

@info "Script ended"

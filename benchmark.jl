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

function _update_ax_limits(ax, x, y)
    # Get data limits with extra space for the legend
    (xmin, xmax), (ymin, ymax) = extrema(x), extrema(y)
    xmax += (xmax - xmin) * 0.5
    ymax += (ymax - ymin) * 0.5

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

function plot_prior(priortraining, closure_name, ig, nles, ifil, Φ, ax, color)
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
    label = Φ isa FaceAverage ? "FA" : "VA"
    y = priortraining[ig, ifil].lhist_val
    lines!(
        ax,
        y;
        label = "$closure_name (n = $nles, $label)",
        color = color,
    )
    ax = _update_ax_limits(ax, collect(1:length(y)), y)
end

function plot_posteriori(posttraining, closure_name, ig, nles, ifil, Φ, ax, color)
    label = Φ isa FaceAverage ? "FA" : "VA"
    y = posttraining[ig, ifil].lhist_val
    ax.xticks = 1:length(y)  # because y is "iteration", it should be integer
    scatterlines!(
        ax,
        y;
        label = "$closure_name (n = $nles, $label)",
        linestyle = :dash,  # should not interpolate between points
        marker = :circle,
        color = color,
    )
    ax = _update_ax_limits(ax, collect(1:length(y)), y)
end

function plot_divergence(divergencehistory, closure_name, ig, nles, ifil, Φ, ax, color)
    I = CartesianIndex(ig, ifil, 1)

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

function plot_energy_evolution(energyhistory, closure_name, ig, nles, ifil, Φ, ax, color)
    I = CartesianIndex(ig, ifil, 1)

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

function _get_spectra(setup, u)
    time_indices = eachindex(u)
    field_names = [:ref, :nomodel, :cnn_prior, :cnn_post]
    # Create a nested structure specs[itime][I]
    specs = map(time_indices) do itime
        I_indices = eachindex(u[itime].ref)
        map(I_indices) do I
            map(field_names) do k
            state = (; u = u[itime][k][I])
            spec = observespectrum(state; setup)
            spec.ehat[]
            end
        end
    end
    return specs
end

function _build_inertia_slope(kmax, specs, κ)
    # Build inertial slope above energy
    logkrange = [0.45 * log(kmax), 0.85 * log(kmax)]
    krange = exp.(logkrange)
    slope, slopelabel = -3.0, L"$\kappa^{-3}$"
    slopeconst = maximum(specs ./ κ .^ slope)
    offset = 3
    inertia = offset .* slopeconst .* krange .^ slope
    return krange, inertia, slopelabel
end

function plot_energy_spectra(solutions, all_specs, κ, closure_name, ig, nles, ifil, Φ, fig, color)
    kmax = maximum(κ)

    I = CartesianIndex(ig, ifil, 1)

    for (itime, t) in enumerate(solutions.t)
        ## Nice ticks
        logmax = round(Int, log2(kmax + 1))
        if logmax > 5
            xticks = [1.0, 4.0, 16.0, 64.0, 256.0]
        else
            xticks = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        end

        ## Make plot
        subfig = fig[:, itime]
        title = "t = $(round(t; digits = 1))"
        ax = Axis(
            subfig;
            xticks,
            title = title,
            xlabel = "κ",
        )

        specs = all_specs[itime][I]
        # add No closure
        lines!(
            ax,
            κ,
            specs[2];
            label = "No closure (n = $nles)",
            linestyle = :dash,
            linewidth = 2,
            color = "black",
            )

        # add reference
        lines!(
            ax,
            κ,
            specs[1];
            color = "black",
            linestyle = :dot,
            linewidth = 2,
            label = "Reference",
        )

        label = Φ isa FaceAverage ? "FA" : "VA"
        lines!(
            ax,
            κ,
            specs[3];
            label = "$closure_name (prior) (n = $nles, $label)",
            linestyle = :dashdot,
            color = color,
        )
        lines!(
            ax,
            κ,
            specs[4];
            label = "$closure_name (post) (n = $nles, $label)",
            color = color,
        )
        krange, inertia, slopelabel = _build_inertia_slope(kmax, specs[1], κ)
        lines!(
            ax,
            krange,
            inertia;
            color = Cycled(1),
            label = slopelabel,
            linestyle = :dot,
        )
        ax.yscale = log10
        ax.xscale = log2

        if itime == 1
            ax.ylabel = "DCF"
        end
        if itime == length(solutions.t)
            fig[:, itime + 1] = Legend(fig, ax)
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
    "energy_spectra" => Dict(
        "title" => "Energy spectra",
    ),
)

#TODO check for the colors of internal loops
for key in keys(plot_labels)
    @info "Plotting $key"

    # Create the figure
    fig = Figure(; size = (950, 600))
    if key != "energy_spectra"
        ax = Axis(
            fig[1, 1];
            title = plot_labels[key]["title"],
            xlabel = plot_labels[key]["xlabel"],
            ylabel = plot_labels[key]["ylabel"],
        )
    end

    # Loop over the configurations
    for (i, conf_file) in enumerate(list_confs)
        @info "Reading configuration file $conf_file"
        closure_name, params, conf = read_config(conf_file, backend)

        # make sure each config has a consistent color
        color = Cycled(i + 1)

        for (ig, nles) in enumerate(params.nles),
            (ifil, Φ) in enumerate(params.filters)

            if key == "prior_error"
                # Load learned parameters and training times
                priortraining = loadprior(
                    outdir, closure_name, params.nles, params.filters
                )
                plot_prior(
                    priortraining, closure_name, ig, nles, ifil, Φ, ax, color
                )

            elseif key == "posteriori_error"
                # Load learned parameters
                projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                posttraining = loadpost(
                    outdir, closure_name, params.nles, params.filters, projectorders
                )
                plot_posteriori(
                    posttraining, closure_name, ig, nles, ifil, Φ, ax, color
                )

            elseif key == "divergence"
                divergence_dir = joinpath(outdir, closure_name,  "history.jld2")
                divergencehistory = namedtupleload(divergence_dir).divergencehistory;
                plot_divergence(
                    divergencehistory, closure_name, ig, nles, ifil, Φ, ax, color
                )

            elseif key == "energy_evolution"
                energy_dir = joinpath(outdir, closure_name, "history.jld2")
                energyhistory = namedtupleload(energy_dir).energyhistory;
                plot_energy_evolution(
                    energyhistory, closure_name, ig, nles, ifil, Φ, ax, color
                )

            elseif key=="energy_spectra"
                energy_dir = joinpath(outdir, closure_name, "solutions.jld2")
                solutions = namedtupleload(energy_dir);
                setup = getsetup(; params, nles)
                κ = IncompressibleNavierStokes.spectral_stuff(setup).κ
                all_specs = _get_spectra(setup, solutions.u)

                Label(
                    fig[0, 0],
                    "Energy spectra for different configurations",

                )
                subfig = fig[i, :]
                plot_energy_spectra(
                    solutions, all_specs, κ, closure_name, ig, nles, ifil, Φ, subfig, color
                )
            end
        end
    end
    # Add legend
    if key != "energy_spectra"
        axislegend(ax, position = :rt)
    end

    # Display and save the figure
    save("$compdir/$(key).pdf", fig)
    @info "Saved $compdir/$(key).pdf"
    display(fig)
end

@info "Script ended"

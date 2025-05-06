# Here we compare all the models trained in the previous notebook
#! format: off
if false
    include("src/Benchmark.jl") #src
end

@info "Script started"
@info VERSION

# Color palette for consistent theme throughout paper
# Colors: Rich Red, Olive Green, Warm Orange, Deep Purple, Coral Pink, Teal
# if you have more than 6 models, you can add more colors to the palette
palette = (; color = ["#cc0000", "#669900", "#ff9900", "#9933cc", "#ff6666", "#66cccc"])

# Identify the models that have been trained
basedir = haskey(ENV, "DEEPDIP") ? ENV["DEEPDIP"] : @__DIR__
outdir = joinpath(basedir, "output", "kolmogorov")
confdir = joinpath(basedir, "configs/local")
@warn "Using configuration files from $confdir"
compdir = joinpath(outdir, "comparison")
ispath(compdir) || mkpath(compdir)

# List configurations files
using Glob
list_confs = glob("*.yaml", confdir)
if isempty(list_confs)
    @error "No configuration files found in $confdir"
end


using Pkg
if "CoupledNODE" in keys(Pkg.installed())
    @info "CoupledNODE already installed"
else
    Pkg.add(PackageSpec(rev = "main", url = "https://github.com/DEEPDIP-project/CoupledNODE.jl.git"))
end

using Benchmark  # for loading post and prior
using CairoMakie # for plotting
using CUDA
using JLD2  # for saving plots
using IncompressibleNavierStokes  # for CPU()
using NeuralClosure  # for models
using CoupledNODE # for reading config

NS = Base.get_extension(CoupledNODE, :NavierStokes)

# Device
using CUDA
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
    backend = IncompressibleNavierStokes.CPU()
end

# Global variables for setting linestyle and colors in all plots
PLOT_STYLES = Dict(
    :no_closure => (color="black", linestyle=:dash, linewidth=2),
    :reference => (color="black", linestyle=:dot, linewidth=2),
    :prior => (color="black", linestyle=:solid, linewidth=1),
    :post => (color="black", linestyle=:dashdot, linewidth=1),
    :inertia => (color="blue", linestyle=:dot, linewidth=1),
    :smag => (color="green", linestyle=:dot, linewidth=1),
)


# Loop over plot types and configurations
plot_labels = Dict(
    :prior_error => (
        title  = "A-priori error for different configurations",
        xlabel = "Iteration",
        ylabel = "A-priori error",
    ),
    :posteriori_error => (
        title  = "A-posteriori error for different configurations",
        xlabel = "Iteration",
        ylabel = "DCF",
    ),
    :divergence => (
        title  = "Divergence for different configurations",
        xlabel = "t",
        ylabel = "Face-average",
    ),
    :energy_evolution => (
        title  = "Energy evolution for different configurations",
        xlabel = "t",
        ylabel = "E(t)",
    ),
    :energy_spectra => (
        title  = "Energy spectra",
    ),
    :prior_time => (
        title  = "A-priori time for different configurations",
        xlabel = "Model",
        ylabel = "Training time (s)",
    ),
    :posteriori_time => (
        title  = "A-posteriori time for different configurations",
        xlabel = "Model",
        ylabel = "Training time (s)",
    ),
)

set_theme!(palette = palette)

for key in keys(plot_labels)
    @info "Plotting $key"

    # Create the figure
    fig = Figure(; size = (950, 600))
    if key != :energy_spectra
        ax = Axis(
            fig[1, 1];
            title = plot_labels[key].title,
            xlabel = plot_labels[key].xlabel,
            ylabel = plot_labels[key].ylabel,
        )
    end

    # empty list for barplots
    bar_positions = Int[]
    bar_labels = String[]

    # Loop over the configurations
    for (i, conf_file) in enumerate(list_confs)
        @info "Reading configuration file $conf_file"
        closure_name, params, conf = read_config(conf_file, backend)

        # Loop over the parameters
        CUDA.allowscalar() do
            for (ig, nles) in enumerate(params.nles),
                (ifil, Φ) in enumerate(params.filters)

                projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                if length(projectorders) > 1
                    @warn "Multiple project orders found in configuration $conf_file. Do not know how to handle this yet."
                    continue
                end

                if !check_necessary_files(
                    outdir,
                    closure_name,
                    nles,
                    Φ,
                    projectorders[1],
                )
                    @error "Some files are missing for configuration $conf_file. Skipping"
                    continue
                end

                # make sure each combination has a consistent color
                #TODO this function should be tested
                col_index = _convert_to_single_index(
                    i, ig, ifil, length(params.nles), length(params.filters)
                )
                color = Cycled(col_index + 1)

                data_index = CartesianIndex(ig, ifil, 1)  # projectorders = 1

                if key == :prior_error
                    plot_prior(
                        outdir, closure_name, nles, Φ, ax, color, PLOT_STYLES
                    )

                elseif key == :posteriori_error
                    projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                    plot_posteriori(
                        outdir, closure_name, nles, Φ, projectorders, ax, color, PLOT_STYLES
                    )

                elseif key == :divergence
                    plot_divergence(
                        outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES
                    )

                elseif key == :energy_evolution
                    plot_energy_evolution(
                        outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES
                    )

                elseif key== :energy_spectra
                    Label(
                        fig[0, :],
                        "Energy spectra for different configurations";
                        font = :bold,
                        tellwidth=false,
                    )
                    plot_energy_spectra(
                        outdir, params, closure_name, nles, Φ, data_index, fig[i, :], color, PLOT_STYLES
                    )

                elseif key == :prior_time
                    plot_prior_time(
                        outdir, closure_name, nles, Φ, col_index, ax, color
                    )
                    push!(bar_positions, col_index)
                    push!(bar_labels, "$closure_name")
                elseif key == :posteriori_time
                    projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                    plot_posteriori_time(
                        outdir, closure_name, nles, Φ, projectorders, col_index, ax, color
                    )
                    push!(bar_positions, col_index)
                    push!(bar_labels, "$closure_name")
                end
            end
        end
    end
    # Add legend
    if key != :energy_spectra
        axislegend(ax, position = :rt)
    end

    # Add xticks in barplot
    if key == :prior_time || key == :posteriori_time
        ax.xticks = (bar_positions, bar_labels)
    end

    # Display and save the figure
    save("$compdir/$(key).pdf", fig)
    @info "Saved $compdir/$(key).pdf"
    display(fig)
end

@info "Script ended"

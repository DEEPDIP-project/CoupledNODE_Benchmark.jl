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
confdir = joinpath(basedir, "configs/local")
confdir = joinpath(basedir, "configs/snellius64")
@warn "Using configuration files from $confdir"
compdir = joinpath(outdir, "comparison")
ispath(compdir) || mkpath(compdir)

# List configurations files
using Glob
exclude_patterns = ["att", "cno", "int", "back", "rk4", "cnn_1" ]
exclude_patterns = ["att", "cno" ]
exclude_patterns = ["rod", "ken", "ow", "tsi", "back" ]
include_patterns = ["cnn_base", "ins"] 
include_patterns = []

if !isempty(include_patterns)
    @warn "Including only configurations with patterns: $(include_patterns)"
    all_confs = glob("*.yaml", confdir)
    list_confs = filter(conf -> any(occursin(pat, conf) for pat in include_patterns), all_confs)
else
    @warn "Excluding configurations with patterns: $(exclude_patterns)"
    all_confs = glob("*.yaml", confdir)
    list_confs = filter(conf -> all(!occursin(pat, conf) for pat in exclude_patterns), all_confs)
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
using LaTeXStrings

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

# Create a new list to store valid configs
valid_confs = String[]
for conf_file in list_confs
    closure_name, params, conf = read_config(outdir, conf_file, backend)
    if !check_necessary_files(
        outdir,
        closure_name,
        params.nles[1],
        params.filters[1],
        eval(Meta.parse(conf["posteriori"]["projectorders"]))[1]
    )
        @error "Some files are missing for configuration $conf_file. Skipping"
        continue
    end
    push!(valid_confs, conf_file)
end
# Replace list_confs with only the valid ones
list_confs = valid_confs

if isempty(list_confs)
    @error "No configuration files found in $confdir"
end

# Global variables for setting linestyle and colors in all plots
PLOT_STYLES = Dict(
    :no_closure => (color="black", linestyle=:dash, linewidth=2),
    :no_closure_proj => (color="red", linestyle=:dash, linewidth=2),
    :reference => (color="black", linestyle=:dot, linewidth=2),
    :reference_proj => (color="red", linestyle=:dot, linewidth=2),
    :prior => (color="black", linestyle=:solid, linewidth=1),
    :post => (color="black", linestyle=:dashdot, linewidth=1),
    :inertia => (color="cyan", linestyle=:dot, linewidth=1),
    :smag => (color="darkgreen", linestyle=:dot, linewidth=1),
)

# Color list: high-contrast, colorblind-friendly palette
colors_list = [
    "#377EB8", # Blue
    "#4DAF4A", # Green
    "#984EA3", # Purple
    "#FF7F00", # Orange
    "#A65628", # Brown
    "#F781BF", # Pink
    "#999999", # Grey
    "#FFD700", # Gold
    "#00CED1", # Dark Turquoise
    "#1E90FF", # Dodger Blue
    "#228B22", # Forest Green
    "#D2691E", # Chocolate
    "#DC143C", # Crimson
    "#8B008B", # Dark Magenta
    "#FF1493", # Deep Pink
    "#00FF7F", # Spring Green
    "#4682B4", # Steel Blue
    "#B22222", # Firebrick
    "#20B2AA", # Light Sea Green
]

# Loop over plot types and configurations
plot_labels = Dict(
    :dns_solution => (
        title  = "DNS solution for different configurations",
        xlabel = "t",
        ylabel = L"\frac{|u(t)-u_{proj}(t)|}{|u(t)|}",
    ),
    :prior_hist => (
        title  = "A-priori training history for different configurations",
        xlabel = "Iteration",
        ylabel = "A-priori error",
    ),
    :posteriori_hist => (
        title  = "A-posteriori training history for different configurations",
        xlabel = "Iteration",
        ylabel = "DCF",
    ),
    :divergence => (
        title  = "Divergence for different configurations",
        xlabel = "t",
        ylabel = "Face-average",
    ),
    :energy_evolution => (
        title = "Energy evolution for different configurations",
        xlabel = "t",
        ylabel = "E(t)",
    ),
    :energy_evolution_hist => (
        title = "Energy histogram for different configurations",
        xlabel = "frequency",
        ylabel = "E(t)",
    ),
    :energy_spectra => (
        title  = "Energy spectra",
    ),
    :num_parameters => (
        title  = "Number of parameters for different configurations",
        xlabel = "Model",
        ylabel = "Number of parameters",
    ),
    :eprior => (
        title  = "A-prior error for different configurations",
        xlabel = "Model",
        ylabel = "A-prior error",
    ),
    :epost => (
        title  = "A-posteriori error for different configurations",
        xlabel = "Model",
        ylabel = "A-posteriori error",
    ),
    :epost_vs_t => (
        title = "A-posteriori error as a function of time",
        xlabel = "t",
        ylabel = L"e_{M}(t)",
    ),
    :training_time => (
        title  = "Training time for different configurations",
        xlabel = "Model",
        ylabel = "Training time (s) (per iteration)",
    ),
    :training_comptime => (
        title  = "Training time for different configurations",
        xlabel = "Model",
        ylabel = "Full Training time (s)",
    ),
    :inference_time => (
        title  = "Inference time for different configurations",
        xlabel = "Model",
        ylabel = "Inference time (s)",
    ),
)

dns_seeds = splitseed(123456, 8)
dns_seeds = splitseed(16, 8)
#dns_seeds = [0x185efb6b]

for key in keys(plot_labels)
    @info "********************  Plotting $key"

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
    bar_positions = Float64[]
    bar_labels = String[]

    # Loop over the configurations
    for (i, conf_file) in enumerate(list_confs)
        @info "Reading configuration file $conf_file"
        closure_name, params, conf = read_config(outdir, conf_file, backend)

        # Loop over the parameters
        CUDA.allowscalar() do
            for (ig, nles) in enumerate(params.nles),
                (ifil, Φ) in enumerate(params.filters)

                projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                if length(projectorders) > 1
                    @warn "Multiple project orders found in configuration $conf_file. Do not know how to handle this yet."
                    continue
                end


                # make sure each combination has a consistent color
                #TODO this function should be tested
                col_index = _convert_to_single_index(
                    i, ig, ifil, length(params.nles), length(params.filters)
                )
                color = colors_list[col_index]

                data_index = CartesianIndex(ig, ifil, 1)  # projectorders = 1
                data_index_v = CartesianIndex(ig, ifil, 1, 6)

                if key == :prior_hist
                    plot_prior_traininghistory(
                        outdir, closure_name, nles, Φ, ax, color, PLOT_STYLES
                    )

                elseif key == :posteriori_hist
                    projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                    plot_posteriori_traininghistory(
                        outdir, closure_name, nles, Φ, projectorders, ax, color, PLOT_STYLES
                    )

                elseif key == :divergence
                    if closure_name == "INS.jl"
                        @info "Skipping INS.jl for divergence plot"
                        continue
                    end
                    plot_divergence(
                        outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES
                    )

                elseif key == :energy_evolution
                    if closure_name == "INS.jl"
                        @info "Skipping INS.jl for energy evolution plot"
                        continue
                    end
                    plot_energy_evolution(
                        outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES
                    )

                elseif key == :energy_evolution_hist
                    plot_energy_evolution_hist(
                        outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES
                    )

                elseif key== :energy_spectra
                    if closure_name == "INS.jl"
                        @info "Skipping INS.jl for energy spectra plot"
                        continue
                    end
                    # Plot energy spectra for all models in the same figure
                    # This is done to compare the models in a single plot
                    # and to avoid creating too many figures
                    num_of_models = length(list_confs)
                    # If INS.jl was in the list of configurations, reduce the number of models
                    if "INS.jl" in list_confs
                        num_of_models -= 1
                    end
                    plot_energy_spectra(
                        outdir, params, closure_name, nles, Φ, data_index, fig, i,
                        num_of_models, color, PLOT_STYLES
                    )

                elseif key == :training_time
                    projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                    bar_label, bar_position = plot_training_time(
                        outdir, closure_name, nles, Φ, projectorders, col_index, ax, color
                    )
                    append!(bar_positions, bar_position)
                    append!(bar_labels, bar_label)
                elseif key == :training_comptime
                    projectorders = eval(Meta.parse(conf["posteriori"]["projectorders"]))
                    bar_label, bar_position = plot_training_comptime(
                        outdir, closure_name, nles, Φ, projectorders, col_index, ax, color
                    )
                    append!(bar_positions, bar_position)
                    append!(bar_labels, bar_label)
                elseif key == :inference_time
                    bar_label, bar_position = plot_inference_time(
                        outdir, closure_name, nles, data_index, col_index, ax, color
                    )
                    append!(bar_positions, bar_position)
                    append!(bar_labels, bar_label)
                elseif key == :num_parameters
                    plot_num_parameters(
                        outdir, closure_name, nles, Φ, col_index, ax, color
                    )
                    push!(bar_positions, col_index)
                    push!(bar_labels, "$closure_name")
                elseif key == :eprior
                    error_file = joinpath(
                        outdir, closure_name, "eprior_nles=$(nles).jld2"
                    )
                    bar_label, bar_position = plot_error(
                        error_file, closure_name, nles, data_index, col_index, ax, color, PLOT_STYLES; outdir=outdir, Φ=Φ, projectorders=projectorders
                    )
                    append!(bar_positions, bar_position)
                    append!(bar_labels, bar_label)
                elseif key == :epost
                    error_file = joinpath(
                        outdir, closure_name, "epost_nles=$(nles).jld2"
                    )
                    bar_label, bar_position = plot_error(
                        error_file, closure_name, nles, data_index_v, col_index, ax, color, PLOT_STYLES; outdir=outdir, Φ=Φ, projectorders=projectorders
                    )
                    append!(bar_positions, bar_position)
                    append!(bar_labels, bar_label)
                elseif key == :epost_vs_t
                    error_file = joinpath(
                        outdir, closure_name, "epost_nles=$(nles).jld2"
                    )
                    plot_epost_vs_t(
                        error_file, closure_name, nles, ax, color, PLOT_STYLES
                    )
                elseif key == :dns_solution
                    infile = "output/kolmogorov/test_dns_proj.jld2"
                    outfile = joinpath(
                        compdir, "projection_dns_test_nles=$(nles).gif"
                    )

                    isfile(outfile) || plot_dns_solution(
                        ax, 5, infile, outfile
                    )
                else
                    @error "Unknown plot type: $key"
                end
            end
        end
    end
    # Add legend
    if !(key in (:energy_spectra, :dns_solution, :energy_evolution_hist))
        Legend(fig[:, end+1], ax)
    end

    # Add xticks in barplot
    #if key in (:training_time, :training_comptime, :inference_time, :num_parameters, :eprior, :epost)
    #    ax.xticks = (bar_positions, bar_labels)
    #end

    # Set log-log scale
    if key == :epost_vs_t
        ax.yscale = log10
        ax.xscale = log10
    end

    # Display and save the figure
    save("$compdir/$(key).pdf", fig)
    @info "Saved $compdir/$(key).pdf"
    display(fig)

end

csv_path = joinpath(outdir, "comparison/a_posteriori_errors.csv")
plot_csv_comparison(csv_path, compdir, colors_list)


@info "Script ended"
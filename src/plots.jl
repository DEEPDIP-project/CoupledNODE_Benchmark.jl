
function _update_ax_limits(ax, x, y)
    if any(isnan, y)
        return ax
    end
    # Get data limits with extra padding for the legend
    (xmin, xmax), (ymin, ymax) = extrema(x), extrema(y)
    xmax += (xmax - xmin) * 0.5 + max(1e-6, abs(xmax) * 1e-6)
    ymax += (ymax - ymin) * 0.5 + max(1e-6, abs(ymax) * 1e-6)

    # Get current axis limits
    current_xmin, current_xmax = something(ax.limits[][1], (xmin, xmax))
    current_ymin, current_ymax = something(ax.limits[][2], (ymin, ymax))

    # Add padding to the axis limits
    new_xmin, new_xmax = extrema((xmin, xmax, current_xmin, current_xmax))
    new_ymin, new_ymax = extrema((ymin, ymax, current_ymin, current_ymax))
    xlims!(ax, new_xmin / 1.01, new_xmax * 1.01)
    ylims!(ax, new_ymin / 1.01, new_ymax * 1.01)

    return ax
end

function _missing_label(ax, label)
    for plt in ax.scene.plots
        if :label in keys(plt.attributes) && plt.attributes[:label][] == label
            return false
        end
    end
    return true
end

function plot_prior_traininghistory(outdir, closure_name, nles, Φ, ax, color, PLOT_STYLES)
    # Load learned parameters
    priortraining = loadprior(outdir, closure_name, [nles], [Φ])

    label = "No closure (n = $nles)"
    if _missing_label(ax, label)  # add No closure only once
        lines!(
            ax,
            priortraining[1].lhist_nomodel,
            label = label,
            linestyle = PLOT_STYLES[:no_closure].linestyle,
            linewidth = PLOT_STYLES[:no_closure].linewidth,
            color = PLOT_STYLES[:no_closure].color,
        )
    end
    label = Φ isa FaceAverage ? "FA" : "VA"
    if closure_name == "INS_ref"
        y = [p[2] for p in priortraining[1].hist]
        x = [p[1] for p in priortraining[1].hist]
        # if x starts from 0, shift it to 1
        if x[1] == 0
            x = x .+ 1
        end
    else
        y = priortraining[1].lhist_val
        x = collect(1:length(y))
    end

    lines!(
        ax,
        x,
        y;
        label = "$closure_name (n = $nles, $label)",
        color = color, # dont change this color
        linestyle = PLOT_STYLES[:prior].linestyle,
        linewidth = PLOT_STYLES[:prior].linewidth,
    )
    ax = _update_ax_limits(ax, x, y)
end

function plot_posteriori_traininghistory(
    outdir,
    closure_name,
    nles,
    Φ,
    projectorders,
    ax,
    color,
    PLOT_STYLES,
)
    label = Φ isa FaceAverage ? "FA" : "VA"
    if closure_name == "INS_ref"
        # Here i need the _checkpoint file generated from PaperDC
        postfile = Benchmark.getpostfile(outdir, closure_name, nles, Φ, projectorders[1])
        checkfile = join(splitext(postfile), "_checkpoint")
        check = namedtupleload(checkfile)
        (; hist) = check.callbackstate
        y = [p[2] for p in hist]
        x = [p[1] for p in hist]
        # if x starts from 0, shift it to 1
        if x[1] == 0
            x = x .+ 1
        end
    else
        posttraining = loadpost(outdir, closure_name, [nles], [Φ], projectorders)
        y = posttraining[1].lhist_val
        x = collect(1:length(y))
    end
    scatterlines!(
        ax,
        x,
        y;
        label = "$closure_name (n = $nles, $label)",
        linestyle = PLOT_STYLES[:post].linestyle,  # should not interpolate between points
        linewidth = PLOT_STYLES[:post].linewidth,
        marker = :circle,
        color = color, # dont change this color
    )

    # TODO: check if ticks are overwriting each other
    ax.xticks = 1:length(y)  # because y is "iteration", it should be integer
    ax = _update_ax_limits(ax, x, y)
end

function plot_divergence(outdir, closure_name, nles, Φ, data_index, ax, color, PLOT_STYLES)
    # Load learned parameters
    divergence_dir = joinpath(outdir, closure_name, "history.jld2")
    if !ispath(divergence_dir)
        @warn "Divergence history not found in $divergence_dir"
        return
    end
    divergencehistory = namedtupleload(divergence_dir).divergencehistory;

    # add No closure only once
    label = "No closure (n = $nles)"
    if _missing_label(ax, label) && label in keys(divergencehistory)
        lines!(
            ax,
            divergencehistory.nomodel[data_index];
            label = label,
            linestyle = PLOT_STYLES[:no_closure].linestyle,
            linewidth = PLOT_STYLES[:no_closure].linewidth,
            color = PLOT_STYLES[:no_closure].color,
        )
    end

    # add reference only once
    label = "Reference"
    if _missing_label(ax, label) && label in keys(divergencehistory)
        lines!(
            ax,
            divergencehistory.ref[data_index];
            color = PLOT_STYLES[:reference].color,
            linestyle = PLOT_STYLES[:reference].linestyle,
            linewidth = PLOT_STYLES[:reference].linewidth,
            label = label,
        )
    end

    label = "smag"
    if _missing_label(ax, label) && label in keys(divergencehistory)
        lines!(
            ax,
            divergencehistory.smag[data_index];
            color = PLOT_STYLES[:smag].color,
            linestyle = PLOT_STYLES[:smag].linestyle,
            linewidth = PLOT_STYLES[:smag].linewidth,
            label = label,
        )
    end

    label = Φ isa FaceAverage ? "FA" : "VA"
    lines!(
        ax,
        divergencehistory.model_prior[data_index];
        label = "$closure_name (prior) (n = $nles, $label)",
        linestyle = PLOT_STYLES[:prior].linestyle,
        linewidth = PLOT_STYLES[:prior].linewidth,
        color = color, # dont change this color
    )
    lines!(
        ax,
        divergencehistory.model_post[data_index];
        label = "$closure_name (post) (n = $nles, $label)",
        linestyle = PLOT_STYLES[:post].linestyle,
        linewidth = PLOT_STYLES[:post].linewidth,
        color = color, # dont change this color
    )

    # update axis limits
    x_values = [point[1] for v in values(divergencehistory) for point in v[data_index]]
    y_values = [point[2] for v in values(divergencehistory) for point in v[data_index]]
    ax = _update_ax_limits(ax, x_values, y_values)

    ax.yscale = log10
end

function plot_energy_evolution(
    outdir,
    closure_name,
    nles,
    Φ,
    data_index,
    ax,
    color,
    PLOT_STYLES,
)
    # Load learned parameters
    energy_dir = joinpath(outdir, closure_name, "history.jld2")
    if !ispath(energy_dir)
        @warn "Energy history not found in $energy_dir"
        return
    end
    energyhistory = namedtupleload(energy_dir).energyhistory;

    # add No closure only once
    label = "No closure (n = $nles)"
    if _missing_label(ax, label) && label in keys(energyhistory)
        lines!(
            ax,
            energyhistory.nomodel[data_index];
            label = label,
            linestyle = PLOT_STYLES[:no_closure].linestyle,
            linewidth = PLOT_STYLES[:no_closure].linewidth,
            color = PLOT_STYLES[:no_closure].color,
        )
    end

    # add reference only once
    label = "Reference"
    if _missing_label(ax, label) && label in keys(energyhistory)
        lines!(
            ax,
            energyhistory.ref[data_index];
            color = PLOT_STYLES[:reference].color,
            linestyle = PLOT_STYLES[:reference].linestyle,
            linewidth = PLOT_STYLES[:reference].linewidth,
            label = label,
        )
    end

    label = "smag"
    if _missing_label(ax, label) && label in keys(energyhistory)
        lines!(
            ax,
            energyhistory.smag[data_index];
            color = PLOT_STYLES[:smag].color,
            linestyle = PLOT_STYLES[:smag].linestyle,
            linewidth = PLOT_STYLES[:smag].linewidth,
            label = label,
        )
    end

    label = Φ isa FaceAverage ? "FA" : "VA"
    lines!(
        ax,
        energyhistory.model_prior[data_index];
        label = "$closure_name (prior) (n = $nles, $label)",
        linestyle = PLOT_STYLES[:prior].linestyle,
        linewidth = PLOT_STYLES[:prior].linewidth,
        color = color, # dont change this color
    )
    lines!(
        ax,
        energyhistory.model_post[data_index];
        label = "$closure_name (post) (n = $nles, $label)",
        linestyle = PLOT_STYLES[:post].linestyle,
        linewidth = PLOT_STYLES[:post].linewidth,
        color = color, # dont change this color
    )

    # update axis limits
    x_values = [point[1] for v in values(energyhistory) for point in v[data_index]]
    y_values = [point[2] for v in values(energyhistory) for point in v[data_index]]
    ax = _update_ax_limits(ax, x_values, y_values)
end

function _get_spectra(setup, u)
    time_indices = eachindex(u)
    field_names = [:ref, :nomodel, :model_prior, :model_post]
    # Create a nested structure specs[itime][I]
    specs = map(time_indices) do itime
        I_indices = eachindex(u[itime].ref)
        map(I_indices) do I
            map(field_names) do k
                if sizeof(u[itime][k][I]) == 0
                    @warn "No data for itime = $itime, I = $I, k = $k, so reusing itime=1. Need better data!"
                    uki = zeros(ComplexF64)
                    uki = u[1][k][I]
                else
                    uki = u[itime][k][I]
                end
                state = (; u = uki)
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

function plot_energy_spectra(
    outdir,
    params,
    closure_name,
    nles,
    Φ,
    data_index,
    fig,
    model_i,
    num_of_models,
    color,
    PLOT_STYLES,
)
    # Load learned parameters
    energy_dir = joinpath(outdir, closure_name, "solutions.jld2")
    if !ispath(energy_dir)
        @warn "Energy spectra not found in $energy_dir"
        return
    end
    solutions = namedtupleload(energy_dir);

    setup = getsetup(; params, nles)
    κ = IncompressibleNavierStokes.spectral_stuff(setup).κ
    all_specs = _get_spectra(setup, solutions.u)

    kmax = maximum(κ)

    # Create a grid of plots and legends
    gtitle = fig[1, 1]  # Title of the figure
    gplot = fig[2, 1]
    gplot_ax = gplot[1, 1]   # Axis for each plot
    gplot_leg = gplot[1, 2]  # The common legend for all plots

    Label(
        gtitle,
        "Energy spectra for different configurations";
        font = :bold,
        tellwidth = false,
    )

    for (itime, t) in enumerate(solutions.t)
        ## Nice ticks
        logmax = round(Int, log2(kmax + 1))
        if logmax > 5
            xticks = [1.0, 4.0, 16.0, 64.0, 256.0]
        else
            xticks = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        end

        ## Make plot
        t_title = model_i == 1 ? "t = $(round(t; digits = 1))" : ""
        k_xlable = model_i == num_of_models ? "κ" : ""
        ax = CairoMakie.Axis(
            gplot_ax[model_i, itime];
            xticks,
            title = t_title,
            xlabel = k_xlable,
        )

        specs = all_specs[itime][data_index]

        # add No closure
        no_closure_label = "No closure (n = $nles)"
        no_closure_plt = lines!(
            ax,
            κ,
            specs[2];
            label = no_closure_label,
            linestyle = PLOT_STYLES[:no_closure].linestyle,
            linewidth = PLOT_STYLES[:no_closure].linewidth,
            color = PLOT_STYLES[:no_closure].color,
        )

        # add reference
        reference_label = "Reference"
        reference_plt = lines!(
            ax,
            κ,
            specs[1];
            color = PLOT_STYLES[:reference].color,
            linestyle = PLOT_STYLES[:reference].linestyle,
            linewidth = PLOT_STYLES[:reference].linewidth,
            label = reference_label,
        )

        label = Φ isa FaceAverage ? "FA" : "VA"
        prior_label = "Prior (n = $nles, $label)"
        prior_plt = lines!(
            ax,
            κ,
            specs[3];
            label = prior_label,
            linestyle = PLOT_STYLES[:prior].linestyle,
            linewidth = PLOT_STYLES[:prior].linewidth,
            color = color, # dont change this color
        )
        post_label = "Post (n = $nles, $label)"
        post_plt = lines!(
            ax,
            κ,
            specs[4];
            label = post_label,
            linestyle = PLOT_STYLES[:post].linestyle,
            linewidth = PLOT_STYLES[:post].linewidth,
            color = color, # dont change this color
        )
        krange, inertia, inertia_label = _build_inertia_slope(kmax, specs[1], κ)
        inertia_plt = lines!(
            ax,
            krange,
            inertia;
            color = PLOT_STYLES[:inertia].color,
            label = inertia_label,
            linestyle = PLOT_STYLES[:inertia].linestyle,
            linewidth = PLOT_STYLES[:inertia].linewidth,
        )
        ax.yscale = log10
        ax.xscale = log2

        ax.xticklabelsize = 8
        ax.yticklabelsize = 8

        if itime == 1
            ax.ylabel = "$closure_name"
        end

        # Add legend only for Prior and Post to each row
        if itime == length(solutions.t)
            Legend(
                gplot_ax[model_i, itime+1],
                [prior_plt, post_plt],
                [prior_label, post_label],
                labelsize = 8,
            )
        end

        # Add legend that is common for all plots
        Legend(
            gplot_leg,
            [no_closure_plt, reference_plt, inertia_plt],
            [no_closure_label, reference_label, inertia_label],
            labelsize = 8,
        )
    end
end

function _convert_to_single_index(i, j, k, dimj, dimk)
    return (i - 1) * dimj * dimk + (j - 1) * dimk + k
end

function plot_prior_time(outdir, closure_name, nles, Φ, model_index, ax, color)
    # Load learned parameters
    priortraining = loadprior(outdir, closure_name, [nles], [Φ])

    # training time in seconds
    training_time = map(p -> p.comptime, priortraining) |> vec .|> x -> round(x; digits = 3)

    label = Φ isa FaceAverage ? "FA" : "VA"
    barplot!(
        ax,
        [model_index],
        training_time;
        label = "$closure_name (n = $nles, $label)",
        color = color, # dont change this color
    )

end

function plot_posteriori_time(
    outdir,
    closure_name,
    nles,
    Φ,
    projectorders,
    model_index,
    ax,
    color,
)

    if closure_name == "INS_ref"
        postfile = Benchmark.getpostfile(outdir, closure_name, nles, Φ, projectorders[1])
        posttraining = namedtupleload(postfile)
        training_time = [round(posttraining.single_stored_object.comptime; digits = 3)]
    else
        posttraining = loadpost(outdir, closure_name, [nles], [Φ], projectorders)
        training_time =
            map(p -> p.comptime, posttraining) |> vec .|> x -> round(x; digits = 3)
    end

    label = Φ isa FaceAverage ? "FA" : "VA"
    barplot!(
        ax,
        [model_index],
        training_time;
        label = "$closure_name (n = $nles, $label)",
        color = color, # dont change this color
    )

end

function plot_num_parameters(outdir, closure_name, nles, Φ, model_index, ax, color)
    # Load learned parameters
    priortraining = loadprior(outdir, closure_name, [nles], [Φ])

    # parameters count
    θ_length = map(p -> length(p.θ), priortraining)[1]

    label = Φ isa FaceAverage ? "FA" : "VA"
    barplot!(
        ax,
        [model_index],
        [θ_length];
        label = "$closure_name (n = $nles, $label)",
        color = color, # dont change this color
    )
end

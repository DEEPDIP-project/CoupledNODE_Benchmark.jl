
function _update_ax_limits(ax, x, y)
    # Get data limits with extra padding for the legend
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

function _missing_label(ax, label)
    for plt in ax.scene.plots
        if :label in keys(plt.attributes) && plt.attributes[:label][] == label
            return false
        end
    end
    return true
end

function plot_prior(outdir, closure_name, nles, Φ, ax, color, PLOT_STYLES)
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
        y = priortraining[1].hist
    else
        y = priortraining[1].lhist_val
    end

    lines!(
        ax,
        y;
        label = "$closure_name (n = $nles, $label)",
        color = color, # dont change this color
        linestyle = PLOT_STYLES[:prior].linestyle,
        linewidth = PLOT_STYLES[:prior].linewidth,
    )
    if closure_name !== "INS_ref"
        ax = _update_ax_limits(ax, collect(1:length(y)), y)
    end
end

function plot_posteriori(
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
        y = hist
    else
        posttraining = loadpost(outdir, closure_name, [nles], [Φ], projectorders)
        posttraining = loadpost(outdir, closure_name, [nles], [Φ], projectorders)
        y = posttraining[1].lhist_val
    end
    ax.xticks = 1:length(y)  # because y is "iteration", it should be integer
    scatterlines!(
        ax,
        y;
        label = "$closure_name (n = $nles, $label)",
        linestyle = PLOT_STYLES[:post].linestyle,  # should not interpolate between points
        linewidth = PLOT_STYLES[:post].linewidth,
        marker = :circle,
        color = color, # dont change this color
    )
    #if startswith(closure_name,"cnn_")
    if closure_name !== "INS_ref" && length(y) > 1
        ax = _update_ax_limits(ax, collect(1:length(y)), y)
    end
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
    @info "time_indices: $(time_indices)"
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
                @info sizeof(state.u), itime, k, I
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
    # exclude the solutions that have length 0
    @info "----------"
    @info sizeof(solutions.u)
    @info "----------"
    #filtered_u = filter(x -> sizeof(x) > 0, solutions.u)
    #filtered_data = (u = filtered_u, t = solutions.t, itime_max_DIF = solutions.itime_max_DIF)
    #solutions = filtered_data

    setup = getsetup(; params, nles)
    κ = IncompressibleNavierStokes.spectral_stuff(setup).κ
    all_specs = _get_spectra(setup, solutions.u)

    kmax = maximum(κ)

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
        ax = CairoMakie.Axis(subfig; xticks, title = title, xlabel = "κ")

        specs = all_specs[itime][data_index]
        # add No closure
        lines!(
            ax,
            κ,
            specs[2];
            label = "No closure (n = $nles)",
            linestyle = PLOT_STYLES[:no_closure].linestyle,
            linewidth = PLOT_STYLES[:no_closure].linewidth,
            color = PLOT_STYLES[:no_closure].color,
        )

        # add reference
        lines!(
            ax,
            κ,
            specs[1];
            color = PLOT_STYLES[:reference].color,
            linestyle = PLOT_STYLES[:reference].linestyle,
            linewidth = PLOT_STYLES[:reference].linewidth,
            label = "Reference",
        )

        label = Φ isa FaceAverage ? "FA" : "VA"
        lines!(
            ax,
            κ,
            specs[3];
            label = "$closure_name (prior) (n = $nles, $label)",
            linestyle = PLOT_STYLES[:prior].linestyle,
            linewidth = PLOT_STYLES[:prior].linewidth,
            color = color, # dont change this color
        )
        lines!(
            ax,
            κ,
            specs[4];
            label = "$closure_name (post) (n = $nles, $label)",
            linestyle = PLOT_STYLES[:post].linestyle,
            linewidth = PLOT_STYLES[:post].linewidth,
            color = color, # dont change this color
        )
        krange, inertia, slopelabel = _build_inertia_slope(kmax, specs[1], κ)
        lines!(
            ax,
            krange,
            inertia;
            color = PLOT_STYLES[:inertia].color,
            label = slopelabel,
            linestyle = PLOT_STYLES[:inertia].linestyle,
            linewidth = PLOT_STYLES[:inertia].linewidth,
        )
        ax.yscale = log10
        ax.xscale = log2

        if itime == 1
            ax.ylabel = "DCF"
        end
        if itime == length(solutions.t)
            fig[:, itime+1] = Legend(fig, ax)
        end
    end
end

function _convert_to_single_index(i, j, k, dimj, dimk)
    return (i - 1) * dimj * dimk + (j - 1) * dimk + k
end

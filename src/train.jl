
function getdatafile(outdir, nles, filter, seed)
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")
end

function load_data_set(outdir, nles, Φ, seeds, dataproj)
    data = []
    for s in seeds
        filename = getdatafile(outdir, nles, Φ, s)
        if dataproj
            filename = replace(filename, ".jld2" => "_projected.jld2")
        end
        data_i = namedtupleload(filename)
        push!(data, data_i)
    end
    return data
end

function createdata(; params, seed, outdir, backend, dataproj)
    for (nles, Φ) in Iterators.product(params.nles, params.filters)

        filename = getdatafile(outdir, nles, Φ, seed)
        if dataproj
            filename = replace(filename, ".jld2" => "_projected.jld2")
        end
        datadir = dirname(filename)
        ispath(datadir) || mkpath(datadir)

        if isfile(filename)
            @info "Data file $(filename) already exists. Skipping."
            return
        end
        @info "Creating data for" nles Φ seed
    	@info filename

        data = NS.create_les_data_projected(
            #nchunks = 8000;
            nchunks = 100;
            params...,
            rng = Xoshiro(seed),
            backend = backend,
        )
        jldsave(filename; data...)
    end
end

function getpriorfile(outdir, closure_name, nles, filter)
    joinpath(
        outdir,
        "priortraining",
        closure_name,
        splatfileparts(; filter, nles) * ".jld2",
    )
end

function reusepriorfile(reuse, outdir, closure_name)
    reusepath = joinpath(outdir, "priortraining", reuse)
    targetpath = joinpath(outdir, "priortraining", closure_name)
    # If the reuse path exists, copy it to the target path
    if ispath(reusepath)
        @info "Reusing prior training from $(reusepath) to $(targetpath)"
        ispath(targetpath) || mkpath(targetpath)
        for file in readdir(reusepath, join = true)
            @info "Copying prior training file $(file) to $(targetpath)"
            cp(file, joinpath(targetpath, basename(file)); force = true)
        end
    else
        @warn "Reuse path $(reusepath) does not exist. Not reusing prior training."
    end
end

function reusepostfile(reuse, outdir, closure_name)
    reusepath = joinpath(outdir, "posttraining", reuse)
    targetpath = joinpath(outdir, "posttraining", closure_name)
    # If the reuse path exists, copy it to the target path
    if ispath(reusepath)
        @info "Reusing post training from $(reusepath) to $(targetpath)"
        ispath(targetpath) || mkpath(targetpath)
        for file in readdir(reusepath, join = true)
            @info "Copying post training file $(file) to $(targetpath)"
            cp(file, joinpath(targetpath, basename(file)); force = true)
        end
    else
        @warn "Reuse path $(reusepath) does not exist. Not reusing post training."
    end
end


"Load a-priori training results from correct file names."
loadprior(outdir, closure_name, nles, filters) = map(
    splat((nles, Φ) -> load_object(getpriorfile(outdir, closure_name, nles, Φ))),
    Iterators.product(nles, filters),
)

"Train with a-priori loss."
function trainprior(;
    T = nothing,
    params,
    priorseed,
    dns_seeds_train,
    dns_seeds_valid,
    taskid,
    outdir,
    plotdir,
    closure,
    closure_name,
    θ_start,
    st,
    opt,
    batchsize,
    loadcheckpoint = true,
    do_plot = false,
    plot_train = false,
    nepoch,
    dataproj,
    λ = nothing,
    ad_type
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    for Φ in params.filters, nles in params.nles
        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-priori" Φ nles
        else
            # Each task does one training
            @info "Skipping a-priori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        priorfile = getpriorfile(outdir, closure_name, nles, Φ)
        priordir = dirname(priorfile)
        ispath(priordir) || mkpath(priordir)
        figdir = joinpath(plotdir, "priortraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(priorfile))[1] * ".pdf")
        checkfile = join(splitext(priorfile), "_checkpoint")
        batchseed, validseed = splitseed(priorseed, 2) # Same seed for all training setups

        # Read the data in the format expected by the CoupledNODE
        if T === nothing
            T = typeof(params.Re)
        end
        setup = []
        for nl in nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
            push!(setup, Setup(; x = x, Re = params.Re))
        end

        NS = Base.get_extension(CoupledNODE, :NavierStokes)

        # Read the data in the format expected by the CoupledNODE
        data_train = load_data_set(outdir, nles, Φ, dns_seeds_train, dataproj)
        data_valid = load_data_set(outdir, nles, Φ, dns_seeds_valid, dataproj)
        @assert length(nles) == 1 "Only one nles for a-priori training"
        io_train = NS.create_io_arrays_priori(data_train, setup[1], device, T)
        io_valid = NS.create_io_arrays_priori(data_valid, setup[1], device, T)

        if contains(closure_name, "FNO")
            θ = θ_start
        else
            θ = device(copy(θ_start))
        end
        dataloader_prior = NS.create_dataloader_prior(
            io_train;
            batchsize = batchsize,
            rng = Random.Xoshiro(dns_seeds_train[itotal]),
            device = device,
        )
        train_data_priori = dataloader_prior()

        # Trigger the loss once and wrap it for the expected Lux interface
        loss_priori_lux(closure, θ, st, train_data_priori)
        loss(model, param, state, data) = loss_priori_lux(model, param, state, data)

        if loadcheckpoint && isfile(checkfile)
            callbackstate, trainstate, epochs_trained =
                CoupledNODE.load_checkpoint(checkfile)
            nepochs_left = nepoch - epochs_trained
            # Put back the data to the correct device
            if CUDA.functional()
                callbackstate = (
                    θmin = callbackstate.θmin,
                    lhist_val = callbackstate.lhist_val,
                    loss_min = callbackstate.loss_min,
                    lhist_train = callbackstate.lhist_train,
                    lhist_nomodel = callbackstate.lhist_nomodel,
                )
                trainstate = trainstate |> Lux.gpu_device()
            end
        else
            callbackstate = trainstate = nothing
            nepochs_left = nepoch
        end

        callbackstate, callback = NS.create_callback(
            closure,
            θ,
            io_valid,
            loss,
            st;
            callbackstate = callbackstate,
            batch_size = batchsize,
            rng = Xoshiro(batchseed),
            do_plot = do_plot,
            plot_train = plot_train,
            figfile = figfile,
            device = device,
            is_fno = contains(closure_name, "FNO")
        )


        if nepochs_left <= 0
            @info "No epochs left to train."
            continue
        else
            l, trainstate = CoupledNODE.train(
                closure,
                θ,
                st,
                dataloader_prior,
                loss;
                tstate = trainstate,
                ad_type = ad_type,
                nepochs = nepochs_left,
                alg = opt,
                cpu = !CUDA.functional(),
                callback = callback,
                λ = λ,
            )
        end
        # Save on the CPU
        CoupledNODE.save_checkpoint(checkfile, callbackstate, trainstate)

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (;
#            θ = contains(closure_name, "FNO") ? θ |> cpu_device() : Array(θ),
            θ = θ |> cpu_device(),
            comptime = time() - starttime,
            callbackstate.lhist_val,
            callbackstate.lhist_nomodel,
            time_per_epoch = (time() - starttime) / nepochs_left,
        )
        save_object(priorfile, results)
    end
    @info "Finished a-priori training."
end

function getpostfile(outdir, closure_name, nles, filter, projectorder)
    joinpath(
        outdir,
        "posttraining",
        closure_name,
        splatfileparts(; projectorder, filter, nles) * ".jld2",
    )
end

"Load a-posteriori training results from correct file names."
loadpost(outdir, closure_name, nles, filters, projectorders) = map(
    splat((nles, Φ, o) -> load_object(getpostfile(outdir, closure_name, nles, Φ, o))),
    Iterators.product(nles, filters, projectorders),
)

"Train with a-posteriori loss function."
function trainpost(;
    T = nothing,
    params,
    projectorders,
    outdir,
    plotdir,
    taskid,
    postseed,
    dns_seeds_train,
    dns_seeds_valid,
    nunroll,
    nsamples = 1,
    closure,
    closure_name,
    θ_start,
    loadcheckpoint = true,
    st,
    opt,
    nunroll_valid,
    nepoch,
    dt = nothing,
    do_plot = false,
    plot_train = false,
    sensealg = nothing,
    sciml_solver = nothing,
    dataproj,
    λ = nothing,
    multishoot_nt = 0,
    ad_type
)
    device(x) = adapt(params.backend, x)
    itotal = 0
    if nsamples === nothing
        nsamples = 1
    end
    for projectorder in projectorders,
        (ifil, Φ) in enumerate(params.filters),
        (igrid, nles) in enumerate(params.nles)

        itotal += 1
        if isnothing(taskid) || itotal == taskid
            @info "Training a-posteriori" projectorder Φ nles
        else
            # Each task does one training
            @info "Skipping a-posteriori training for iteration $(itotal) != $(taskid)"
            continue
        end
        starttime = time()
        postfile = getpostfile(outdir, closure_name, nles, Φ, projectorder)
        ispath(dirname(postfile)) || mkpath(dirname(postfile))
        figdir = joinpath(plotdir, "posttraining")
        ispath(figdir) || mkpath(figdir)
        figfile = joinpath(figdir, splitext(basename(postfile))[1] * ".pdf")
        checkfile = join(splitext(postfile), "_checkpoint")
        setup = getsetup(; params, nles)
        psolver = default_psolver(setup)
        if T === nothing
            T = typeof(params.Re)
        end

        # Read the data in the format expected by the CoupledNODE
        data_train = load_data_set(outdir, nles, Φ, dns_seeds_train, dataproj)
        data_valid = load_data_set(outdir, nles, Φ, dns_seeds_valid, dataproj)

        NS = Base.get_extension(CoupledNODE, :NavierStokes)
        io_train = NS.create_io_arrays_posteriori(data_train, setup, device, T)
        io_valid = NS.create_io_arrays_posteriori(data_valid, setup, device, T)
        θ = device(copy(θ_start[itotal]))
        dataloader_post = NS.create_dataloader_posteriori(
            io_train;
            nunroll = nunroll,
            nsamples = nsamples,
            rng = Random.Xoshiro(dns_seeds_train[itotal]),
            device = device,
        )

        dudt_nn = NS.create_right_hand_side_with_closure(setup, psolver, closure, st)
        griddims = ntuple(_ -> Colon(), params.D)
        inside = ntuple(_ -> 2:(nles+1), params.D)
        loss = CoupledNODE.create_loss_post_lux(
            dudt_nn,
            griddims,
            inside,
            dt;
            ensemble = nsamples > 1,
            sciml_solver = sciml_solver,
            sensealg = sensealg,
            multiple_shooting = multishoot_nt,
        )

        if loadcheckpoint && isfile(checkfile)
            callbackstate, trainstate, epochs_trained =
                CoupledNODE.load_checkpoint(checkfile)
            nepochs_left = nepoch - epochs_trained
            # Put back the data to the correct device
            if CUDA.functional()
                callbackstate = (
                    θmin = callbackstate.θmin,
                    lhist_val = callbackstate.lhist_val,
                    loss_min = callbackstate.loss_min,
                    lhist_train = callbackstate.lhist_train,
                    lhist_nomodel = callbackstate.lhist_nomodel,
                )
                trainstate = trainstate |> Lux.gpu_device()
            end
        else
            callbackstate = trainstate = nothing
            nepochs_left = nepoch
        end


        # For the callback I am going to use the a-posteriori error estimator
        filename = getdatafile(outdir, nles, Φ, dns_seeds_valid[1])
        if dataproj
            filename = replace(filename, ".jld2" => "_projected.jld2")
        end
        sample = namedtupleload(filename)
        it = 1:(nunroll_valid+1)
        data_cb = (;
            u = selectdim(sample.u, ndims(sample.u), it) |> collect |> device,
            t = sample.t[it],
        )
        tspan = (data_cb.t[1], data_cb.t[end])
        tsave = [nunroll_valid]
        dudt_cb = NS.create_right_hand_side_with_closure_inplace(
            setup, psolver, closure, st)
        loss_cb(_model, pp, _st, _data ) = compute_epost(dudt_cb, sciml_solver, pp , tspan, data_cb, tsave, dt)[1][end]

        callbackstate, callback = NS.create_callback(
            closure,
            θ,
            io_valid,
            loss_cb,
            st;
            callbackstate = callbackstate,
            nunroll = nunroll_valid,
            rng = Xoshiro(postseed),
            do_plot = do_plot,
            plot_train = plot_train,
            figfile = figfile,
            device = device,
        )
        if nepochs_left <= 0
            @info "No epochs left to train."
            continue
        else
            l, trainstate = CoupledNODE.train(
                closure,
                θ,
                st,
                dataloader_post,
                loss;
                ad_type = ad_type,
                tstate = trainstate,
                nepochs = nepochs_left,
                alg = opt,
                cpu = !CUDA.functional(),
                callback = callback,
                λ = λ,
            )
        end
        # Save on the CPU
        CoupledNODE.save_checkpoint(checkfile, callbackstate, trainstate)

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (;
            θ = Array(θ),
            comptime = time() - starttime,
            lhist_val = callbackstate.lhist_val,
            time_per_epoch = (time() - starttime) / nepochs_left,
        )
        save_object(postfile, results)
    end
    @info "Finished a-posteriori training."
end

function compute_eprior(closure, θ, st, x, y)
    y_pred, _ = Lux.apply(closure, x, θ, st)[1:2]
    return norm(y_pred - y) / norm(y)
end

function compute_t_prior_inference(closure, θ, st, x, y, nreps = 1000)
    size_in = size(x)
    T = eltype(x)
    _, t, _, _ = @timed begin
        for _ = 1:nreps
            if x isa CUDA.CuArray
                x = CUDA.rand(T, size_in)
            else
                x = rand(T, size_in)
            end
            Lux.apply(closure, x, θ, st)
        end
    end
    return t/nreps
end


function compute_epost(rhs, sciml_solver, ps, tspan, (u, t), tsave, dt)
    griddims = ((:) for _ = 1:(ndims(u)-2))
    inside = ((2:(size(u, 1)-1)) for _ = 1:(ndims(u)-2))
    x = u[griddims..., :, 1]
    y = u[griddims..., :, 2:end]
    prob = ODEProblem(rhs, x, tspan, ps)
    t0 = time()
    pred = solve(
        prob,
        sciml_solver;
        u0 = x,
        p = ps,
        adaptive = true,
        saveat = Array(t),
        tstops = Array(t),
        tspan = tspan,
        save_start = false,
        dt = dt,
    )
    inf_time = time() - t0

    e = 0.0
    es = []

    for it = 1:size(y, 4)
        yref = y[inside..., :, it]
        ypred = pred[inside..., :, it]

        a = sum(abs2, ypred .- yref)
        b = sum(abs2, yref)

        e += sqrt(a) / sqrt(b)

        if it in tsave
            push!(es, e / (it - 1))
        end
    end

    return es, inf_time

end


function create_test_dns_proj(;
        nchunks = 100,
        D,
        Re,
        lims,
        nles,
        ndns,
        filters,
        tburn,
        tsim,
        savefreq,
        Δt,
        method = RKMethods.RK44(; T = typeof(Re)),
        create_psolver = default_psolver,
        backend = IncompressibleNavierStokes.CPU(),
        icfunc = (setup, psolver, rng) -> random_field(setup, typeof(Re)(-1); psolver, rng),
        processors = (; log = timelogger(; nupdate = 9)),
        rng,
        filename,
        sciml_solver = RK4(),
        kwargs...
)

    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    T = typeof(Re)

    # Build setup and assemble operators
    dns = Setup(; x = ntuple(α -> LinRange(lims..., ndns + 1), D), Re, backend, kwargs...)

    # Since the grid is uniform and identical for x and y, we may use a specialized
    # spectral pressure solver
    psolver = create_psolver(dns)

    # Initial conditions
    ustart = icfunc(dns, psolver, rng)

    while any(u -> any(isnan, u), ustart)
        @warn "Initial conditions contain NaNs. Regenerating..."
        ustart = icfunc(dns, psolver, rng)
    end

    _dns = dns

    # Solve burn-in DNS using INS
    (; u, t), outputs = solve_unsteady(; setup = _dns, ustart, tlims = (T(0), tburn), Δt, psolver)
    ustart = copy(u)
    @info "Burn-in DNS simulation finished"
    any(u -> any(isnan, u), u) && @warn "NaNs after burn-in"


    # After the burn-in I want to solve the DNS both using INS and SciML
    @info "Starting DNS simulation with INS and SciML"
    tdatapoint = collect(T(0):(savefreq * Δt):tsim)
    all_ules_ref = Array{T}(undef, (ndns[1] + 2, ndns[1]+2, D, length(tdatapoint)))
    all_t_ref = Array{T}(undef, (length(tdatapoint)))
    idx = Ref(1)
    stepper = create_stepper(method; setup =_dns, psolver=psolver, u = u, temp = nothing, t = T(0))
    cache = IncompressibleNavierStokes.ode_method_cache(method, _dns)
    while stepper.t <= tsim
        if abs(stepper.t - tdatapoint[idx[]]) < 1e-8
            all_ules_ref[:, :, :, idx[]] .= Array(stepper.u)
            all_t_ref[idx[]] = stepper.t
            idx[] += 1
        end
        stepper = IncompressibleNavierStokes.timestep!(method, stepper, Δt; cache = cache )
    end
    @info "finished INS DNS simulation"

    # And now we try SciML
    u = ustart

    # Define the callback function for the filter
    function condition(u, t, integrator)
        t in tdatapoint && return true
        return false
    end
    all_ules = Array{T}(undef, (ndns[1] + 2, ndns[1]+2, D, length(tdatapoint)))
    all_t = Array{T}(undef, (length(tdatapoint)))
    idx = Ref(1)
    function _callback(integrator)
        ubc = IncompressibleNavierStokes.apply_bc_u(integrator.u, integrator.t, _dns)
        all_ules[:, :, :, idx[]] .= Array(ubc)
        all_t[idx[]] = integrator.t
        idx[] += 1
    end
    cb = DiscreteCallback(condition, _callback)

    # Now use SciML to solve the DNS
    rhs! = NS.create_right_hand_side_inplace(dns, psolver)
    t0 = T(0)
    tfinal = tsim
    dt_chunk = tsim / nchunks
    tchunk = collect(t0:dt_chunk:tfinal)  # Save at the end of each chunk

    u_current = ustart # Initial condition
    prob = ODEProblem(rhs!, u_current, nothing, nothing)

    # Store the data at t=0
    _callback((; u = u_current, t = T(0)))

    any(u -> any(isnan, u), u_current) &&
        @warn "Solution contains NaNs. Probably dt is too large."


    for (i, t_start) in enumerate(tchunk[1:(end - 1)])
        GC.gc()
        if CUDA.functional()
            CUDA.reclaim()
        end
        t_end = tchunk[i + 1]
        tspan_chunk = (t_start, t_end)
        prob = ODEProblem(rhs!, u_current, tspan_chunk, nothing)

        sol = solve(
            prob, sciml_solver; u0 = u_current, p = nothing,
            adaptive = false, dt = Δt, save_end = true, callback = cb,
            tspan = tspan_chunk, tstops = tdatapoint
        )

        u_current = sol.u[end]
    end

    @info "DNS simulation finished"

    any(u -> any(isnan, u), u_current) &&
        @warn "Solution contains NaNs. Probably dt is too large."

    data = (; u = all_ules, uref=all_ules_ref, t = all_t, tref = all_t_ref)
    jldsave(filename; data...)
    @info "Test data comparison stored in $(filename)"
    
end

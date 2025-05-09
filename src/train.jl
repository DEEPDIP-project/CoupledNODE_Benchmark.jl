
function getdatafile(outdir, nles, filter, seed)
    joinpath(outdir, "data", splatfileparts(; seed = repr(seed), filter, nles) * ".jld2")
end

function createdata(; params, seed, outdir, backend)
    @info "Creating DNS trajectory for seed $(repr(seed))"
    filenames = []
    for (nles, Φ) in Iterators.product(params.nles, params.filters)
        f = getdatafile(outdir, nles, Φ, seed)
        datadir = dirname(f)
        ispath(datadir) || mkpath(datadir)
        push!(filenames, f)
    end
    if isfile(filenames[1])
        @info "Data file $(filenames[1]) already exists. Skipping."
        return
    end
    data = create_les_data(;
        params...,
        rng = Xoshiro(seed),
        filenames,
        Δt = params.Δt,
        backend = backend,
    )
    @info(
        "Trajectory info:",
        data[1].comptime / 60,
        length(data[1].t),
        Base.summarysize(data) * 1e-9,
    )
end

function getpriorfile(outdir, closure_name, nles, filter)
    joinpath(
        outdir,
        "priortraining",
        closure_name,
        splatfileparts(; filter, nles) * ".jld2",
    )
end

"Load a-priori training results from correct file names."
loadprior(outdir, closure_name, nles, filters) = map(
    splat((nles, Φ) -> load_object(getpriorfile(outdir, closure_name, nles, Φ))),
    Iterators.product(nles, filters),
)

"Train with a-priori loss."
function trainprior(;
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
        T = eltype(params.Re)
        setup = []
        for nl in nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
            push!(setup, Setup(; x = x, Re = params.Re))
        end

        # Read the data in the format expected by the CoupledNODE
        data_train = []
        for s in dns_seeds_train
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_train, hcat(data_i))
        end
        data_valid = []
        for s in dns_seeds_valid
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_valid, hcat(data_i))
        end
        NS = Base.get_extension(CoupledNODE, :NavierStokes)
        #function this_create_io_arrays(data, setup)
        #    (; dimension, N, Iu) = setup.grid
        #    T = eltype(data[1].t)
        #    D = dimension()
        #    colons = ntuple(Returns(:), D)
        #    fields = map((:u, :c)) do usym
        #        u = map(data) do trajectory
        #            nt = length(trajectory.t)
        #            u = zeros(T, (N .- 2)..., D, nt)
        #            for it = 1:nt, α = 1:D
        #                copyto!(
        #                    view(u, colons..., α, it),
        #                    view(getfield(trajectory, usym), Iu[α], α, it),
        #                )
        #            end
        #            u
        #        end
        #        u = cat(u...; dims = D + 2)
        #        usym => u
        #    end
        #    (; fields...)
        #end
        io_train = NS.create_io_arrays_priori(data_train, setup)
        io_valid = NS.create_io_arrays_priori(data_valid, setup)

        #setup = getsetup(; params, nles)
        #data_train =
        #    map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_train)
        #data_valid =
        #    map(s -> namedtupleload(getdatafile(outdir, nles, Φ, s)), dns_seeds_valid)
        #io_train = this_create_io_arrays(data_train, setup)
        #io_valid = this_create_io_arrays(data_valid, setup)

        θ = device(copy(θ_start))
        dataloader_prior = NS.create_dataloader_prior(
            io_train[itotal];
            batchsize = batchsize,
            rng = Random.Xoshiro(dns_seeds_train[itotal]),
            device = device,
        )
        train_data_priori = dataloader_prior()

        loss_priori_lux(closure, θ, st, train_data_priori)
        loss = loss_priori_lux

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
            io_valid[itotal],
            loss,
            st;
            callbackstate = callbackstate,
            batch_size = batchsize,
            rng = Xoshiro(batchseed),
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
                dataloader_prior,
                loss;
                tstate = trainstate,
                nepochs = nepochs_left,
                alg = opt,
                cpu = !CUDA.functional(),
                callback = callback,
            )
        end
        # Save on the CPU
        CoupledNODE.save_checkpoint(checkfile, callbackstate, trainstate)

        θ = callbackstate.θmin # Use best θ instead of last θ
        results = (;
            θ = Array(θ),
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
    params,
    projectorders,
    outdir,
    plotdir,
    taskid,
    postseed,
    dns_seeds_train,
    dns_seeds_valid,
    nunroll,
    closure,
    closure_name,
    θ_start,
    loadcheckpoint = true,
    st,
    opt,
    nunroll_valid,
    nepoch,
    dt,
    do_plot = false,
    plot_train = false,
    sensealg = nothing,
)
    device(x) = adapt(params.backend, x)
    itotal = 0
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
        # Read the data in the format expected by the CoupledNODE
        T = eltype(params.Re)
        setup = []
        for nl in nles
            x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
            push!(setup, Setup(; x = x, Re = params.Re, params.backend))
        end

        # Read the data in the format expected by the CoupledNODE
        data_train = []
        for s in dns_seeds_train
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_train, hcat(data_i))
        end

        data_valid = []
        for s in dns_seeds_valid
            data_i = namedtupleload(getdatafile(outdir, nles, Φ, s))
            push!(data_valid, hcat(data_i))
        end

        NS = Base.get_extension(CoupledNODE, :NavierStokes)
        io_train = NS.create_io_arrays_posteriori(data_train, setup)
        io_valid = NS.create_io_arrays_posteriori(data_valid, setup)
        θ = device(copy(θ_start[itotal]))
        dataloader_post = NS.create_dataloader_posteriori(
            io_train[itotal];
            nunroll = nunroll,
            rng = Random.Xoshiro(dns_seeds_train[itotal]),
            device = device,
        )

        dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
        loss = create_loss_post_lux(
            dudt_nn;
            sciml_solver = RK4(),
            dt = dt,
            use_cuda = CUDA.functional(),
            sensealg = sensealg,
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

        callbackstate, callback = NS.create_callback(
            closure,
            θ,
            io_valid[itotal],
            loss,
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
                tstate = trainstate,
                nepochs = nepochs_left,
                alg = opt,
                cpu = !CUDA.functional(),
                callback = callback,
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

function compute_epost(rhs, ps, dt, tspan, (u, t), dev)
    griddims = ((:) for _ = 1:(ndims(u)-2))
    x = u[griddims..., :, 1] |> dev
    y = u[griddims..., :, 2:end] |> dev # remember to discard sol at the initial time step
    prob = ODEProblem(rhs, x, tspan, ps)
    t0 = time()
    pred = dev(
        solve(
            prob,
            RK4();
            u0 = x,
            p = ps,
            adaptive = false,
            saveat = Array(t),
            dt = dt,
            tspan = tspan,
        ),
    )
    t = time() - t0
    a = sum(y[griddims..., :, 1:(size(pred, 4)-1)] - pred[griddims..., :, 2:end])
    b = sum(abs2, y[griddims..., :, 1:(size(pred, 4)-1)])
    return mean(sqrt.(a) ./ sqrt.(b)), t
end

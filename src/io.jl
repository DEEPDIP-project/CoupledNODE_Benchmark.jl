
# Define some functions
function read_config(outdir, config_file, backend)
    conf = NS.read_config(config_file)
    closure_name = conf["closure"]["name"]
    model_path = joinpath(outdir, closure_name)

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


function check_necessary_files(outdir, closure_name, nles, filter, projectorder)
    divergence = joinpath(outdir, closure_name, "history_nles=$(nles).jld2")
    energy = joinpath(outdir, closure_name, "solutions_nles=$(nles).jld2")
    post = getpostfile(outdir, closure_name, nles, filter, projectorder)
    prior = getpriorfile(outdir, closure_name, nles, filter)
    incomplete = false

    if !ispath(divergence)
        @warn "Divergence file $divergence not found"
        incomplete = true
    end
    if !ispath(energy)
        @warn "Energy file $energy not found"
        incomplete = true
    end
    if !ispath(post)
        @warn "Posteriori file $post not found"
        incomplete = true
    end
    if !ispath(prior)
        @warn "Prior file $prior not found"
        incomplete = true
    end

    if incomplete
        return false
    else
        return true
    end
end

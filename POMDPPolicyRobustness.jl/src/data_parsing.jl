function worst_vals(df)
    worst_idxs = []
    worst_vals = []
    # worst_sem = []
    for gap_i in unique(df[!,:η_target])
        fdf = filter(:η_target => x->x==gap_i, df)
        vals = fdf[!,:η]
        value_worst,idx_worst = findmax(vals)
        push!(worst_idxs,idx_worst)
        push!(worst_vals,value_worst)
        # push!(worst_sem,fdf[:mc_sem,idx_worst])
    end
    return worst_vals
end

function worst_vals_x(df)
    worst_idxs = []
    worst_vals = []
    # worst_sem = []
    for gap_i in unique(df[!,:x])
        fdf = filter(:x => x->x==gap_i, df)
        vals = fdf[!,:η]
        value_worst,idx_worst = findmax(vals)
        push!(worst_idxs,idx_worst)
        push!(worst_vals,value_worst)
        # push!(worst_sem,fdf[:mc_sem,idx_worst])
    end
    return worst_vals
end

function get_mean(pomdp,mc,rew_mat,depth,runs)
    vec_rew = mc_simulation_vector(mc,rew_mat,POMDPs.discount(pomdp)^0.5,runs;max_depth=depth,verbose=false)./POMDPs.discount(pomdp)^0.5
    mrew = mean(vec_rew)
    @assert length(vec_rew) == runs
    semrew = 3 * std(vec_rew) / sqrt(runs)
    return mrew, semrew
end
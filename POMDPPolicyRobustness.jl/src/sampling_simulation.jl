function istermloop(t_mat::SparseMatrixCSC,r_mat::SparseVector,idx::Int) #Write tests for this
    term = false
    if idx >= obs_start_idx(t_mat.n) #Observations are not terminal
        # @warn "Observations are not terminal. Check index."
        return term
    end
    sps = findall(x->x>0.0,t_mat[:,idx])
    if length(sps) > 1
        return term
    elseif sps[1] != idx
        return term
    elseif r_mat[idx] != 0.0
        return term
    else
        return true
    end
end

function mc_simulation(t_mat::SparseMatrixCSC,r_mat::SparseVector,discount::Float64;max_depth::Int=typemax(Int),verbose::Bool=false,rng=MersenneTwister())
    s_idx = 1
    d = 1
    r_tot = 0.0
    disc = 1.0
    s_list = 1:size(t_mat,2)
    while !istermloop(t_mat,r_mat,s_idx) && d < 2*max_depth #Accounts for observation steps
        rew = r_mat[s_idx]
        r_tot += disc*rew
        s_idx = rand(rng,SparseCat(s_list,t_mat[:,s_idx]))
        if verbose
            println("State: $s_idx, Reward: $rew")
            #something with translating MC states to POMDP States, Observations and Nodes here?
        end
        d+=1
        disc *= discount
    end
    return r_tot
end

function mc_simulation_vector(t_mat::SparseMatrixCSC,r_mat::SparseVector,discount::Float64,n_sims::Int;max_depth::Int=typemax(Int),verbose::Bool=false,rng=MersenneTwister())
    rew_mat = zeros(n_sims)
    @showprogress for i in 1:n_sims
        rew_mat[i] = mc_simulation(t_mat,r_mat,discount;max_depth=max_depth,verbose=verbose,rng=rng)
    end
    return rew_mat
end

function mc_simulation_hist(t_mat::SparseMatrixCSC,r_mat::SparseVector;max_depth::Int=typemax(Int),verbose::Bool=false,rng=MersenneTwister(),s_idx1=1)
    s_idx = s_idx1
    d = 1
    s_list = 1:size(t_mat,2)
    state_hist = Int.(zeros(2*max_depth)) 
    while !istermloop(t_mat,r_mat,s_idx) && d < 2*max_depth #Accounts for observation steps
        s_idx = rand(rng,SparseCat(s_list,t_mat[:,s_idx]))
        if verbose
            println("State: $s_idx")
            #something with translating MC states to POMDP States, Observations and Nodes here?
        end
        state_hist[d] = s_idx
        d+=1
    end
    return state_hist
end

function sample_from_interval_not_uniform(lower::SparseMatrixCSC,upper::SparseMatrixCSC;rng=MersenneTwister())
    # t_mat = deepcopy(lower)
    t_mat = spzeros(size(lower)) + lower
    gap = upper-lower
    gapnz = findnz(gap)
    u_col = unique(gapnz[2])
    for j in u_col
        j_idxs = findall(x->x==j,gapnz[2])
        i_list = shuffle(rng,gapnz[1][j_idxs])
        gap_available = 1-sum(lower[:,j])
        for i in i_list
            if i != last(i_list)
                added_prob = min(rand(rng)*gap[i,j],gap_available)
                t_mat[i,j] += added_prob
                gap_available -= added_prob
                if gap_available <= 0.0
                    break
                end
            else
                t_mat[i,j] += gap_available
            end
        end
    end
    @assert all(lower .<= t_mat .<= upper)
    @assert all( 1-1e-10 .<= sum(t_mat,dims=1) .<= 1+1e-10)
    return t_mat
end

function sample_omax(lower::SparseMatrixCSC,upper::SparseMatrixCSC;rng=MersenneTwister())
    # @assert all(lower .<= upper)
    t_mat = spzeros(size(lower)) + lower
    gap = upper-lower
    rows = rowvals(gap)
    vals = nonzeros(lower)
    for j in axes(gap,1)
        nzis = nzrange(gap, j)
        lownzis = nzrange(lower, j)
        i_list = shuffle(rng,rows[nzis])
        gap_available = 1-sum(vals[lownzis])
        for i in i_list
            added_prob = min(gap[i,j],gap_available)
            t_mat[i,j] += added_prob
            gap_available -= added_prob
            if gap_available <= 0.0
                break
            end
        end
    end
    # @assert all(lower .<= t_mat .<= upper)
    # @assert all( 1-1e-10 .<= sum(t_mat,dims=1) .<= 1+1e-10)
    return t_mat
end

function sample_from_z_omax!(s_pomdp::EvalTabularPOMDP,x::Float64;rng=MersenneTwister(),eps_pres=0.0)
    lb_offset = x
    for a in eachindex(s_pomdp.O2)
        # @show a
        for sp in 1:size(s_pomdp.O2[a],2)
            # @show sp
            po = @view s_pomdp.O2[a][:,sp]
            nz_entries = findnz(po)[1]
            nzl = length(nz_entries)
            gap = nzl*lb_offset #1-sum(lb[:,j])=1-sum(mc .- x [for nzs])
            if nzl > 1 #&& gap > 0
                for (i,o) in enumerate(shuffle(rng,nz_entries))
                    if gap > 0.0
                        # @show 2*x
                        # @show 1-(po[o]-lb_offset)
                        obs_prob = po[o]
                        val = min(gap,2*x,1-(obs_prob-lb_offset)) #In case where x allows greater than 1 Breaks in others?
                        if (val + obs_prob-lb_offset) > (1.0-eps_pres)
                            val -= eps_pres
                        end
                        # @show val
                        po[o] += -lb_offset+val
                        s_pomdp.O[a][sp,o] += -lb_offset+val
                        gap -= val
                    else
                        po[o] += -lb_offset
                        s_pomdp.O[a][sp,o] += -lb_offset
                    end
                end
            end
        end
    end
    # for a in eachindex(s_pomdp.O)
    #     @assert all(s_pomdp.O2[a] .- x .<= s_pomdp.O2[a] .<= s_pomdp.O2[a] .+ x)
    #     @assert all( 1-1e-10 .<= sum(s_pomdp.O2[a],dims=1) .<= 1+1e-10)
    #     @assert all(s_pomdp.O[a] .- x .<= s_pomdp.O[a] .<= s_pomdp.O[a] .+ x)
    #     @assert all( 1-1e-10 .<= sum(s_pomdp.O[a],dims=2) .<= 1+1e-10)
    # end
    return s_pomdp
end
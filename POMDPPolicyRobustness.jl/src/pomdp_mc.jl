function obs_start_idx(mc_size::Int)
    return Int((mc_size-1)/2+2)
end

function pg_children_check(pomdp::POMDP,pg::PolicyGraph)
    passing = true
    for n in eachindex(pg.nodes)
        children = false
        for o in observations(pomdp)
            if haskey(pg.edges,(n,o))
                children = true
            end
        end
        if passing && !children
            @warn "Check for children at all graph nodes fails at:"
            @info "Node: $n"
            passing = false
        elseif !children
            @info "Node: $n"
        end
    end
    return passing
end

function sn_mapping(n_nodes,n_states)
    s_mc = vcat((0,0,0),collect(Iterators.product(1:n_nodes,1:n_states,0:1))...)
    return Dict(s_mc .=> 1:length(s_mc))
end

function rsn_mapping(n_nodes,n_states)
    s_mc = vcat((0,0,0),collect(Iterators.product(1:n_nodes,1:n_states,0:1))...)
    return Dict(1:length(s_mc) .=> s_mc)
end

function rsa_mapping(n_nodes,n_states,pg)
    s_mc = vcat((0,0,0),collect(Iterators.product(1:n_nodes,1:n_states,0:1))...)
    sa_list = [(pg.nodes[s[1]],s[2]) for s in s_mc[2:end]] #Includes placeholder action for initial state
    return Dict(1:length(s_mc) .=> [(pg.nodes[1],s_mc[2]),sa_list...])
end

function unreachable_check(t_mat::SparseMatrixCSC,idx::Int)
    empty_list = true
    if idx != 1
        new_list = findall(x->x>0.0,t_mat[idx,:])
        for s in new_list
            if s!=idx
                if s != 1
                    e_l = unreachable_check(t_mat,s)
                    if e_l == false
                        empty_list = e_l
                    end
                else
                    empty_list = false
                end
            end
        end
    else
        empty_list = false
    end
    return empty_list
end

function pg_to_mc(pomdp::POMDP,s_pomdp::EvalTabularPOMDP,pol_graph::PolicyGraph;up=DiscreteUpdater(pomdp),b0::DiscreteBelief=initialize_belief(up,initialstate(pomdp)))
    if !pg_children_check(pomdp,pol_graph)
        @warn "One or more childless policy graph nodes reached. Consider a finite horizon POMDP?"
    end
    n_nodes = length(pol_graph.nodes)
    all_s = ordered_states(pomdp)
    n_states = length(all_s)
    n_obs = length(s_pomdp.O[1][1,:])
    #Matrix is NxS+1 where the first entry handles the initial belief
    t_mat = spzeros(2*n_nodes*n_states+1,2*n_nodes*n_states+1)
    r_mat = spzeros(2*n_nodes*n_states+1)
    ordered_obs = ordered_observations(pomdp)
    #initial belief to initial state dist
    mapping = sn_mapping(n_nodes,n_states)
    for s in 1:n_states
        ns_map_idx = mapping[(1,s,0)]
        t_mat[ns_map_idx,1] = b0.b[s]
    end
    #all others
    for i in 1:n_nodes
        # println("node $i ====")
        a = actionindex(pomdp,pol_graph.nodes[i])
        oa = s_pomdp.O2[a]
        for s in 1:n_states #Should iterate over s in nz belief instead?
            # println("s $s ++++")
            # @show (i,s,0)
            s_idx = mapping[(i,s,0)]
            if !s_pomdp.isterminal[s]
                r_mat[s_idx] = s_pomdp.R[s,a,1] #Assumes scalar reward
                for sp in 1:n_states
                    tprob = s_pomdp.T[a][sp,s]
                    if tprob > 0.0
                        sp_node_key = (i,sp,1)
                        # println("sp $sp ---")
                        # @show tprob
                        # @show sp_node_key
                        sp_idx = mapping[sp_node_key]
                        t_mat[sp_idx,s_idx] += tprob
                    end
                end
            else
                #If a state is terminal, return to that state #CHECK ME
                t_mat[s_idx,s_idx] = 1.0
            end
            oasp = @view oa[:,s]
            s_idx2 = mapping[(i,s,1)]
            for o in 1:n_obs
                # println("obs $o")
                if oasp[o] > 0.0
                    if haskey(pol_graph.edges,(i,ordered_obs[o]))
                        new_node = pol_graph.edges[(i,ordered_obs[o])]
                        o_node_key = (new_node,s,0)
                        # @show o_node_key
                        # @show s_pomdp.O[a][s,o]
                        t_mat[mapping[o_node_key],s_idx2] += s_pomdp.O[a][s,o]
                        # display(t_mat)
                        #check belief/graph for return to root???
                    end
                end
            end
        end
    end

    # IntervalMDPs Compat - If not transition probability out, just stay in state #Usually due to nodes being unreachable bc observations are not matched. #Cannot simply check for no parent MC nodes bc some states have nodeless parents that will transition to the state.
    col_sums = sum(t_mat,dims=1)
    warn_zeros = false
    warn_normalizing = false
    for i in eachindex(col_sums)
        cs = col_sums[i]
        if cs .== 0.0
            warn_zeros && @warn "Setting zero probability nodes to self transitions."
            warn_zeros = false
            # unreachable_check(t_mat,i) ? nothing : @warn("Not Parentless") #Disable this at some point
            t_mat[i,i] = 1.0
        elseif cs .<= 1.0-eps(cs)
            warn_normalizing && @warn "Normalizing non-one columns."
            warn_normalizing = false
            vals = nonzeros(t_mat)
            for j in nzrange(t_mat, i)
                vals[j] /= cs
            end
        end
    end

    # sum_list = sum(t_mat,dims=1)
    # nonones = vcat(sum_list...) .!= 1.0
    # for i in eachindex(nonones)
    #     if nonones[i]
    #         unreachable_check(t_mat,i) ? nothing : @warn("Not Parentless") #Disable this at some point
    #         t_mat[i,i] = 1.0-sum_list[i]
    #     end
    # end
    return t_mat, r_mat, 1
end

function pg_to_mc(pomdp::POMDP,pol_graph::PolicyGraph;up=DiscreteUpdater(pomdp),b0::DiscreteBelief=initialize_belief(up,initialstate(pomdp))) #Separate these at some point
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(pomdp)
    return pg_to_mc(pomdp,s_pomdp,pol_graph;up=up,b0=b0) 
end

function policy_to_mc(pomdp::POMDP,policy::Policy,depth::Int;up=DiscreteUpdater(pomdp),b0=initialize_belief(up,initialstate(pomdp))) #Separate these at some point
    pol_graph = gen_polgraph(pomdp, policy, b0, depth)
    return pg_to_mc(pomdp,pol_graph;up=up,b0=b0)
end

function isadist(t_mat)
    res = true
    for j in 1:size(t_mat,2)
        col_tot = sum(t_mat[:,j])
        if !isapprox(col_tot,1.0;atol=eps(col_tot))
            res = false
            break
        end
    end
    return res
end
#Product MC Mappings
function sn_mapping_old(n_nodes,n_states)
    s_mc = vcat((0,0),collect(Iterators.product(1:n_nodes,1:n_states))...)
    return Dict(s_mc .=> 1:length(s_mc))
end

function rsn_mapping_old(n_nodes,n_states)
    s_mc = vcat((0,0),collect(Iterators.product(1:n_nodes,1:n_states))...)
    return Dict(1:length(s_mc) .=> s_mc)
end

#Helpers
function add_or_augment!(dict::Dict,key,val::String)
    if haskey(dict,key)
        dict[key] *= "+$val"
    else
        push!(dict, key => val)
    end
end

function params_string(inds::UnitRange{Int})
    my_str = ""
    for i in inds
        my_str *= "-p$i"
    end
    return my_str
end

#Make MC Dictionary - Sticky
function make_prob_dicts(pomdp::POMDP, pol_graph::PolicyGraph;up=DiscreteUpdater(pomdp),b0::DiscreteBelief=initialize_belief(up,initialstate(pomdp)))
    if !pg_children_check(pomdp,pol_graph)
        @warn "One or more childless policy graph nodes reached. Consider a finite horizon POMDP?"
    end
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(pomdp)
    n_nodes = length(pol_graph.nodes)
    all_s = ordered_states(pomdp)
    n_states = length(all_s)
    n_obs = length(s_pomdp.O[1][1,:])
    #Matrix is NxS+1 where the first entry handles the initial belief
    ordered_obs = ordered_observations(pomdp)
    #initial belief to initial state dist
    mapping = sn_mapping_old(n_nodes,n_states)
    t_prob_dict = Dict()
    r_dict = Dict()
    
    for s in 1:n_states
        ns_map_idx = mapping[(1,s)]
        if b0.b[s] > 0.0
            add_or_augment!(t_prob_dict,(1,ns_map_idx),string(b0.b[s]))
        end
    end
    #all others

    param_counter = 1
    param_dict = Dict()
    default_params = Float64[]

    for i in 1:n_nodes
        # println("node $i")
        a = actionindex(pomdp,pol_graph.nodes[i])
        oa = s_pomdp.O2[a]
        for s in 1:n_states #Should iterate over s in nz belief instead?
            ns_map_idx = mapping[(i,s)]
            if !s_pomdp.isterminal[s]
                r_val = s_pomdp.R[s,a,1]
                if r_val != 0.0
                    push!(r_dict,ns_map_idx => string(r_val)) #Assumes scalar reward
                end
                for sp in 1:n_states
                    if s_pomdp.T[a][sp,s] != 0.0
                        # println("sp $sp")
                        oao = @view oa[:,sp]
                        o_count = 1
                        num_nz = length(nzrange(oa,sp))
                        for o in 1:n_obs
                            # println("obs $o")
                            if oao[o] > 0.0
                                if haskey(pol_graph.edges,(i,ordered_obs[o]))
                                    new_node = pol_graph.edges[(i,ordered_obs[o])]
                                    node_key = (new_node,sp)
                                    # t_mat[mapping[node_key],ns_map_idx] += s_pomdp.T[a][sp,s]*s_pomdp.O[a][sp,o]

                                    if !haskey(param_dict,(a,sp,o))
                                        if o_count < num_nz
                                            par_str = "*p$param_counter"
                                            push!(param_dict,(a,sp,o)=>par_str)
                                            push!(default_params,s_pomdp.O[a][sp,o])
                                            param_counter += 1
                                        else
                                            par_str = "*(1.0"*params_string(param_counter-(num_nz-1):(param_counter-1))*")"
                                            push!(param_dict,(a,sp,o)=>par_str) #Iterate over num_nz previous parameters here, maybe a function for this?
                                            # push!(default_params,s_pomdp.O[a][sp,o])
                                        end

                                        add_or_augment!(t_prob_dict,(ns_map_idx,mapping[node_key]),string(s_pomdp.T[a][sp,s])*par_str)
                                    else
                                        add_or_augment!(t_prob_dict,(ns_map_idx,mapping[node_key]),string(s_pomdp.T[a][sp,s])*param_dict[(a,sp,o)])
                                    end
                                    # printn

                                    
                                end
                                o_count += 1
                            end
                        end
                    end
                end
            else
                #If a state is terminal, return to that state
                add_or_augment!(t_prob_dict,(ns_map_idx,ns_map_idx),string(1.0))
            end
        end
    end
    @info  "$(param_counter-1) parameters"
    return t_prob_dict, r_dict, param_dict, default_params
end

#Make MC Dictionary - Non-Sticky
function make_ns_prob_dicts(pomdp::POMDP, pol_graph::PolicyGraph;up=DiscreteUpdater(pomdp),b0::DiscreteBelief=initialize_belief(up,initialstate(pomdp)))
    if !pg_children_check(pomdp,pol_graph)
        @warn "One or more childless policy graph nodes reached. Consider a finite horizon POMDP?"
    end
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(pomdp)
    n_nodes = length(pol_graph.nodes)
    all_s = ordered_states(pomdp)
    n_states = length(all_s)
    n_obs = length(s_pomdp.O[1][1,:])
    #Matrix is NxS+1 where the first entry handles the initial belief
    ordered_obs = ordered_observations(pomdp)
    #initial belief to initial state dist
    mapping = sn_mapping_old(n_nodes,n_states)
    t_prob_dict = Dict()
    r_dict = Dict()
    
    for s in 1:n_states
        ns_map_idx = mapping[(1,s)]
        if b0.b[s] > 0.0
            add_or_augment!(t_prob_dict,(1,ns_map_idx),string(b0.b[s]))
        end
    end
    #all others

    param_counter = 1
    default_params = Float64[]

    for i in 1:n_nodes
        # println("node $i")
        a = actionindex(pomdp,pol_graph.nodes[i])
        oa = s_pomdp.O2[a]
        for s in 1:n_states #Should iterate over s in nz belief instead?
            ns_map_idx = mapping[(i,s)]
            if !s_pomdp.isterminal[s]
                r_val = s_pomdp.R[s,a,1]
                if r_val != 0.0
                    push!(r_dict,ns_map_idx => string(r_val)) #Assumes scalar reward
                end
                for sp in 1:n_states
                    if s_pomdp.T[a][sp,s] != 0.0
                        # println("sp $sp")
                        oao = @view oa[:,sp]
                        o_count = 1
                        num_nz = length(nzrange(oa,sp))
                        for o in 1:n_obs
                            # println("obs $o")
                            if oao[o] > 0.0
                                if haskey(pol_graph.edges,(i,ordered_obs[o]))
                                    new_node = pol_graph.edges[(i,ordered_obs[o])]
                                    node_key = (new_node,sp)
                                    # t_mat[mapping[node_key],ns_map_idx] += s_pomdp.T[a][sp,s]*s_pomdp.O[a][sp,o]
                                    
                                    if o_count < num_nz
                                        par_str = "*p$param_counter"
                                        # push!(param_dict,(a,sp,o)=>par_str)
                                        push!(default_params,s_pomdp.O[a][sp,o])
                                        param_counter += 1
                                    else
                                        par_str = "*(1.0"*params_string(param_counter-(num_nz-1):(param_counter-1))*")"
                                        # push!(param_dict,(a,sp,o)=>par_str) #Iterate over num_nz previous parameters here, maybe a function for this?
                                        # push!(default_params,s_pomdp.O[a][sp,o])
                                    end

                                    add_or_augment!(t_prob_dict,(ns_map_idx,mapping[node_key]),string(s_pomdp.T[a][sp,s])*par_str)

                                end
                                o_count += 1
                            end
                        end
                    end
                end
            else
                #If a state is terminal, return to that state #CHECK ME
                add_or_augment!(t_prob_dict,(ns_map_idx,ns_map_idx),string(1.0))
            end
        end
    end
    @info  "$(param_counter-1) parameters"
    return t_prob_dict, r_dict, default_params
end

#Write product MC (sticky or non-sticky) to file
function write_mc_transition(pomdp::POMDP, pol_graph::PolicyGraph;filename::String="pomdp.pm",up=DiscreteUpdater(pomdp),b0::DiscreteBelief=initialize_belief(up,initialstate(pomdp)), file_check=false, parametric_mode=true, sticky=true)
    if isfile(filename)
        if file_check == true
            throw("Filename exists, disable check to overwrite")
        else
            @warn "Overwriting previous file $filename"
        end
    end
    n_nodes = length(pol_graph.nodes)
    all_s = ordered_states(pomdp)
    n_states = length(all_s)
    n_mc_states = 1+n_nodes*n_states

    if sticky
        @info "Making Sticky"
        t_prob_dict, r_dict, _, default_params = make_prob_dicts(pomdp, pol_graph;up=up,b0=b0)
    else
        @info "Making Non-Sticky"
        t_prob_dict, r_dict, default_params = make_ns_prob_dicts(pomdp, pol_graph;up=up,b0=b0)
    end

    open(filename, "w") do f
        #Name
        write(f, "// $(replace(string(pomdp),"\n"=>" ")) \n ")
        # write(f, "// $(pg) \n")

        #Type
        write(f, "dtmc \n")
        write(f, "\n")

        #Constants
        write(f, "//Parameters \n")
        for (i,p) in enumerate(default_params)
            if parametric_mode #Don't write values to the parameters
                write(f, "const double p$i; \n")
            else
                write(f, "const double p$i=$p; \n")
            end
        end
        write(f, "\n")

        #Module (Transitions)
        write(f, "//Transitions \n")
        write(f, "module pomdp_mc \n")
        write(f, "\t s: [0..$(n_mc_states-1)] init 0; \n")
        #States
        for s in 1:n_mc_states
            first_sp = true
            for sp in 1:n_mc_states
                if haskey(t_prob_dict,(s,sp))
                    if first_sp
                        write(f, "\t [] s=$(s-1) ->  $(t_prob_dict[(s,sp)]): (s'=$(sp-1))")
                        first_sp = false
                    else
                        write(f, "+ $(t_prob_dict[(s,sp)]): (s'=$(sp-1))")
                    end
                end
            end
            !first_sp && write(f, ";\n")
        end
        write(f, "endmodule \n")
        write(f, "\n")

        #Reward
        write(f, "//Rewards \n")
        # write(f, "rewards \"undiscounted\" \n")
        write(f, "rewards \n")
        #States
        for s in 1:n_mc_states
            if haskey(r_dict,s)
                write(f, "\t s=$(s-1) : $(r_dict[s]);\n")
            end
        end
        write(f, "endrewards")
    end
    return default_params
end

#Output Parsing Tools
function fraction_parsing(frac::AbstractString)
    fracs = split(frac,"/")
    fracints = parse.(Int,fracs)
    @assert length(fracints) == 2
    return fracints[1]/fracints[2]
end

function parse_pipe(filename::String)
    val = -Inf
    for l in eachline(filename)
        if startswith(l, "Result")
            # @show l
            substr = split(l," ")[5]
            isfrac = occursin("/",substr)
            if isfrac
                val = fraction_parsing(substr)
            else
                val = parse(Int,substr)
            end
            break
        end
    end
    return val
end

function parse_pipe_with_time(filename::String)
    val = -Inf
    time = 0.0
    for l in eachline(filename)
        if startswith(l, "Result")
            substr = split(l," ")[5]
            isfrac = occursin("/",substr)
            if isfrac
                val = fraction_parsing(substr)
            else
                val = parse(Int,substr)
            end
        end
        if startswith(l, "Time")
            substr = split(l," ")[end]
            time += parse(Float64,substr[1:end-2])
        end
    end
    return val, time
end

#Property Writing to File
function write_props(region_list::String,tol,h;filename="pomdp.props",file_check=false)
    if isfile(filename)
        if file_check == true
            throw("Filename exists, disable check to overwrite")
        else
            @warn "Overwriting previous file $filename"
        end
    end

    open(filename, "w") do f
        my_props = region_list*";"
        write(f, my_props)
    end
end

#Evaluate using STORM, given a file
function eval_STORM(filename::String,default_params::Vector,x::Float64,h::Int;rounding_digits=10,tol=1e-7,verbose=true,eps_pres=0.01)
    verbose && @info "Preserving structure with probability: $eps_pres"
    region_list = ""
    for (i,p) in enumerate(default_params)
        if p == 0.0
            up_clamp = 1-eps_pres
            low_clamp = 0.0
        elseif p == 1.0
            up_clamp = 1.0
            low_clamp = eps_pres
        else
            up_clamp = 1-eps_pres
            low_clamp = eps_pres
        end
            lb = round(clamp(p-x,low_clamp,up_clamp),digits=rounding_digits)
            ub = round(clamp(p+x,low_clamp,up_clamp),digits=rounding_digits)
        if i == length(default_params)
            region_str = "$lb<=p$i<=$ub"
        else
            region_str = "$lb<=p$i<=$ub,"
        end
        region_list *= region_str
    end
    prop_name = filename[7:end-2]*"props"
    src_path = joinpath(dirname(pwd()),"STORMFiles")
    prop_path = joinpath(src_path,prop_name)
    write_props(region_list,tol,h;filename=prop_path)
    my_cmd = ["docker", "run", "--mount", "type=bind,source=$src_path,target=/data","-w","/opt/storm/build/bin","--rm", "-it","--name", "storm", "movesrwth/storm:1.9.0", "storm-pars", "--mode", "feasibility", "--feasibility:method", "pla", "--prism", filename, "--prop", "R=? [C<=$(h+1)]", "--direction", "min", "--region",  filename[1:6]*prop_name,  "--guarantee", string(tol), "abs"] #,  "--timemem"]#,"--exportresult", "/data/myresult.json"]
    all_cmd = Cmd(my_cmd) #,storm_cmd]) #`eval $bash_cmd`#Cmd([bash_cmd])#,storm_cmd])

    if verbose
        if Sys.iswindows()
            @warn "Julia: Verbose STORM outputs not supported in Windows. See 'out.txt'."
            run(pipeline(all_cmd,"out.txt"))
        else
            run(pipeline(all_cmd,`tee out.txt`))
        end
    else
        run(pipeline(all_cmd,"out.txt"))
    end

    # return parse_pipe("out.txt")
    return parse_pipe_with_time("out.txt")
end

#Value of MC using Storm
function get_storm_value(pomdp::POMDP, pol_graph::PolicyGraph, x::Float64, horizon::Int; filename::String="mypomdp.pm",sticky=false,eps_pres=0.01)
    param_vals = write_mc_transition(pomdp,pol_graph;filename=joinpath(dirname(pwd()),"STORMFiles",filename),sticky=sticky)
    return eval_STORM("/data/"*filename,param_vals,x,horizon;eps_pres=eps_pres)
end

function parse_value_and_time(gap,vals)
    return (gap-vals[1],vals[2])
end

##Percent-based Code for finding x
function find_x_pct_storm(pomdp::POMDP, pol_graph::PolicyGraph, horizon::Int, per_deg::Float64; filename::String="mypomdp.pm",sticky=false,verbose=true)
    param_vals = write_mc_transition(pomdp,pol_graph;filename=joinpath(dirname(pwd()),"STORMFiles",filename),sticky=sticky)
    V,t1 = eval_STORM("/data/"*filename,param_vals,0.0,horizon;verbose=verbose)
    @info "First call time: $t1"
    δV = per_deg*abs(V)
    @info "Initial Policy Value is $V"
    @info "Target Value is $(V-δV)"

    res = upper_bisection_search(x->parse_value_and_time(V-δV,eval_STORM("/data/"*filename,param_vals,x,horizon;verbose=verbose)),0.0,1.0;max_iters=1000,eps=1e-7)
    return res[1],res[2]+t1
end

##Value-based Code for finding x
function find_x_storm(pomdp::POMDP, pol_graph::PolicyGraph, horizon::Int, δV::Float64; filename::String="mypomdp.pm",sticky=false,verbose=true)
    param_vals = write_mc_transition(pomdp,pol_graph;filename=joinpath(dirname(pwd()),"STORMFiles",filename),sticky=sticky)
    V,t1 = eval_STORM("/data/"*filename,param_vals,0.0,horizon;verbose=verbose)
    @info "First call time: $t1"
    # δV = V-value
    @info "Initial Policy Value is $V"
    @info "Target Value is $(V-δV)"

    res = upper_bisection_search(x->parse_value_and_time(V-δV,eval_STORM("/data/"*filename,param_vals,x,horizon;verbose=verbose)),0.0,1.0;max_iters=1000,eps=1e-7)
    return res[1],res[2]+t1
end
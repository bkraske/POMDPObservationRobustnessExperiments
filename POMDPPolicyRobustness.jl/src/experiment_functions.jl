##Code for Experiments
#Utilities
function pct_calc(V,V_star)
    return (V_star-V)/abs(V_star)
end
#

## Value Calculation Methods ##
#POMDPs.jl VI Stuff for Infinite Horizon
function make_smdp(mc::SparseMatrixCSC,rew::SparseVector,is::Int,disc::Float64)
    initial_probs = spzeros(size(mc,1))
    initial_probs[is] = 1.0
    return SparseTabularMDP([transpose(mc)],SparseMatrixCSC(rew),initial_probs,Set{Int}(),disc::Float64)
end

function get_value_VI(mc, is, spec)
    disc = IntervalMDP.discount(spec.prop)
    mdp = make_smdp(mc,spec.prop.reward,is,disc)
    spvi_sol =  SparseValueIterationSolver(;max_iterations=typemax(Int64),belres=spec.prop.convergence_eps,include_Q=true,verbose=false)
    V = solve(spvi_sol,mdp).qmat
    return V[1]/disc
end
#

# Get Finite Horizon VI Value from IntervalMDPs.jl
function get_value_udisc(pomdp::POMDP,policy_graph::PolicyGraph,horizon::Int,x::Float64)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)
    spec = make_finite_spec(pomdp,rew,horizon)
    return get_value(mc,is,spec,x)
end

function get_value_udisc(pomdp::POMDP,policy_graph::PolicyGraph,horizon::Int)
    return get_value_udisc(pomdp::POMDP,policy_graph::PolicyGraph,horizon::Int,0.0)
end
#

#Get Values from PGs,POMDPs
function pg_val_on_spomdp_VI(pomdp::POMDP,spomdp::EvalTabularPOMDP,pg::PolicyGraph)
    mc, rew, is = pg_to_mc(pomdp,spomdp,pg)

    spec = make_infinite_spec(pomdp,rew;precision=1e-7)

    V = get_value_VI(mc, is, spec)
    return V
end

function pg_mc_value_VI(pomdp::POMDP,pg::PolicyGraph)
    spomdp = EvalTabularPOMDP(pomdp)
    return pg_val_on_spomdp_VI(pomdp,spomdp,pg)
end
#
##


##Methods for Finding X##
#Simulation Structs
struct MCSim{P,G}
    pomdp::P
    pg::G
    V_star::Float64
    h::Int
end

function MCSim(pomdp::POMDP,pg::PolicyGraph,h::Int)
    MCSim(pomdp,pg,pg_mc_value_VI(pomdp,pg),h)
end
#

#Get x's from η's - infinite
function η_to_x(sim::MCSim,η::Float64)
    x = find_x_pct(sim.pomdp,sim.pg,η)
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η, :x => x)
end

function η_to_x(sim::MCSim,η::Vector{Float64})
    df = η_to_x(sim,η[1])
    for p in η[2:end]
        df = vcat(df,η_to_x(sim,p))
    end
    return df
end

#Get x's from η's - finite
function η_to_x_udisc(sim::MCSim,η::Float64;eps_pres=0.0)
    x = find_x_pct_udisc(sim.pomdp,sim.pg,sim.h,η;eps_pres=eps_pres)
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η, :x => x)
end

function η_to_x_udisc(sim::MCSim,η::Vector{Float64};eps_pres=0.0)
    df = η_to_x_udisc(sim,η[1];eps_pres=eps_pres)
    for p in η[2:end]
        df = vcat(df,η_to_x_udisc(sim,p;eps_pres=eps_pres))
    end
    return df
end
#

#Get x's from η's - finite for STORM
function η_to_x_storm(sim::MCSim,η::Float64;sticky=false,verbose=true)
    x = find_x_pct_storm(sim.pomdp,sim.pg,sim.h,η;sticky=sticky,verbose=verbose)
    @info "STORM time is $(x[2])"
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η, :x => x[1], :STORM_time => x[2])
end

function η_to_x_storm(sim::MCSim,η::Vector{Float64};sticky=false,verbose=true)
    df = η_to_x_storm(sim,η[1];sticky=sticky,verbose=verbose)
    for p in η[2:end]
        df = vcat(df,η_to_x_storm(sim,p;sticky=sticky,verbose=verbose))
    end
    return df
end
#
##

## Validation Methods ##
#Evaluate worst case T resulting from x
function vi_worstT_inf(sim::MCSim,x::Float64,η_target::Float64)
    _, newT, is, rew = worst_case_T_infinite(sim.pomdp,sim.pg,x)
    spec =  make_infinite_spec(sim.pomdp,rew;precision=1e-7)
    valu = get_value_VI(newT, is, spec)
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(valu,sim.V_star), :IVI_Worst => valu)
end

function vi_worstT_inf(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64})
    df = vi_worstT_inf(sim,xs[1],η_targets[1])
    for i in 2:length(xs)
        df = vcat(df,vi_worstT_inf(sim,xs[i],η_targets[i]))
    end
    return df
end

#Evaluate x's on interval over Ts - worst case and random
function vi_mcT_inf(sim::MCSim,lower_mc,upper_mc,is,spec,η_target,x,id_seed)
    mc = sample_omax(lower_mc,upper_mc,rng=MersenneTwister(id_seed))
    value = get_value_VI(mc, is, spec)
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(value,sim.V_star), :IVI_Worst => value)
end

function vi_mcT_inf(sim::MCSim,x::Float64,η_target::Float64,mc::SparseMatrixCSC,spec,is::Int,n_samples::Int)
    lower_mc,upper_mc = PolicyRobustness.matrix_from_bound(mc, x)
    id_seed = rand(UInt)
    df = vi_mcT_inf(sim,lower_mc,upper_mc,is,spec,η_target,x,id_seed)
    @showprogress for _ in 2:n_samples
        id_seed = rand(UInt)
        df = vcat(df,vi_mcT_inf(sim,lower_mc,upper_mc,is,spec,η_target,x,id_seed))
    end
    return df
end

function vi_mcT_inf(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64},n_samples::Int)
    mc, rew, is = pg_to_mc(sim.pomdp,sim.pg)
    spec = make_infinite_spec(sim.pomdp,rew;precision=1e-7)
    @info "x = $(xs[1]) - 1/$(length(η_targets))"
    df = vi_mcT_inf(sim,xs[1],η_targets[1],mc,spec,is,n_samples)
    for i in 2:length(xs)
        @info "x = $(xs[i]) - $i/$(length(η_targets))"
        df = vcat(df,vi_mcT_inf(sim,xs[i],η_targets[i],mc,spec,is,n_samples))
    end
    return df
end

#Generate and evaluate POMDP by omax sampling
function vi_pomdp_from_x_inf(sim::MCSim,mod_spomdp,x,η_target;eps_pres=0.0)
    id_seed = rand(UInt)
    sample_from_z_omax!(mod_spomdp,x;rng=MersenneTwister(id_seed),eps_pres=eps_pres)
    V = pg_val_on_spomdp_VI(sim.pomdp,mod_spomdp,sim.pg)
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(V,sim.V_star), :POMDP_worst => V,  :sample_id => id_seed)
end

function vi_N_pomdp_from_x_inf(sim::MCSim,x,η_target,n_samples)
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)
    mod_spomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)
    df = vi_pomdp_from_x_inf(sim,mod_spomdp,x,η_target)
    for a in eachindex(s_pomdp.O)
        mod_spomdp.O2[a] .= s_pomdp.O2[a]
        mod_spomdp.O[a] .= s_pomdp.O[a]
    end
    @showprogress for _ in 2:n_samples
        df = vcat(df,vi_pomdp_from_x_inf(sim,mod_spomdp,x,η_target))
        for a in eachindex(s_pomdp.O)
            mod_spomdp.O2[a] .= s_pomdp.O2[a]
            mod_spomdp.O[a] .= s_pomdp.O[a]
        end
    end
    return df
end

function vi_pomdps_from_param_inf(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64},n_samples::Int)
    @info "x = $(xs[1]) - 1/$(length(xs))"
    df = vi_N_pomdp_from_x_inf(sim,xs[1],η_targets[1],n_samples)
    for i in 2:length(xs)
        @info "x = $(xs[i]) - $i/$(length(xs))"
        df = vcat(df,vi_N_pomdp_from_x_inf(sim,xs[i],η_targets[i],n_samples))
    end
    return df
end
##
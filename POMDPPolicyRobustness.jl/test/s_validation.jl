include("instantiate_packages.jl")
using Pkg
Pkg.activate(".")

using PolicyRobustness
using POMDPs
using POMDPTools
using SparseArrays
using LinearAlgebra
using IntervalMDP
using Random
using ProgressMeter
using POMDPPolicyGraphs

Pkg.activate("test")
using POMDPModels
using RockSample
using FiniteHorizonPOMDPs
using Test
using Statistics
using Plots
using DataFrames
using Latexify
using JLD2
using NativeSARSOP
# import SARSOP
using CSV
using Dates
today_date = Dates.format(now(),"yyyymmdd_HHmmss")
function display_and_csv(res,date,name)
    display(res[1])
    display(res[2])
    @info "Worst POMDP"
    display(worst_vals(res[3]))
    for i in 1:3
        CSV.write(date*name*string(i)*".csv",res[i])
    end
end

function soln_params(pomdp::POMDP,h::Int)
    # pol = solve(SARSOPSolver(precision=1e-5),pomdp)
    pol = solve(SARSOPSolver(;precision=1e-5,max_time=typemax(Float64),verbose=true,max_steps=typemax(Int)),pomdp)
    up = DiscreteUpdater(pomdp)
    b0 = initialize_belief(up,initialstate(pomdp))
    pol_graph = PolicyRobustness.gen_polgraph(pomdp, pol, b0, h;store_beliefs=true)
    return pol,up,b0,pol_graph
end

function mc_sims(pomdp,pol_graph,h;runs=10000)
    b0 = initialize_belief(DiscreteUpdater(pomdp),initialstate(pomdp))
    pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(pomdp, PolicyRobustness.EvalTabularPOMDP(pomdp), pol_graph; tolerance=1e-7), b0)[1]

    up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
    bel02 = initialize_belief(up2,initialstate(pomdp))
    
    simlist2 = [Sim(pomdp, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
    mc_res_raw2 = run(simlist2) do sim, hist
        return [:disc_rew => undiscounted_reward(hist)]
    end
    # @show pg_val
    @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
    @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
    return mc_res2,mc_res_sem2
end

##Test Full Optimization
function soln_params2(pomdp::POMDP,h::Int;precision=1e-5)
    # pol = solve(SARSOP.SARSOPSolver(precision=precision),pomdp)
    pol = solve(SARSOPSolver(;precision=precision,max_time=typemax(Float64),verbose=true,max_steps=typemax(Int)),pomdp)
    up = DiscreteUpdater(pomdp)
    b0 = initialize_belief(up,initialstate(pomdp))
    s_pomdp = EvalTabularPOMDP(pomdp)
    pol_graph = PolicyRobustness.gen_polgraph(pomdp, s_pomdp, pol, b0, h; store_beliefs=true)
    return pol,up,b0,s_pomdp,pol_graph
end

function eval_x_STORM(sim::MCSim,x::Float64,η_target::Float64,sticky::Bool,filename::String;verbose=true)
    param_vals = write_mc_transition(sim.pomdp,sim.pg;filename=joinpath(dirname(pwd()),"STORMFiles",filename),sticky=sticky)
    valu,_ = eval_STORM("/data/"*filename,param_vals,x,sim.h;verbose=verbose)
    V_star,_ = eval_STORM("/data/"*filename,param_vals,0.0,sim.h;verbose=verbose) #Note - don't use the sim-stored value as it will differ
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(valu,V_star), :PLA_Worst => valu)
end

function eval_x_STORM(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64};sticky=true,filename::String="mypomdp.pm",verbose=true)
    df = eval_x_STORM(sim,xs[1],η_targets[1],sticky,filename;verbose=verbose)
    for i in 2:length(xs)
        df = vcat(df,eval_x_STORM(sim,xs[i],η_targets[i],sticky,filename;verbose=verbose))
    end
    return df
end

function get_pomdp_from_x_storm(sim::MCSim,mod_spomdp,x,η_target,v_star;eps_pres=0.01)
    id_seed = rand(UInt)
    sample_from_z_omax!(mod_spomdp,x;rng=MersenneTwister(id_seed),eps_pres=eps_pres)
    mc, rew, is = pg_to_mc(sim.pomdp,mod_spomdp,sim.pg)
    spec = make_finite_spec(sim.pomdp,rew,sim.h)
    V = get_value(mc, is, spec, 0.0) #Better method for this???
    return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(V,v_star), :PLA_worst => V,  :sample_id => id_seed)
end

function get_pomdps_from_x_storm(sim::MCSim,x,η_target,n_samples;eps_pres=0.01)
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)
    mod_spomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)

    mc, rew, is = pg_to_mc(sim.pomdp,s_pomdp,sim.pg)
    spec = make_finite_spec(sim.pomdp,rew,sim.h)
    v_star = get_value(mc, is, spec, 0.0) #Better method for this???

    df = get_pomdp_from_x_storm(sim,mod_spomdp,x,η_target,v_star;eps_pres=eps_pres)
    for a in eachindex(s_pomdp.O)
        mod_spomdp.O2[a] .= s_pomdp.O2[a]
        mod_spomdp.O[a] .= s_pomdp.O[a]
    end
    @showprogress for _ in 2:n_samples
        df = vcat(df,get_pomdp_from_x_storm(sim,mod_spomdp,x,η_target,v_star;eps_pres=eps_pres))
        for a in eachindex(s_pomdp.O)
            mod_spomdp.O2[a] .= s_pomdp.O2[a]
            mod_spomdp.O[a] .= s_pomdp.O[a]
        end
    end
    return df
end

function get_pomdps_from_param_storm(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64},n_samples::Int;eps_pres=0.01)
    @info "x = $(xs[1]) - 1/$(length(xs))"
    @info "Using Sim horizon $(sim.h)"
    df = get_pomdps_from_x_storm(sim,xs[1],η_targets[1],n_samples;eps_pres=eps_pres)
    for i in 2:length(xs)
        @info "x = $(xs[i]) - $i/$(length(xs))"
        df = vcat(df,get_pomdps_from_x_storm(sim,xs[i],η_targets[i],n_samples;eps_pres=eps_pres))
    end
    return df
end

function storm_sim(pomdp,param_set,runs,horizon;sticky=false,verbose=true)
    param_set = collect(param_set)
    _,_,_,_,pomdp_pg = soln_params2(pomdp,horizon)
    pomdp_sim = PolicyRobustness.MCSim(pomdp,pomdp_pg,horizon)
    x_res = η_to_x_storm(pomdp_sim,collect(param_set);sticky=sticky,verbose=verbose)
    xs = x_res[!,:x]
    df1 = eval_x_STORM(pomdp_sim,xs,param_set;sticky=sticky,filename="mypomdp.pm",verbose=verbose)
    df2 = get_pomdps_from_param_storm(pomdp_sim,xs,param_set,runs)
    return x_res, df1, df2
end

function just_z_storm_sim(pomdp,param_set,runs,horizon,x_res)
    param_set = collect(param_set)
    _,_,_,_,pomdp_pg = soln_params2(pomdp,horizon)
    pomdp_sim = PolicyRobustness.MCSim(pomdp,pomdp_pg,horizon)
    xs = x_res[!,:x]
    df2 = get_pomdps_from_param_storm(pomdp_sim,xs,param_set,runs)
    return df2
end

function udisc_sim(pomdp,param_set,runs,horizon)
    param_set = collect(param_set)
    _,_,_,_,pomdp_pg = soln_params2(pomdp,horizon)
    pomdp_sim = PolicyRobustness.MCSim(pomdp,pomdp_pg,horizon)
    x_res = η_to_x_udisc(pomdp_sim,collect(param_set);eps_pres=0.01)
    # xs = x_res[!,:x]
    # df1 = vi_worstT_inf(pomdp_sim,xs,collect(param_set))
    # df2 = vi_mcT_inf(pomdp_sim,xs,collect(param_set),runs)
    # df3 = vi_pomdps_from_param_inf(pomdp_sim,xs,collect(param_set),runs)
    # return x_res,df1,df2,df3
    return x_res
end

function undisc_val(pomdp,pg,x,h;eps_pres=0.01)
    mc, rew, is = pg_to_mc(pomdp,pg)
    t_spec = PolicyRobustness.make_finite_spec(pomdp,rew,h)
    return get_value(mc, is, t_spec, x; eps_pres=eps_pres)
end

function undisc_val(pomdp,x,h)
    _,_,_,_,pomdp_pg = soln_params2(pomdp,h)
    return undisc_val(pomdp,pomdp_pg,x,h)
end

function udisc_x_to_finite_val(pomdp,df,horizon)
    _,_,_,_,pomdp_pg = soln_params2(pomdp,horizon)
    vals = []
    for x in df[!,:x]
        push!(vals,pct_calc( undisc_val(pomdp,pomdp_pg,x,horizon),undisc_val(pomdp,pomdp_pg,0.0,horizon)))
    end
    return vals
end

#Tiger
h = 20 #h=3 for nonsticky
tiger = TigerPOMDP()
POMDPs.discount(pomdp::FiniteHorizonPOMDPs.FixedHorizonPOMDPWrapper{Bool, Int64, Bool, TigerPOMDP}) = 0.9999
ht = h+1
tiger_pol,tiger_up,tiger_b0,tiger_pol_graph = soln_params(tiger,50)

#Finding X
tiger_param_set = 0.05:0.2:1.0 #[0.05]
tiger_runs = 1000
t0 = time()
tigerdf1 = storm_sim(tiger,tiger_param_set,tiger_runs,ht;sticky=true,verbose=true)
tigerdt = time()-t0 #Includes starting Docker.
# @save "tiger_strm_6_19.jld2" tigerdf1
@info tigerdt
display(round.(tigerdf1[2][!,:x],digits=4))
display(round.(tigerdf1[2][!,:η],digits=4))
display(round.(worst_vals(tigerdf1[3]),digits=4))
display_and_csv(tigerdf1,today_date,"stigerdf1")
# @load "tiger_strm_6_19.jld2"
# tiger_new_pomdps_df = just_z_storm_sim(tiger,tiger_param_set,tiger_runs,ht,tigerdf1[1])

##RS
h = 20
rs = RockSamplePOMDP(map_size = (3,3),rocks_positions = [(1,1), (2,2), (3,3)],)
POMDPs.discount(pomdp::FiniteHorizonPOMDPs.FixedHorizonPOMDPWrapper{RSState{3}, Int64, Int64, RockSamplePOMDP{3}}) = 0.9999
ht = h+1
_,_,_,_,rs_pol_graph = soln_params2(rs,ht)

#Finding X
rs_param_set = 0.05:0.2:1.0 #[0.05]
rs_runs = 1000
t0 = time()
rsdf1 = storm_sim(rs,rs_param_set,rs_runs,ht;sticky=true,verbose=true)
rsdt = time()-t0 #Includes starting Docker.
# @save "rs_strm_6_19.jld2" rsdf1
@info rsdt
display(round.(rsdf1[2][!,:x],digits=4))
display(round.(rsdf1[2][!,:η],digits=4))
display(round.(worst_vals(rsdf1[3]),digits=4))
display_and_csv(rsdf1,today_date,"srsdf1")
# @load "rs_strm_6_19.jld2"
# rs_new_pomdps_df = just_z_storm_sim(rs,rs_param_set,rs_runs,ht,rsdf1[1])


##Baby
h = 30
baby = BabyPOMDP()
POMDPs.discount(pomdp::FiniteHorizonPOMDPs.FixedHorizonPOMDPWrapper{Bool, Int64, Bool, TigerPOMDP}) = 0.9
ht = h+1

#Finding X
baby_param_set = 0.05:0.2:1.0 #[0.05]
baby_runs = 1000
t0 = time()
bbdf1 = storm_sim(baby,baby_param_set,baby_runs,ht;sticky=true,verbose=false)
bbdt = time()-t0 #Includes starting Docker.
# @save "bb_strm_6_19.jld2" bbdf1
@info bbdt
display(round.(bbdf1[2][!,:x],digits=4))
display(round.(bbdf1[2][!,:η],digits=4))
display(round.(worst_vals(bbdf1[3]),digits=4))
display_and_csv(bbdf1,today_date,"sbbdf1")
# @load "bb_strm_6_19.jld2"
# bb_new_pomdps_df = just_z_storm_sim(baby,baby_param_set,baby_runs,ht,bbdf1[1])
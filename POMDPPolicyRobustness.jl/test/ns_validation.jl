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
# using SARSOP
using Test
using Statistics
using Plots
using DataFrames
using Latexify
using JLD2
using NativeSARSOP
using CSV
using Dates
today_date = Dates.format(now(),"yyyymmdd_HHmmss")
function display_and_csv(res,date,name)
    display(res[1])
    @info "Worst MC"
    display(worst_vals(res[2]))
    @info "Worst POMDP"
    display(worst_vals(res[3]))
    for i in 1:3
        CSV.write(date*name*string(i)*".csv",res[i])
    end
end


#Get Solution Parameters
function soln_params(pomdp::POMDP,h::Int;precision=1e-5)
    pol = solve(SARSOPSolver(;precision=precision,max_time=typemax(Float64),verbose=true,max_steps=typemax(Int)),pomdp)
    up = DiscreteUpdater(pomdp)
    b0 = initialize_belief(up,initialstate(pomdp))
    s_pomdp = EvalTabularPOMDP(pomdp)
    pol_graph = PolicyRobustness.gen_polgraph(pomdp, s_pomdp, pol, b0, h; store_beliefs=true)
    return pol,up,b0,s_pomdp,pol_graph
end

function sim_set(pomdp,param_set,runs,horizon)
    param_set = collect(param_set)
    _,_,_,_,pomdp_pg = soln_params(pomdp,horizon)
    # return pomdp_pg
    pomdp_sim = MCSim(pomdp,pomdp_pg,horizon)
    x_res = η_to_x(pomdp_sim,param_set)
    xs = x_res[!,:x]
    df1 = vi_worstT_inf(pomdp_sim,xs,param_set)
    df2 = vi_mcT_inf(pomdp_sim,xs,param_set,runs)
    df3 = vi_pomdps_from_param_inf(pomdp_sim,xs,param_set,runs)
    return df1,df2,df3
end

function sim_set(pomdp,pomdp_pg,param_set,runs,horizon)
    param_set = collect(param_set)
    pomdp_sim = MCSim(pomdp,pomdp_pg,horizon)
    x_res = η_to_x(pomdp_sim,param_set)
    xs = x_res[!,:x]
    df1 = vi_worstT_inf(pomdp_sim,xs,param_set)
    df2 = vi_mcT_inf(pomdp_sim,xs,param_set,runs)
    df3 = vi_pomdps_from_param_inf(pomdp_sim,xs,param_set,runs)
    return df1,df2,df3
end

function sim_set_z(pomdp,param_set,runs,horizon)
    param_set = collect(param_set)
    _,_,_,_,pomdp_pg = soln_params(pomdp,horizon)
    pomdp_sim = MCSim(pomdp,pomdp_pg,horizon)
    x_res = η_to_x(pomdp_sim,param_set)
    xs = x_res[!,:x]
    df3 = get_pomdps_from_param(pomdp_sim,xs,param_set,runs)
    return df3
end

#Basic Runs
#Tiger
tiger = TigerPOMDP()
tiger_param_set = 0.05:0.2:1.0
tiger_horizon = 100
tiger_runs = 100_000
tigerdfs = sim_set(tiger,tiger_param_set,tiger_runs,tiger_horizon)
display_and_csv(tigerdfs,today_date,"tigerdfs")
# @save "tigeres_inf_6_19.jld2" tigerdfs

#Baby
baby = BabyPOMDP()
baby_param_set = 0.05:0.2:1.0
baby_horizon = 31
baby_runs = 200_000
babydfs = sim_set(baby,baby_param_set,baby_runs,baby_horizon)
display_and_csv(babydfs,today_date,"babydfs")
# @save "babyres_inf_6_19.jld2" babydfs

#RS
rs = RockSamplePOMDP()
rs_param_set = 0.05:0.2:1.0
rs_horizon = 31
rs_runs = 100_000
rsdfs = sim_set(rs,rs_param_set,rs_runs,rs_horizon)
display_and_csv(rsdfs,today_date,"rsdfs")
# @save "rsres_inf_6_19.jld2" rsdfs
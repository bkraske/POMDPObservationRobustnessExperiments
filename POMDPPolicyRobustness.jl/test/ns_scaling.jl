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
    display(res)
    display(res[!,:x])
    CSV.write(date*name*".csv",res)
    CSV.write(date*name*"_x"*".csv",res[!,:x])
end


#Get Solution Parameters
function soln_params(pomdp::POMDP,h::Int;precision=1e-5)
    pol = solve(SARSOPSolver(;precision=precision,max_time=typemax(Float64),verbose=true,max_steps=1000),pomdp)
    up = DiscreteUpdater(pomdp)
    b0 = initialize_belief(up,initialstate(pomdp))
    s_pomdp = EvalTabularPOMDP(pomdp)
    pol_graph = PolicyRobustness.gen_polgraph(pomdp, s_pomdp, pol, b0, h; store_beliefs=true)
    return pol,up,b0,s_pomdp,pol_graph
end

function make_diagonal_rocks(sz)
    rock_locs = [(1,1)]
    for n_r in 2:sz[2]
        push!(rock_locs,(n_r,n_r))
    end
    return RockSamplePOMDP(map_size=(sz[1],sz[1]),rocks_positions=rock_locs)
end

function rs_scale_run(sz,horizon,param;rng=MersenneTwister(5))
    # rs_i = fixhorizon(RockSamplePOMDP(sz...,rng),horizon-1)
    # rs_i = RockSamplePOMDP(sz...,rng)
    rock_locs = [(1,1)]
    for n_r in 2:sz[2]
        push!(rock_locs,(n_r,n_r))
    end
    rs_i = RockSamplePOMDP(map_size=(sz[1],sz[1]),rocks_positions=rock_locs)
    @show rs_i.rocks_positions
    rs_pol,rs_up,rs_b0,rsspomdp,rs_pg_i = soln_params(rs_i,horizon)
    @info length(rs_pg_i.nodes)
    rs_sim_i = MCSim(rs_i,rs_pg_i,horizon)
    t = time()
    x_res = η_to_x(rs_sim_i,param)
    @show dt = time()-t
    return DataFrame(:sz=>sz,:time=>dt,:pg_nodes=>length(rs_pg_i.nodes),:pg_edges=>length(rs_pg_i.edges),:states=>size(ordered_states(rs_i)),:mc_states=>size(ordered_states(rs_i))*length(rs_pg_i.edges)+1,:η=>param,:x=>x_res)
end

function rs_scale_run(pomdp::POMDP,pg::PolicyGraph,horizon,param)
    @info pomdp.rocks_positions
    @info length(pg.nodes)
    rs_sim_i = MCSim(pomdp,pg,horizon)
    t = time()
    x_res = η_to_x(rs_sim_i,param)
    @show dt = time()-t
    return DataFrame(:sz=>pomdp.map_size,:time=>dt,:pg_nodes=>length(pg.nodes),:pg_edges=>length(pg.edges),:states=>size(ordered_states(pomdp)),:mc_states=>(2*size(ordered_states(pomdp))[1]*length(pg.nodes))+1,:η=>param,:x=>x_res)
end

function rs_scaling_runs(rs_sz,horizon,param;rng=MersenneTwister(5))
    mydf = rs_scale_run(rs_sz[1],horizon,param;rng=MersenneTwister(5))
    for sz in rs_sz[2:end]
        mydf = vcat(mydf,rs_scale_run(sz,horizon,param;rng=MersenneTwister(5)))
    end
    return mydf
end

function tiger_scale_run(pomdp::POMDP,pg::PolicyGraph,horizon,param)
    @info pomdp.horizon
    @info length(pg.nodes)
    rs_sim_i = MCSim(pomdp,pg,horizon)
    t = time()
    x_res = η_to_x_udisc(rs_sim_i,param)
    @show dt = time()-t
    return DataFrame(:sz=>pomdp.horizon,:time=>dt,:pg_nodes=>length(pg.nodes),:pg_edges=>length(pg.edges),:states=>size(ordered_states(pomdp)),:mc_states=>(2*size(ordered_states(pomdp))[1]*length(pg.nodes))+1,:η=>param,:x=>x_res)
end

# scaling_runs = rs_scaling_runs(rs_sz,25,0.2;rng=MersenneTwister(5))
# @save "rs_scaling_2_9.jld2" scaling_runs

####
η=0.1
h=35
####

pomdp33 = make_diagonal_rocks((3,3))
pol33,up33,b033,s_pomdp33,pg33=soln_params(pomdp33,h;precision=1e-5)
res33 = rs_scale_run(pomdp33,pg33,h,η)
display_and_csv(res33,today_date,"res33")
# @save "rs33_6_22.jld2" res33

pomdp44 = make_diagonal_rocks((4,4))
pol44,up44,b044,s_pomdp44,pg44=soln_params(pomdp44,h;precision=1e-5)
res44 = rs_scale_run(pomdp44,pg44,h,η)
display_and_csv(res44,today_date,"res44")
# @save "rs44_6_22.jld2" res44

pomdp55 = make_diagonal_rocks((5,5))
pol55,up55,b055,s_pomdp55,pg55=soln_params(pomdp55,h;precision=1e-5)
res55 = rs_scale_run(pomdp55,pg55,h,η)
display_and_csv(res55,today_date,"res55")
# @save "rs55_6_22.jld2" res55

pomdp66 = make_diagonal_rocks((6,6))
pol66,up66,b066,s_pomdp66,pg66=soln_params(pomdp66,h;precision=1e-5)
res66 = rs_scale_run(pomdp66,pg66,h,η)
display_and_csv(res66,today_date,"res66")
# @save "rs66_6_22.jld2" res66

pomdp77 = make_diagonal_rocks((7,7))
pol77,up77,b077,s_pomdp77,pg77=soln_params(pomdp77,h;precision=1e-5)
res77 = rs_scale_run(pomdp77,pg77,h,η)
display_and_csv(res77,today_date,"res77")
# @save "rs77_6_22.jld2" res77

pomdp88 = make_diagonal_rocks((8,8))
pol88,up88,b088,s_pomdp88,pg88=soln_params(pomdp88,h;precision=1e-5)
res88 = rs_scale_run(pomdp88,pg88,h,η)
display_and_csv(res88,today_date,"res88")
# @save "rs88_6_22.jld2" res88

##Tiger Scaling with Horizon
h=3
pomdp3 = fixhorizon( TigerPOMDP(),h)
pol3,up3,b03,s_pomdp3,pg3=soln_params(pomdp3,h+1;precision=1e-5);
res3 = tiger_scale_run(pomdp3,pg3,h+1,η)
display_and_csv(res3,today_date,"res3")
# @save "t3_6_20.jld2" res3

h=4
pomdp4 = fixhorizon( TigerPOMDP(),h)
pol4,up4,b04,s_pomdp4,pg4=soln_params(pomdp4,h+1;precision=1e-5);
res4 = tiger_scale_run(pomdp4,pg4,h+1,η)
display_and_csv(res4,today_date,"res4")
# @save "t4_6_20.jld2" res4

h=5
pomdp5 = fixhorizon( TigerPOMDP(),h)
pol5,up5,b05,s_pomdp5,pg5=soln_params(pomdp5,h+1;precision=1e-5);
res5 = tiger_scale_run(pomdp5,pg5,h+1,η)
display_and_csv(res5,today_date,"res5")
# @save "t5_6_20.jld2" res5

h=6
pomdp6 = fixhorizon( TigerPOMDP(),h)
pol6,up6,b06,s_pomdp6,pg6=soln_params(pomdp6,h+1;precision=1e-5);
res6 = tiger_scale_run(pomdp6,pg6,h+1,η)
display_and_csv(res6,today_date,"res6")
# @save "t6_6_20.jld2" res6

h=7
pomdp7 = fixhorizon( TigerPOMDP(),h)
pol7,up7,b07,s_pomdp7,pg7=soln_params(pomdp7,h+1;precision=1e-5);
res7 = tiger_scale_run(pomdp7,pg7,h+1,η)
display_and_csv(res7,today_date,"res7")
# @save "t7_6_20.jld2" res7
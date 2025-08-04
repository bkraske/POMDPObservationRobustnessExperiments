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
    display(res[!,:x][1])
    CSV.write(date*name*".csv",res)
    CSV.write(date*name*"_x"*".csv",res[!,:x][1])
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
    sz_map = sz[1]
    sz_rocks = sz[2]
    for n_r in 2:sz_map
        if n_r <= sz_rocks
            push!(rock_locs,(n_r,n_r))
        end
    end
    if sz_rocks > sz_map
        for n_r in 1:sz_rocks-sz_map
            push!(rock_locs,(n_r,sz_map-n_r+1))
        end
    end
    return RockSamplePOMDP(map_size=(sz[1],sz[1]),rocks_positions=rock_locs)
end

function make_diagonal_rocks_ul(sz)
    sz_map = sz[1]
    sz_rocks = sz[2]
    rock_locs = [(sz_map,sz_map)]
    count = 1
    for n_r in sz_map-1:-1:1
        if count < sz_rocks
            count += 1
            push!(rock_locs,(n_r,n_r))
        end
    end
    if sz_rocks > sz_map
        for n_r in 1:sz_rocks-sz_map
            push!(rock_locs,(n_r,sz_map-n_r+1))
        end
    end
    return RockSamplePOMDP(map_size=(sz[1],sz[1]),rocks_positions=rock_locs)
end

function rs_scale_run_storm(pomdp::POMDP,pg::PolicyGraph,horizon,param;sticky=true,verbose=true)
    @info pomdp.rocks_positions
    @info length(pg.nodes)
    rs_sim_i = MCSim(pomdp,pg,horizon)
    t = time()
    x_res = η_to_x_storm(rs_sim_i,param;sticky=sticky,verbose=verbose)
    @show dt = time()-t
    return DataFrame(:sz=>pomdp.map_size,:time=>dt,:pg_nodes=>length(pg.nodes),:pg_edges=>length(pg.edges),:states=>size(ordered_states(pomdp)),:mc_states=>(2*size(ordered_states(pomdp))[1]*length(pg.nodes))+1,:η=>param,:x=>x_res)
end

function tiger_scale_run_storm(pomdp::POMDP,pg::PolicyGraph,horizon,param;sticky=true,verbose=true)
    @info pomdp.horizon
    @info length(pg.nodes)
    rs_sim_i = MCSim(pomdp,pg,horizon)
    t = time()
    x_res = η_to_x_storm(rs_sim_i,param;sticky=sticky,verbose=verbose)
    @show dt = time()-t
    return DataFrame(:sz=>pomdp.horizon,:time=>dt,:pg_nodes=>length(pg.nodes),:pg_edges=>length(pg.edges),:states=>size(ordered_states(pomdp)),:mc_states=>(2*size(ordered_states(pomdp))[1]*length(pg.nodes))+1,:η=>param,:x=>x_res)
end

# scaling_runs = rs_scaling_runs(rs_sz,25,0.2;rng=MersenneTwister(5))
# @save "rs_scaling_2_9.jld2" scaling_runs

####
η=0.1
h=35
####

pomdp22 = make_diagonal_rocks((2,2))
pol22,up22,b022,s_pomdp22,pg22=soln_params(pomdp22,h;precision=1e-5)
res22 = rs_scale_run_storm(pomdp22,pg22,h,η,sticky=true,verbose=false)
display_and_csv(res22,today_date,"sres22")
# @save "rs22_strm_6_19.jld2" res22

pomdp33 = make_diagonal_rocks((3,3))
pol33,up33,b033,s_pomdp33,pg33=soln_params(pomdp33,h;precision=1e-5)
res33 = rs_scale_run_storm(pomdp33,pg33,h,η,sticky=true,verbose=false)
display_and_csv(res33,today_date,"sres33")
# @save "rs33_strm_6_19.jld2" res33

# pomdp34 = make_diagonal_rocks((5,6))
# pol34,up34,b034,s_pomdp34,pg34=soln_params(pomdp34,h;precision=1e-5)
# res34 = rs_scale_run_storm(pomdp34,pg34,h,η,sticky=true,verbose=true)

pomdp44 = make_diagonal_rocks((4,4))
pol44,up44,b044,s_pomdp44,pg44=soln_params(pomdp44,h;precision=1e-5)
res44 = rs_scale_run_storm(pomdp44,pg44,h,η,sticky=true,verbose=false)
display_and_csv(res44,today_date,"sres44")
# @save "rs44_strm_6_19.jld2" res44

pomdp55 = make_diagonal_rocks((5,5))
pol55,up55,b055,s_pomdp55,pg55=soln_params(pomdp55,h;precision=1e-5)
res55 = rs_scale_run_storm(pomdp55,pg55,h,η,sticky=true,verbose=false)
display_and_csv(res55,today_date,"sres55")
# @save "rs55_strm_6_19.jld2" res55

pomdp66 = make_diagonal_rocks((6,6))
pol66,up66,b066,s_pomdp66,pg66=soln_params(pomdp66,h;precision=1e-5)
res66 = rs_scale_run_storm(pomdp66,pg66,h,η,sticky=true,verbose=false)
display_and_csv(res66,today_date,"sres66")
# @save "rs66_strm_6_19.jld2" res66

pomdp77 = make_diagonal_rocks((7,7))
pol77,up77,b077,s_pomdp77,pg77=soln_params(pomdp77,h;precision=1e-5);
res77 = rs_scale_run_storm(pomdp77,pg77,h,η,sticky=true,verbose=false)
display_and_csv(res77,today_date,"sres77")
# @save "rs77_strm_6_19.jld2" res77

##Tiger Scaling with Horizon
h=3
pomdp3 = fixhorizon( TigerPOMDP(),h)
pol3,up3,b03,s_pomdp3,pg3=soln_params(pomdp3,h+1;precision=1e-5);
res3 = tiger_scale_run_storm(pomdp3,pg3,h+1,η,sticky=true,verbose=true)
display_and_csv(res3,today_date,"sres3")
@show res3[!,:x][1]
# @save "t3_strm_6_20.jld2" res3

h=4
pomdp4 = fixhorizon( TigerPOMDP(),h)
pol4,up4,b04,s_pomdp4,pg4=soln_params(pomdp4,h+1;precision=1e-5);
res4 = tiger_scale_run_storm(pomdp4,pg4,h+1,η,sticky=true,verbose=true)
display_and_csv(res4,today_date,"sres4")
@show res4[!,:x][1]
# @save "t4_strm_6_20.jld2" res4

h=5
pomdp5 = fixhorizon( TigerPOMDP(),h)
pol5,up5,b05,s_pomdp5,pg5=soln_params(pomdp5,h+1;precision=1e-5);
res5 = tiger_scale_run_storm(pomdp5,pg5,h+1,η,sticky=true,verbose=true)
display_and_csv(res5,today_date,"sres5")
@show res5[!,:x][1]
# @save "t5_strm_6_20.jld2" res5

h=6
pomdp6 = fixhorizon( TigerPOMDP(),h)
pol6,up6,b06,s_pomdp6,pg6=soln_params(pomdp6,h+1;precision=1e-5);
res6 = tiger_scale_run_storm(pomdp6,pg6,h+1,η,sticky=true,verbose=true)
display_and_csv(res6,today_date,"sres6")
# @save "t6_strm_6_20.jld2" res6

h=7
pomdp7 = fixhorizon( TigerPOMDP(),h)
pol7,up7,b07,s_pomdp7,pg7=soln_params(pomdp7,h+1;precision=1e-5);
res7 = tiger_scale_run_storm(pomdp7,pg7,h+1,η,sticky=true,verbose=true)
display_and_csv(res7,today_date,"sres7")
# @save "t7_strm_6_20.jld2" res7
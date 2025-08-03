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


#Get Solution Parameters
function soln_params(pomdp::POMDP,h::Int;precision=1e-5)
    pol = solve(SARSOPSolver(;precision=precision,max_time=typemax(Float64),verbose=true,max_steps=1000),pomdp)
    up = DiscreteUpdater(pomdp)
    b0 = initialize_belief(up,initialstate(pomdp))
    s_pomdp = EvalTabularPOMDP(pomdp)
    pol_graph = PolicyRobustness.gen_polgraph(pomdp, s_pomdp, pol, b0, h; store_beliefs=true)
    return pol,up,b0,s_pomdp,pol_graph
end

# function soln_params(pomdp::POMDP,h::Int)
#     pol = solve(SARSOPSolver(precision=1e-5),pomdp)
#     up = DiscreteUpdater(pomdp)
#     b0 = initialize_belief(up,initialstate(pomdp))
#     pol_graph = PolicyRobustness.gen_polgraph(pomdp, pol, b0, h;store_beliefs=true)
#     return pol,up,b0,pol_graph
# end

# function undisc_val(pomdp,pg,x,h)
#     mc, rew, is = pg_to_mc(pomdp,pg)
#     t_spec = PolicyRobustness.make_finite_spec(pomdp,rew,h)
#     # @show t_spec
#     return get_value(mc, is, t_spec, x)
# end

# function mc_sims(pomdp,pol_graph,h;runs=10000)
#     b0 = initialize_belief(DiscreteUpdater(pomdp),initialstate(pomdp))
#     pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(pomdp, PolicyRobustness.EvalTabularPOMDP(pomdp), pol_graph; tolerance=1e-7), b0)[1]

#     up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
#     bel02 = initialize_belief(up2,initialstate(pomdp))
    
#     simlist2 = [Sim(pomdp, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
#     mc_res_raw2 = run(simlist2) do sim, hist
#         return [:disc_rew => undiscounted_reward(hist)]
#     end
#     # @show pg_val
#     @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
#     @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
#     return mc_res2,mc_res_sem2
# end

##Test Full Optimization
# function soln_params2(pomdp::POMDP,h::Int;precision=1e-5)
#     pol = solve(SARSOPSolver(precision=precision),pomdp)
#     up = DiscreteUpdater(pomdp)
#     b0 = initialize_belief(up,initialstate(pomdp))
#     s_pomdp = EvalTabularPOMDP(pomdp)
#     pol_graph = PolicyRobustness.gen_polgraph(pomdp, s_pomdp, pol, b0, h; store_beliefs=true)
#     return pol,up,b0,s_pomdp,pol_graph
# end

# function eval_x_STORM(sim::MCSim,x::Float64,η_target::Float64,sticky::Bool,filename::String;verbose=true)
#     param_vals = write_mc_transition(sim.pomdp,sim.pg;filename="../../MyDocker/STORM/"*filename,sticky=sticky)
#     valu = eval_STORM("/data/"*filename,param_vals,x,sim.h;verbose=verbose)
#     V_star = eval_STORM("/data/"*filename,param_vals,0.0,sim.h;verbose=verbose) #Note - don't use the sim-stored value as it will differ
#     return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(valu,V_star), :PLA_Worst => valu)
# end

# function eval_x_STORM(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64};sticky=true,filename::String="mypomdp.pm",verbose=true)
#     df = eval_x_STORM(sim,xs[1],η_targets[1],sticky,filename;verbose=verbose)
#     for i in 2:length(xs)
#         df = vcat(df,eval_x_STORM(sim,xs[i],η_targets[i],sticky,filename;verbose=verbose))
#     end
#     return df
# end


# #CHANGE Sampling - IS O_MAX LEGIT?
# function get_pomdp_from_x_storm(sim::MCSim,mod_spomdp,x,η_target,v_star)
#     id_seed = rand(UInt)
#     sample_from_z_omax!(mod_spomdp,x;rng=MersenneTwister(id_seed))
#     mc, rew, is = pg_to_mc(sim.pomdp,mod_spomdp,sim.pg)
#     spec = make_finite_spec(sim.pomdp,rew,sim.h)
#     V = get_value(mc, is, spec, 0.0) #Better method for this???
#     return DataFrame(:Model=> string(sim.pomdp), :Horizon => sim.h, :η_target => η_target, :x => x, :η => pct_calc(V,v_star), :PLA_worst => V,  :sample_id => id_seed)
# end

# function get_pomdps_from_x_storm(sim::MCSim,x,η_target,n_samples)
#     s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)
#     mod_spomdp = POMDPPolicyGraphs.EvalTabularPOMDP(sim.pomdp)

#     mc, rew, is = pg_to_mc(sim.pomdp,s_pomdp,sim.pg)
#     spec = make_finite_spec(sim.pomdp,rew,sim.h)
#     v_star = get_value(mc, is, spec, 0.0) #Better method for this???

#     df = get_pomdp_from_x_storm(sim,mod_spomdp,x,η_target,v_star)
#     for a in eachindex(s_pomdp.O)
#         mod_spomdp.O2[a] .= s_pomdp.O2[a]
#         mod_spomdp.O[a] .= s_pomdp.O[a]
#     end
#     @showprogress for _ in 2:n_samples
#         df = vcat(df,get_pomdp_from_x_storm(sim,mod_spomdp,x,η_target,v_star))
#         for a in eachindex(s_pomdp.O)
#             mod_spomdp.O2[a] .= s_pomdp.O2[a]
#             mod_spomdp.O[a] .= s_pomdp.O[a]
#         end
#     end
#     return df
# end

# function get_pomdps_from_param_storm(sim::MCSim,xs::Vector{Float64},η_targets::Vector{Float64},n_samples::Int)
#     @info "x = $(xs[1]) - 1/$(length(xs))"
#     @info "Using Sim horizon $(sim.h)"
#     df = get_pomdps_from_x_storm(sim,xs[1],η_targets[1],n_samples)
#     for i in 2:length(xs)
#         @info "x = $(xs[i]) - $i/$(length(xs))"
#         df = vcat(df,get_pomdps_from_x_storm(sim,xs[i],η_targets[i],n_samples))
#     end
#     return df
# end

# function storm_sim(pomdp,param_set,runs,horizon;sticky=false,verbose=true)
#     param_set = collect(param_set)
#     pomdp_pol,pomdp_up,pomdp_b0,spomdp,pomdp_pg = soln_params2(pomdp,horizon)
#     pomdp_sim = PolicyRobustness.MCSim(pomdp,pomdp_pg,horizon)
#     x_res = η_to_x_storm(pomdp_sim,collect(param_set);sticky=sticky,verbose=verbose)
#     xs = x_res[!,:x]
#     df1 = eval_x_STORM(pomdp_sim,xs,param_set;sticky=sticky,filename="mypomdp.pm",verbose=verbose)
#     df2 = get_pomdps_from_param_storm(pomdp_sim,xs,param_set,runs)
#     return x_res, df1, df2
# end

# function udisc_sim(pomdp,param_set,runs,horizon)
#     param_set = collect(param_set)
#     pomdp_pol,pomdp_up,pomdp_b0,spomdp,pomdp_pg = soln_params2(pomdp,horizon)
#     pomdp_sim = PolicyRobustness.MCSim(pomdp,pomdp_pg,horizon)
#     x_res = η_to_x_udisc(pomdp_sim,collect(param_set))
#     return x_res
# end

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
# @save "rs22_strm_6_19.jld2" res22

pomdp33 = make_diagonal_rocks((3,3))
pol33,up33,b033,s_pomdp33,pg33=soln_params(pomdp33,h;precision=1e-5)
res33 = rs_scale_run_storm(pomdp33,pg33,h,η,sticky=true,verbose=false)
# @save "rs33_strm_6_19.jld2" res33

# pomdp34 = make_diagonal_rocks((5,6))
# pol34,up34,b034,s_pomdp34,pg34=soln_params(pomdp34,h;precision=1e-5)
# res34 = rs_scale_run_storm(pomdp34,pg34,h,η,sticky=true,verbose=true)

pomdp44 = make_diagonal_rocks((4,4))
pol44,up44,b044,s_pomdp44,pg44=soln_params(pomdp44,h;precision=1e-5)
res44 = rs_scale_run_storm(pomdp44,pg44,h,η,sticky=true,verbose=false)
# @save "rs44_strm_6_19.jld2" res44

pomdp55 = make_diagonal_rocks((5,5))
pol55,up55,b055,s_pomdp55,pg55=soln_params(pomdp55,h;precision=1e-5)
res55 = rs_scale_run_storm(pomdp55,pg55,h,η,sticky=true,verbose=false)
# @save "rs55_strm_6_19.jld2" res55

pomdp66 = make_diagonal_rocks((6,6))
pol66,up66,b066,s_pomdp66,pg66=soln_params(pomdp66,h;precision=1e-5)
res66 = rs_scale_run_storm(pomdp66,pg66,h,η,sticky=true,verbose=false)
# @save "rs66_strm_6_19.jld2" res66

pomdp77 = make_diagonal_rocks((7,7))
pol77,up77,b077,s_pomdp77,pg77=soln_params(pomdp77,h;precision=1e-5);
res77 = rs_scale_run_storm(pomdp77,pg77,h,η,sticky=true,verbose=false)
# @save "rs77_strm_6_19.jld2" res77

##Tiger Scaling with Horizon
h=3
pomdp3 = fixhorizon( TigerPOMDP(),h)
pol3,up3,b03,s_pomdp3,pg3=soln_params(pomdp3,h+1;precision=1e-5);
res3 = tiger_scale_run_storm(pomdp3,pg3,h+1,η,sticky=true,verbose=true)
@show res3[!,:x][1]
# @save "t3_strm_6_20.jld2" res3

h=4
pomdp4 = fixhorizon( TigerPOMDP(),h)
pol4,up4,b04,s_pomdp4,pg4=soln_params(pomdp4,h+1;precision=1e-5);
res4 = tiger_scale_run_storm(pomdp4,pg4,h+1,η,sticky=true,verbose=true)
@show res4[!,:x][1]
# @save "t4_strm_6_20.jld2" res4

h=5
pomdp5 = fixhorizon( TigerPOMDP(),h)
pol5,up5,b05,s_pomdp5,pg5=soln_params(pomdp5,h+1;precision=1e-5);
res5 = tiger_scale_run_storm(pomdp5,pg5,h+1,η,sticky=true,verbose=true)
@show res5[!,:x][1]
# @save "t5_strm_6_20.jld2" res5

h=6
pomdp6 = fixhorizon( TigerPOMDP(),h)
pol6,up6,b06,s_pomdp6,pg6=soln_params(pomdp6,h+1;precision=1e-5);
res6 = tiger_scale_run_storm(pomdp6,pg6,h+1,η,sticky=true,verbose=true)
# @save "t6_strm_6_20.jld2" res6

h=7
pomdp7 = fixhorizon( TigerPOMDP(),h)
pol7,up7,b07,s_pomdp7,pg7=soln_params(pomdp7,h+1;precision=1e-5);
res7 = tiger_scale_run_storm(pomdp7,pg7,h+1,η,sticky=true,verbose=true)
# @save "t7_strm_6_20.jld2" res7
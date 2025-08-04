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
import SARSOP


## Cancer
include("models/cancer_screening.jl")
c_param_set = 0.0:0.01:1.0
c_horizon = 50
c_runs = 500
cpomdp = CancerPOMDP()
# cancer_pg = PolicyGraph([:wait,:wait,:wait,:wait,:test,:test,:treat],Dict((1,:negative)=>2,(2,:negative)=>3,(3,:negative)=>4,(4,:negative)=>5,(5,:negative)=>1,(5,:positive)=>6,(6,:positive)=>7,(6,:negative)=>5,(7,:positive)=>1,(7,:negative)=>1),1,SparseVector{Float64, Int64}[],Int64[])
cancer_pg2 = PolicyGraph([:wait,:wait,:wait,:wait,:test,:test,:treat,:test],Dict((1,:negative)=>2,(2,:negative)=>3,(3,:negative)=>4,(4,:negative)=>5,(5,:negative)=>8,(5,:positive)=>6,(6,:positive)=>7,(6,:negative)=>5,(7,:positive)=>1,(7,:negative)=>1,(8,:negative)=>1,(8,:positive)=>5),1,SparseVector{Float64, Int64}[],Int64[])

csims2 = MCSim(cpomdp,cancer_pg2,c_horizon)
x2 = η_to_x(csims2,collect(c_param_set))

plot(x2[!,:η_target].*98.53156677380078,x2[!,:x],label="",color = :green,xlabel="Δ - Value Degradation (QALYs)",ylabel="δ - Admissible Observation Deviation",title="Admissible Observation Deviation with Value Degradation",linewidth=2,titlefont=("Computer Modern",13),guidefont=("Computer Modern"),xtickfont=("Computer Modern"),ytickfont=("Computer Modern"))
#savefig(plot,"name")
# L"\Delta"* - LaTeXStrings


## Part Checking
include("models/part_checking.jl")
p_param_set = 0.01:0.02:1.0
p_horizon = 200
p_runs = 500
p_pomdp1 = PartPOMDP(0.0, 0.0, -1.0, 0.0, 0.99, 0.95, 1.0)
pg1 = PolicyGraph([0,1,0,2],Dict((1,true)=>2,(1,false)=>3,(3,false)=>4,(3,true)=>2,(4,true)=>1,(4,false)=>1,(2,true)=>1,(2,false)=>1),1,SparseVector{Float64, Int64}[],Int64[])
p_pomdp2 = PartPOMDP(0.0, 0.0, -1.0, 0.0, 0.90234, 0.95, 1.0)
pg2 = PolicyGraph([0,0,1,0,2],Dict((1,true)=>2,(2,true)=>3,(2,false)=>4,(1,false)=>4,(4,false)=>5,(4,true)=>2,(5,true)=>1,(5,false)=>1,(3,true)=>1,(3,false)=>1),1,SparseVector{Float64, Int64}[],Int64[])
v1i = get_value_udisc(p_pomdp1,pg1,p_horizon) #Uses IVI
v2i = get_value_udisc(p_pomdp2,pg2,p_horizon) #Uses IVI
x1_list = []
x2_list = []
for p in p_param_set
    push!(x1_list,find_x_udisc(p_pomdp1,pg1,p_horizon,p+v1i))
    push!(x2_list,find_x_udisc(p_pomdp2,pg2,p_horizon,p+v2i))
end
param_x1 = p_param_set .+ v1i
param_x2 = p_param_set .+v2i

offset_x1 = clamp.((1.0-0.99).+x1_list,0,1)
offset_x2 = clamp.((1.0-0.90234).+x2_list,0,1)
plot([param_x1,param_x2],[offset_x1,offset_x2],label=["Policy 1 on POMDP 1" "Policy 2 on POMDP 2"],xlabel="Δ - Value Degredation (Fraction Incorrect)",ylabel="Admissible Inaccuracy",title="NS - Admissible Inaccuracy with Fraction Incorrect",color=[:green :black],linewidth=2,titlefont=("Computer Modern",13),guidefont=("Computer Modern"),legendfont=("Computer Modern"),xtickfont=("Computer Modern"),ytickfont=("Computer Modern"))

v1is,t1i = get_storm_value(p_pomdp1, pg1, 0.0, p_horizon;sticky=false)
v2is,t2i = get_storm_value(p_pomdp2, pg2, 0.0, p_horizon;sticky=false)
x1_list_storm = []
x2_list_storm = []
p_param_set_storm = 0.01:0.02:1.0
for p in p_param_set_storm
    push!(x1_list_storm,find_x_storm(p_pomdp1,pg1,p_horizon,p+v1is;verbose=false,sticky=false)[1])
    push!(x2_list_storm,find_x_storm(p_pomdp2,pg2,p_horizon,p+v2is;verbose=false,sticky=false)[1])
end

param_x1_storm = p_param_set_storm .+ v1is
param_x2_storm = p_param_set_storm .+ v2is

offset_x1_storm = clamp.((1.0-0.99).+x1_list_storm,0,1)
offset_x2_storm = clamp.((1.0-0.90234).+x2_list_storm,0,1)
plt=plot([param_x1_storm,param_x2_storm],[offset_x1_storm,offset_x2_storm],label=["Policy 1 on POMDP 1" "Policy 2 on POMDP 2"],xlabel="Δ - Value Degredation (Fraction Incorrect)",ylabel="Admissible Inaccuracy",title="S - Admissible Inaccuracy with Fraction Incorrect",color=[:green :black],linewidth=2,titlefont=("Computer Modern",13),guidefont=("Computer Modern"),legendfont=("Computer Modern"),xtickfont=("Computer Modern"),ytickfont=("Computer Modern"))
# savefig(plt,"partplot_storm.pdf")
# @save "PartCheckStorm_715.jld2" x1_list_storm x2_list_storm param_x1_storm param_x2_storm


## Toy Rover
my_pgs = PolicyGraph([6,1,1,2,4],
                    Dict((1,1)=>2,(1,2)=>3, #Check if in [1,2] or [3,4]
                         (2,1)=>4,(2,2)=>5, #Check if 1 or 2
                         (3,1)=>5,(3,2)=>4, #Check if 3 or 4
                         (4,3)=>4,(5,3)=>5,
                        ),
                    1,SparseVector{Float64, Int64}[],Int64[])

                    #Gap Calculation with smaller POMDP
include("models/simple_sandrover_nobatt.jl")
srpomdp = SDBatteryPOMDP(max_fuel=30,sand_sensor_efficiency=0.99,per_step_rew=-0.0,damagedstates= [-1,0])

st_mat,sr_mat,sis = pg_to_mc(srpomdp,my_pgs)
sr_horizon = 5
@show δ = find_x_pct(st_mat,sis,PolicyRobustness.make_finite_spec(srpomdp,sr_mat,sr_horizon),0.1;eps_pres=0.01,eps_bisect=1e-5)

function find_x_pct_storm_short(pomdp::POMDP, pol_graph::PolicyGraph, horizon::Int, per_deg::Float64; filename::String="mypomdp.pm",sticky=false,verbose=true,upper_bound=1.0,eps_bisect=1e-7)
    param_vals = write_mc_transition(pomdp,pol_graph;filename=joinpath(dirname(pwd()),"STORMFiles",filename),sticky=sticky)
    V,t1 = eval_STORM("/data/"*filename,param_vals,0.0,horizon;verbose=verbose)
    @info "First call time: $t1"
    δV = per_deg*abs(V)
    @info "Initial Policy Value is $V"
    @info "Target Value is $(V-δV)"

    res = PolicyRobustness.upper_bisection_search(x->PolicyRobustness.parse_value_and_time(V-δV,eval_STORM("/data/"*filename,param_vals,x,horizon;verbose=verbose)),0.0,upper_bound;max_iters=1000,eps=eps_bisect)
    return res[1],res[2]+t1
end
@show δ2 = find_x_pct_storm_short(srpomdp,my_pgs,sr_horizon,0.1;sticky=true,verbose=true,upper_bound=0.12,eps_bisect=1e-5)

rover_param_vals_s = write_mc_transition(srpomdp,my_pgs;filename=joinpath(dirname(pwd()),"STORMFiles","rover_s.pm"),sticky=true)
rover_val_s_s = PolicyRobustness.eval_STORM("/data/rover_s.pm",rover_param_vals_s,0.1,sr_horizon)
rover_val_ipe,Tmin = PolicyRobustness.get_value_withT(st_mat,sis,PolicyRobustness.make_finite_spec(srpomdp,sr_mat,sr_horizon),0.1;eps_pres=0.01)
@show rover_val_s_s[1]
@show rover_val_ipe


## Full Rover
include("models/damaged_sandrover_nobatt.jl")
s_param_set = 0.05:0.2:1.0
s_horizon = 50
s_runs = 1000
rpomdp = DBatteryPOMDP(max_fuel=30,sand_sensor_efficiency=0.99,per_step_rew=-0.0,damagedstates= [-1,0])

my_pg = PolicyGraph([6,1,1,2,4,4,2,2,2,2,5],
                    Dict((1,1)=>2,(1,2)=>3, #Check if in [1,2] or [3,4]
                         (2,1)=>4,(2,2)=>5, #Check if 1 or 2
                         (3,1)=>5,(3,2)=>4,#Check if 3 or 4
                         (4,3)=>4, #Go Thru
                         (5,3)=>6,(6,3)=>7,(7,3)=>8,(8,3)=>9,(9,3)=>10,(9,3)=>10,
                         (10,3)=>11,(11,3)=>11), #Go Around
                    1,SparseVector{Float64, Int64}[],Int64[])

#MC Sims
function get_occupancy_mc(pomdp,pol_graph,t_mat,r_mat,max_depth,nruns;s_idx1=1)
    map = rsn_mapping(length(pol_graph.nodes),length(ordered_states(pomdp)))
    occ_lists = Matrix{Float64}[]
    state_list = ordered_states(pomdp)
    x = 0
    for _ in 1:nruns
        mc_state_hist = mc_simulation_hist(t_mat,r_mat;max_depth=max_depth,verbose=false,rng=MersenneTwister(),s_idx1=s_idx1)
        occ_list = zeros(3,5)
        for s_mc_idx in mc_state_hist
            x+=1
            # @show s_mc_idx
            s_mc = map[s_mc_idx]
            if s_mc[3] == 0
                s = state_list[s_mc[2]]
                if isterminal(pomdp,s)
                    break
                end
                if s.pos != [-1,-1]
                    occ_list[s.pos...] += 1
                end
            end
        end
        push!(occ_lists,occ_list)
    end
    display(sum(occ_lists)./nruns)
    return sum(occ_lists)./nruns, std(occ_lists)./sqrt(nruns)
end

#t_mat state 274 - sandy
#t_mat state 34 - not sandy

t_mat,r_mat,is = pg_to_mc(rpomdp,my_pg)
function state_to_tmat_idx(pomdp,state,pg)
    sidx = stateindex(pomdp,state)
    map_to_idx = sn_mapping(length(pg.nodes),length(ordered_states(pomdp)))
    return map_to_idx[(1,sidx,0)]
end
mc_occ1_sandy1 = get_occupancy_mc(rpomdp,my_pg,t_mat,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 2, 0),my_pg))
mc_occ1_sandy2 = get_occupancy_mc(rpomdp,my_pg,t_mat,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 3, 0),my_pg))
mc_occ1_notsandy1 = get_occupancy_mc(rpomdp,my_pg,t_mat,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 1, 0),my_pg))
mc_occ1_notsandy2 = get_occupancy_mc(rpomdp,my_pg,t_mat,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 4, 0),my_pg))

#Deviation Values
δ_r = find_x_pct(t_mat,is,PolicyRobustness.make_finite_spec(rpomdp,r_mat,s_horizon),0.2)
# t_mat_w = worst_case_T_infinite(rpomdp,t_mat,r_mat,is,δ_r)
rover_val_ipe_r,Tmin_r = PolicyRobustness.get_value_withT(t_mat,is,PolicyRobustness.make_finite_spec(rpomdp,r_mat,s_horizon),δ_r;eps_pres=0.0)
mc_occ2_sandy1 = get_occupancy_mc(rpomdp,my_pg,Tmin_r,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 2, 0),my_pg))
mc_occ2_sandy2 = get_occupancy_mc(rpomdp,my_pg,Tmin_r,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 3, 0),my_pg))
mc_occ2_notsandy1 = get_occupancy_mc(rpomdp,my_pg,Tmin_r,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 1, 0),my_pg))
mc_occ2_notsandy2 = get_occupancy_mc(rpomdp,my_pg,Tmin_r,r_mat,15,10000;s_idx1=state_to_tmat_idx(rpomdp,DRoverState([3, 1], 30, 4, 0),my_pg))
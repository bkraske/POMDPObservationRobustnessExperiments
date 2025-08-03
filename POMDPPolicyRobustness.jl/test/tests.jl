using Pkg
Pkg.activate(".")

using PolicyRobustness
using POMDPs
using POMDPTools
using SparseArrays
using LinearAlgebra
using IntervalMDP
using Random
using POMDPPolicyGraphs

Pkg.activate("test")
using POMDPModels
using RockSample
using FiniteHorizonPOMDPs
# using SARSOP
using Test
using Statistics
using NativeSARSOP


function soln_params(tiger::POMDP,h::Int)
    pol = solve(SARSOPSolver(precision=1e-5),tiger)
    up = DiscreteUpdater(tiger)
    b0 = initialize_belief(up,initialstate(tiger))
    pol_graph = PolicyRobustness.gen_polgraph(tiger, pol, b0, h;store_beliefs=true)
    return pol,up,b0,pol_graph
end

function formulation_test(tiger)
    h = 30
    pol,up,b0,pol_graph = soln_params(tiger, h)
    # reach_s = PolicyRobustness.reached_states(tiger,pol_graph)
    # s_maps = indexin(ordered_states(tiger),reach_s)

    tiger_mc, tiger_rew, tiger_is = policy_to_mc(tiger,pol,h)

    rmapping = PolicyRobustness.rsn_mapping(length(pol_graph.nodes),length(ordered_states(tiger)))

    #Basic Probability Check
    # @test all(sum(tiger_mc,dims=1) .== 1.0) #NOTE: Change this to a check for nodes which have parents
    has_parents = vcat(sum(tiger_mc,dims=2)...) .> 0.0
    for i in eachindex(has_parents)
        if has_parents[i]
            # @show i
            # @show node,state = rmapping[i]
            # @show real_s = indexin(reach_s,ordered_states(tiger))[state]
            # @show isterminal(tiger,ordered_states(tiger)[real_s])
            # @show sum(tiger_mc[i,:])
            @test sum(tiger_mc[:,i]) == 1.0
        end
    end

    @show sum(sum(tiger_mc,dims=1))
    @show length(sum(tiger_mc,dims=1))
    mapping = PolicyRobustness.sn_mapping(length(pol_graph.nodes),length(ordered_states(tiger)))
    for (i,s) in enumerate(b0.state_list)
        reach_idx = i#s_maps[i]
        if !isnothing(reach_idx)
            @test tiger_mc[mapping[(1,reach_idx,0)],tiger_is] == b0.b[i]
        end
    end

    edge_list = [e for e in pol_graph.edges]
    #PG Transition Checks
    for n in eachindex(pol_graph.nodes)
        for (i,s) in enumerate(ordered_states(tiger))
            reach_idx_s = i
            current_idx = mapping[(n,reach_idx_s,0)]

            #Reward Check
            s_rew = isterminal(tiger,s) ? 0.0 : POMDPs.reward(tiger,s,pol_graph.nodes[n])
            @test tiger_rew[current_idx] == s_rew

            #Transition Tests
            if !isterminal(tiger,s)
                sps = transition(tiger,s,pol_graph.nodes[n])
                for sp in support(sps)
                    reach_idx_sp = stateindex(tiger,sp)
                    pdf_val = pdf(sps,sp)
                    next_idx = mapping[(n,reach_idx_sp,1)]
                    # @info next_idx, current_idx
                    @test tiger_mc[next_idx,current_idx] == pdf_val
                    
                    #Observation Tests
                    node_edges = findall(x->x[1]==n,edge_list)
                    next_nodes = unique([pol_graph.edges[e] for e in node_edges])
                    node_map = Dict(unique([pol_graph.edges[e] for e in node_edges]).=> 1:length(next_nodes))
                    node_probs = zeros(length(next_nodes))
                    for edgy in node_edges
                        next_node = pol_graph.edges[edgy]
                        node_probs[node_map[next_node]] += pdf(observation(tiger,pol_graph.nodes[n],sp),o)
                    end
                    for i in eachindex(next_nodes)
                        next_obs_idx = mapping[(next_nodes[i],reach_idx_sp,0)]
                        @test tiger_mc[next_obs_idx,next_idx] == node_probs[i]
                    end
                end
            else
                @test tiger_mc[current_idx,current_idx] == 1.0
            end
        end
    end
end

function tofromtest(mat)
    obs_idx = PolicyRobustness.obs_start_idx(size(mat,1))
    for i in 2:size(mat)[2]
        inds_list = findall(x->x>0, mat[:,i])
        for indy in inds_list
            if indy != i
                if i < obs_idx
                    @test indy >= obs_idx
                else
                    @test indy < obs_idx
                end
            end
        end
    end
end

@testset "MC Formulation Tests: Tiger" begin
    my_tiger = TigerPOMDP()
    formulation_test(my_tiger)
end

@testset "MC Formulation Tests: RockSample" begin
    my_rock = fixhorizon(RockSamplePOMDP(),10)
    formulation_test(my_rock)
end

@testset "MC Formulation Tests: Baby" begin
    my_rock = fixhorizon(BabyPOMDP(),15)
    formulation_test(my_rock)
end

function mc_comparison(lower,original)
    del = maximum(abs.(original-lower))
    @show del
    lb,ub = PolicyRobustness.matrix_from_bound(original, del)
    @test all(lb .<= original)
    @test all(original .<= ub)
    @test all(lb .- eps.(lb) .<= lower)
    @test all(lower .<= ub+eps.(ub))
end

function make_upper_best(mc)
    upper_mc = copy(mc)
    mc_size = size(mc)[1]
    obs_start = PolicyRobustness.obs_start_idx(mc_size)
    for j in obs_start:mc_size
        for i in 1:obs_start-1 #see tofromtest - Observations should only transition to states
            if mc[i,j] != 0.0 && j != 1
                upper_mc[i,j] = 1.0
            end
        end
    end
    return upper_mc
end

@testset "Tiger Value Calculation Tests" begin
    tiger1 = TigerPOMDP()
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    tiger_mc_original, _, _ = pg_to_mc(tiger1,pol_graph)

    tofromtest(tiger_mc_original)

    upper_best = make_upper_best(tiger_mc_original)
    display(upper_best)

    parm = [0.85,1.0,0.95,0.75,0.65,0.5]
    for p in parm
        @show p
        tiger2 = TigerPOMDP(-1.0,-100.0,10.0,p,0.95)
        pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

        up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
        bel02 = initialize_belief(up2,initialstate(tiger1))
        
        h = 200
        # if p == 1.0
        #     h = 10000
        # end
        runs = 10000
        simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
        mc_res_raw2 = run(simlist2) do sim, hist
            return [:disc_rew => discounted_reward(hist)]
        end
        @show pg_val
        @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
        @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
        # if p != 1.0 #Don't test in deterministic observation cases due to low/no variance? ####
            # @test abs(pg_val-mc_res2) <= mc_res_sem2 #Maybe reactivate, but Sims are not infinite horizon
        # end

        tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
        tofromtest(tiger_mc)
        lower_mc = copy(tiger_mc)
        upper_mc = copy(tiger_mc)

        tiger_prob = IntervalProbabilities(;lower=lower_mc,upper=upper_mc)
        tiger_imc = IntervalMarkovChain(tiger_prob, [tiger_is])

        tiger_prop = InfiniteTimeReward(tiger_rew, POMDPs.discount(tiger2)^0.5, 1e-7)
        tiger_spec = Specification(tiger_prop, Pessimistic, Minimize)
        tiger_problem = Problem(tiger_imc, tiger_spec)

        V, k, residual = value_iteration(tiger_problem)
        @show v_adjust = V[1]/IntervalMDP.discount(tiger_prop)

        @test isapprox(pg_val,v_adjust;atol=0.001)


        tiger_prob2 = IntervalProbabilities(;lower=lower_mc,upper=upper_best)
        tiger_imc2 = IntervalMarkovChain(tiger_prob2, [tiger_is])
        tiger_problem2 = Problem(tiger_imc2, tiger_spec)

        V2, k2, residual2 = value_iteration(tiger_problem2)
        @show v_adjust2 = V2[1]/IntervalMDP.discount(tiger_prop)
        display(lower_mc)
        @test isapprox(v_adjust,v_adjust2)

        if p <= tiger1.p_listen_correctly
            mc_comparison(lower_mc,tiger_mc_original)
            @show V3 = get_value(tiger_mc_original, tiger_is, tiger_spec, tiger1.p_listen_correctly-p)
            @test isapprox(v_adjust,V3)
        end

    end
end

function test_everything(tiger1,pol)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    tiger_mc, tiger_rew, tiger_is, upper_best = pg_to_mc(tiger1,pol_graph)

    lower_mc = copy(tiger_mc)
    upper_mc = copy(tiger_mc)

    tiger_prob = IntervalProbabilities(;lower=lower_mc,upper=upper_mc)
    tiger_imc = IntervalMarkovChain(tiger_prob, [tiger_is])

    tiger_prop = InfiniteTimeReward(tiger_rew, POMDPs.discount(tiger1)^0.5, 1e-7)
    tiger_spec = Specification(tiger_prop, Pessimistic, Minimize)
    tiger_problem = Problem(tiger_imc, tiger_spec)

    V, k, residual = value_iteration(tiger_problem)
    @show v_adjust = V[1]/IntervalMDP.discount(tiger_prop)
end


@testset "RS Value Calculation Tests" begin
    tiger1 = fixhorizon(RockSamplePOMDP(),10)
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    tiger_mc_original, _, _ = pg_to_mc(tiger1,pol_graph)
    tofromtest(tiger_mc_original)
    upper_best = make_upper_best(tiger_mc_original)
    display(upper_best)

    parm = [1.0,3.0,5.0,10.0,20.0,500.0]
    for p in parm
        @show p
        tiger2 =  fixhorizon(RockSamplePOMDP(;sensor_efficiency=p),10)
        pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

        up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
        bel02 = initialize_belief(up2,initialstate(tiger1))
        
        h = 200
        runs = 10000
        simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
        mc_res_raw2 = run(simlist2) do sim, hist
            return [:disc_rew => discounted_reward(hist)]
        end
        @show pg_val
        @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
        @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
        @test abs(pg_val-mc_res2) <= mc_res_sem2

        tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
        tofromtest(tiger_mc)
        lower_mc = copy(tiger_mc)
        upper_mc = copy(tiger_mc)

        tiger_prob = IntervalProbabilities(;lower=lower_mc,upper=upper_mc)
        tiger_imc = IntervalMarkovChain(tiger_prob, [tiger_is])

        tiger_prop = InfiniteTimeReward(tiger_rew, POMDPs.discount(tiger2)^0.5, 1e-7)
        tiger_spec = Specification(tiger_prop, Pessimistic, Minimize)
        tiger_problem = Problem(tiger_imc, tiger_spec)

        V, k, residual = value_iteration(tiger_problem)
        @show v_adjust = V[1]/IntervalMDP.discount(tiger_prop)

        @test isapprox(pg_val,v_adjust;atol=0.001)


        tiger_prob2 = IntervalProbabilities(;lower=lower_mc,upper=upper_best)
        tiger_imc2 = IntervalMarkovChain(tiger_prob2, [tiger_is])
        tiger_problem2 = Problem(tiger_imc2, tiger_spec)

        V2, k2, residual2 = value_iteration(tiger_problem2)
        @show v_adjust2 = V2[1]/IntervalMDP.discount(tiger_prop)

        @test isapprox(v_adjust,v_adjust2)

        # if p <= 20.0 ####Fix me! - Made difficult by state-dependent observation probs
        #     mc_comparison(lower_mc,tiger_mc_original)
        #     @show V3 = get_value(tiger_mc_original, tiger_is, tiger_spec, 0.95-p)
        #     @test isapprox(v_adjust,V3)
        # end

    end
end

@testset "Baby Value Calculation Tests" begin #Adjust Scaling
    hp = 10
    tiger1 = fixhorizon(BabyPOMDP(),hp)
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    @show size(b0.b)
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    @info length(pol_graph.nodes)
    @info length(states(tiger1))
    tiger_mc_original, _, _ = pg_to_mc(tiger1,pol_graph)
    tofromtest(tiger_mc_original)
    upper_best = make_upper_best(tiger_mc_original)
    @info upper_best
    display(upper_best)
    @info tiger_mc_original

    parm = [0.85,1.0,0.95,0.75,0.65,0.5]
    parm2 = [0.85,1.0,0.95,0.75,0.65,0.5]
    for p1 in parm
        for p2 in round.(1.0 .- parm ; digits=3)
            # if p1 <= tiger1.m.p_cry_when_hungry && p2 >= tiger1.m.p_cry_when_not_hungry && round(abs(p1-tiger1.m.p_cry_when_hungry),digits=3) == round(abs(p2-tiger1.m.p_cry_when_not_hungry),digits=3)
            @show p1
            @show p2
            tiger2 = fixhorizon(BabyPOMDP(-5.0, -10.0, 0.1, p1, p2, 0.9),hp)
            pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

            up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
            bel02 = initialize_belief(up2,initialstate(tiger2))
            
            h = 200
            runs = 10000
            simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h,rng=MersenneTwister()) for _ in 1:runs]
            mc_res_raw2 = run(simlist2) do sim, hist
                return [:disc_rew => discounted_reward(hist)]
            end
            @show pg_val
            @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
            @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
            @test abs(pg_val-mc_res2) <= mc_res_sem2

            tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
            tofromtest(tiger_mc)
            lower_mc = round.(tiger_mc,digits=10)
            upper_mc = round.(tiger_mc,digits=10)

            #Testing accuracy with no range
            tiger_prob = IntervalProbabilities(;lower=lower_mc,upper=upper_mc)
            tiger_imc = IntervalMarkovChain(tiger_prob, [tiger_is])

            tiger_prop = InfiniteTimeReward(tiger_rew, POMDPs.discount(tiger2)^0.5, 1e-7)
            tiger_spec = Specification(tiger_prop, Pessimistic, Minimize)
            tiger_problem = Problem(tiger_imc, tiger_spec)

            V, k, residual = value_iteration(tiger_problem)
            @show v_adjust = V[1]/IntervalMDP.discount(tiger_prop)

            @test isapprox(pg_val,v_adjust;atol=0.001)

            #Testing accuracy with some range (optimistic UB)
            tiger_prob2 = IntervalProbabilities(;lower=lower_mc,upper=upper_best)
            tiger_imc2 = IntervalMarkovChain(tiger_prob2, [tiger_is])
            tiger_problem2 = Problem(tiger_imc2, tiger_spec)

            V2, k2, residual2, delt2 = value_iteration(tiger_problem2;include_T=true)
            @show V2
            @show v_adjust2 = V2[1]/IntervalMDP.discount(tiger_prop)
            @info V2
            @test isapprox(v_adjust,v_adjust2)
            @info lower_mc

            if p1 <= tiger1.m.p_cry_when_hungry && p2 >= tiger1.m.p_cry_when_not_hungry && round(abs(p1-tiger1.m.p_cry_when_hungry),digits=3) == round(abs(p2-tiger1.m.p_cry_when_not_hungry),digits=3)
                @warn round(abs(p1-tiger1.m.p_cry_when_hungry),digits=3)
                mc_comparison(lower_mc,tiger_mc_original)
                V3 = get_value(tiger_mc_original, tiger_is, tiger_spec, abs(p1-tiger1.m.p_cry_when_hungry))
                # V3 = V3[1]/IntervalMDP.discount(tiger_prop)
                # @test isapprox(v_adjust,V3)
                @test isapprox(v_adjust,V3;atol=abs(v_adjust))
                V32, low_mat, up_mat, delt3 = PolicyRobustness.get_value_test(tiger_mc_original, tiger_is, tiger_spec, abs(p1-tiger1.m.p_cry_when_hungry))
                V32val = V[1]/IntervalMDP.discount(tiger_spec.prop)
                @test isapprox(V32val,V3;atol=abs(V32val))
                if lower_mc+delt2 != low_mat+delt3
                    differs = (lower_mc+delt2)-(low_mat+delt3)
                    differ_idxs = findnz(differs)
                    for idx in unique(differ_idxs[2])
                        val_diffs = V32[findnz(differs[:,idx])[1]]
                        @test length(unique(val_diffs)) == 1
                    end
                else
                    @test lower_mc+delt2 == low_mat+delt3
                end
            end
        end
    end
end

#Simulation Tests
@testset "MC Sims - Baby" begin
    hp = 10
    tiger1 = fixhorizon(BabyPOMDP(),hp)
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    @show size(b0.b)
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    nl = length(pol_graph.nodes)
    sl = length(states(tiger1))
    tiger_mc_original, rew_mat, _ = pg_to_mc(tiger1,pol_graph)
    mapy = sn_mapping(nl,sl)
    rmapy = rsn_mapping(nl,sl)

    #Terminal Loop Check
    os = ordered_states(tiger1)
    for s_idx in 2:PolicyRobustness.obs_start_idx(tiger_mc_original.n)-1
        if !unreachable_check(tiger_mc_original,s_idx)
            state_indx = rmapy[s_idx][2]
            if isterminal(tiger1,os[state_indx]) != istermloop(tiger_mc_original,rew_mat,s_idx)
                @info rmapy[s_idx][2]
                @info s_idx
            end
            @test isterminal(tiger1,os[state_indx]) == istermloop(tiger_mc_original,rew_mat,s_idx)
        end
    end

    h = 200
    runs = 10000

    up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
    bel02 = initialize_belief(up2,initialstate(tiger1))

    simlist2 = [Sim(tiger1, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
    mc_res_raw2 = run(simlist2) do sim, hist
        return [:disc_rew => discounted_reward(hist)]
    end

    pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger1, PolicyRobustness.EvalTabularPOMDP(tiger1), pol_graph; tolerance=1e-7), b0)[1]

    @show pg_val
    @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
    @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
    @test abs(pg_val-mc_res2) <= mc_res_sem2

    mc_simulation(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5;max_depth=11,verbose=false)

    vec_rew = mc_simulation_vector(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=11,verbose=false)./POMDPs.discount(tiger1)^0.5
    mrew = mean(vec_rew)
    semrew = 3 * std(vec_rew) / sqrt(runs)
    @show mrew
    @show semrew
    @test abs(pg_val-mrew) <= semrew

    parm = [0.85,1.0,0.95,0.75,0.65,0.5]
    parm2 = [0.85,1.0,0.95,0.75,0.65,0.5]
    for p1 in parm
        for p2 in round.(1.0 .- parm ; digits=3)
            # if p1 <= tiger1.m.p_cry_when_hungry && p2 >= tiger1.m.p_cry_when_not_hungry && round(abs(p1-tiger1.m.p_cry_when_hungry),digits=3) == round(abs(p2-tiger1.m.p_cry_when_not_hungry),digits=3)
            @show p1
            @show p2
            tiger2 = fixhorizon(BabyPOMDP(-5.0, -10.0, 0.1, p1, p2, 0.9),hp)
            pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

            up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
            bel02 = initialize_belief(up2,initialstate(tiger2))
            
            h = 200
            runs = 50000
            simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h,rng=MersenneTwister()) for _ in 1:runs]
            mc_res_raw2 = run(simlist2) do sim, hist
                return [:disc_rew => discounted_reward(hist)]
            end
            @show pg_val
            @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
            @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
            @test abs(pg_val-mc_res2) <= mc_res_sem2

            tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
            lower_mc = round.(tiger_mc,digits=10)
            upper_mc = round.(tiger_mc,digits=10)

            vec_rew = mc_simulation_vector(lower_mc,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=11,verbose=false)./POMDPs.discount(tiger1)^0.5
            mrew = mean(vec_rew)
            semrew = 3 * std(vec_rew) / sqrt(runs)
            @show mrew
            @show semrew
            @test abs(pg_val-mrew) <= semrew
        end
    end
end

@testset "MC Sims - RockSample" begin
    hp = 10
    tiger1 = fixhorizon(RockSamplePOMDP(),hp)
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    @show size(b0.b)
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    nl = length(pol_graph.nodes)
    sl = length(states(tiger1))
    tiger_mc_original, rew_mat, _ = pg_to_mc(tiger1,pol_graph)
    mapy = sn_mapping(nl,sl)
    rmapy = rsn_mapping(nl,sl)

    #Terminal Loop Check
    os = ordered_states(tiger1)
    for s_idx in 2:PolicyRobustness.obs_start_idx(tiger_mc_original.n)-1
        if !unreachable_check(tiger_mc_original,s_idx)
            state_indx = rmapy[s_idx][2]
            if isterminal(tiger1,os[state_indx]) != istermloop(tiger_mc_original,rew_mat,s_idx)
                @info rmapy[s_idx][2]
                @info s_idx
            end
            @test isterminal(tiger1,os[state_indx]) == istermloop(tiger_mc_original,rew_mat,s_idx)
        end
    end

    h = 200
    runs = 10000

    up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
    bel02 = initialize_belief(up2,initialstate(tiger1))

    simlist2 = [Sim(tiger1, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
    mc_res_raw2 = run(simlist2) do sim, hist
        return [:disc_rew => discounted_reward(hist)]
    end

    pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger1, PolicyRobustness.EvalTabularPOMDP(tiger1), pol_graph; tolerance=1e-7), b0)[1]

    @show pg_val
    @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
    @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
    @test abs(pg_val-mc_res2) <= mc_res_sem2

    mc_simulation(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5;max_depth=11,verbose=false)

    vec_rew = mc_simulation_vector(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=11,verbose=false)./POMDPs.discount(tiger1)^0.5
    mrew = mean(vec_rew)
    semrew = 3 * std(vec_rew) / sqrt(runs)
    @show mrew
    @show semrew
    @test abs(pg_val-mrew) <= semrew

    parm = [1.0,3.0,5.0,10.0,20.0,500.0]
    for p in parm
        @show p
        tiger2 =  fixhorizon(RockSamplePOMDP(;sensor_efficiency=p),10)
        pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

        up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
        bel02 = initialize_belief(up2,initialstate(tiger2))
        
        h = 200
        runs = 10000
        simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h,rng=MersenneTwister()) for _ in 1:runs]
        mc_res_raw2 = run(simlist2) do sim, hist
            return [:disc_rew => discounted_reward(hist)]
        end
        @show pg_val
        @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
        @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
        @test abs(pg_val-mc_res2) <= mc_res_sem2

        tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
        lower_mc = round.(tiger_mc,digits=10)
        upper_mc = round.(tiger_mc,digits=10)

        vec_rew = mc_simulation_vector(lower_mc,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=11,verbose=false)./POMDPs.discount(tiger1)^0.5
        mrew = mean(vec_rew)
        semrew = 3 * std(vec_rew) / sqrt(runs)
        @show mrew
        @show semrew
        @test abs(pg_val-mrew) <= semrew
    end
end

@testset "MC Sims - Tiger" begin
    hp = 15
    tiger1 = fixhorizon(TigerPOMDP(),hp)
    pol = solve(SARSOPSolver(precision=1e-5),tiger1)
    up = DiscreteUpdater(tiger1)
    b0 = initialize_belief(up,initialstate(tiger1))
    @show size(b0.b)
    pol_graph = PolicyRobustness.gen_polgraph(tiger1, pol, b0, 30)
    nl = length(pol_graph.nodes)
    sl = length(states(tiger1))
    tiger_mc_original, rew_mat, _ = pg_to_mc(tiger1,pol_graph)
    mapy = sn_mapping(nl,sl)
    rmapy = rsn_mapping(nl,sl)

    #Terminal Loop Check
    os = ordered_states(tiger1)
    for s_idx in 2:PolicyRobustness.obs_start_idx(tiger_mc_original.n)-1
        if !unreachable_check(tiger_mc_original,s_idx)
            state_indx = rmapy[s_idx][2]
            if isterminal(tiger1,os[state_indx]) != istermloop(tiger_mc_original,rew_mat,s_idx)
                @info rmapy[s_idx][2]
                @info s_idx
            end
            @test isterminal(tiger1,os[state_indx]) == istermloop(tiger_mc_original,rew_mat,s_idx)
        end
    end

    h = 200
    runs = 10000

    up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
    bel02 = initialize_belief(up2,initialstate(tiger1))

    simlist2 = [Sim(tiger1, pol_graph, up2, bel02, rand(b0); max_steps=h, rng=MersenneTwister()) for _ in 1:runs]
    mc_res_raw2 = run(simlist2) do sim, hist
        return [:disc_rew => discounted_reward(hist)]
    end

    pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger1, PolicyRobustness.EvalTabularPOMDP(tiger1), pol_graph; tolerance=1e-7), b0)[1]

    @show pg_val
    @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
    @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
    @test abs(pg_val-mc_res2) <= mc_res_sem2

    mc_simulation(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5;max_depth=hp+1,verbose=false)

    vec_rew = mc_simulation_vector(tiger_mc_original,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=hp+1,verbose=false)./POMDPs.discount(tiger1)^0.5
    mrew = mean(vec_rew)
    semrew = 3 * std(vec_rew) / sqrt(runs)
    @show mrew
    @show semrew
    @test abs(pg_val-mrew) <= semrew

    parm = [0.85,1.0,0.95,0.75,0.65,0.5]
    for p in parm
        @show p
        tiger2 = fixhorizon(TigerPOMDP(-1.0,-100.0,10.0,p,0.95),hp)
        pg_val = PolicyRobustness.calc_belvalue_polgraph(pol_graph, PolicyRobustness.eval_polgraph(tiger2, PolicyRobustness.EvalTabularPOMDP(tiger2), pol_graph; tolerance=1e-7), b0)[1]

        up2 = PolicyRobustness.PolicyGraphUpdater(pol_graph)
        bel02 = initialize_belief(up2,initialstate(tiger2))
        
        h = 200
        runs = 10000
        simlist2 = [Sim(tiger2, pol_graph, up2, bel02, rand(b0); max_steps=h,rng=MersenneTwister()) for _ in 1:runs]
        mc_res_raw2 = run(simlist2) do sim, hist
            return [:disc_rew => discounted_reward(hist)]
        end
        @show pg_val
        @show mc_res2 = mean(mc_res_raw2[!, :disc_rew])
        @show mc_res_sem2 = 3 * std(mc_res_raw2[!, :disc_rew]) / sqrt(runs)
        @show eps(pg_val-mc_res2)
        @test (abs(pg_val-mc_res2) <= mc_res_sem2) || (abs(pg_val-mc_res2) <= 1e-7)

        tiger_mc, tiger_rew, tiger_is = pg_to_mc(tiger2,pol_graph)
        lower_mc = round.(tiger_mc,digits=10)
        upper_mc = round.(tiger_mc,digits=10)

        vec_rew = mc_simulation_vector(lower_mc,rew_mat,POMDPs.discount(tiger1)^0.5,runs;max_depth=hp+1,verbose=false)./POMDPs.discount(tiger1)^0.5
        mrew = mean(vec_rew)
        semrew = 3 * std(vec_rew) / sqrt(runs)
        @show mrew
        @show semrew
        @test (abs(pg_val-mrew) <= semrew) || (abs(pg_val-mrew) <= 1e-7)
    end
end

@testset "Interval Sampling" begin
    low = spzeros(3,3)
    upper = sparse(ones(3,3))
    for _ in 1:1000
        tmat = sample_from_interval_not_uniform(low,upper)
        @test all(0.0 .<= tmat .<= 1.0)
        @test all(sum(tmat,dims=1).==1)
    end
end


function sample_z_and_test(pomdp,x)
    s_pomdp = POMDPPolicyGraphs.EvalTabularPOMDP(pomdp)
    mod_spomdp = POMDPPolicyGraphs.EvalTabularPOMDP(pomdp)
    for _ in 1:30
        sample_from_z_omax!(mod_spomdp,x)
        for a in eachindex(s_pomdp.O)
            @test all(s_pomdp.O2[a] .- x .<= mod_spomdp.O2[a] .<= s_pomdp.O2[a] .+ x)
            @test all( 1-1e-10 .<= sum(mod_spomdp.O2[a],dims=1) .<= 1+1e-10)
            @test all(s_pomdp.O[a] .- x .<= mod_spomdp.O[a] .<= s_pomdp.O[a] .+ x)
            @test all( 1-1e-10 .<= sum(mod_spomdp.O[a],dims=2) .<= 1+1e-10)
            mod_spomdp.O2[a] .= s_pomdp.O2[a]
            mod_spomdp.O[a] .= s_pomdp.O[a]
        end
    end
end

@testset "Z Sampling" begin
    pomdps = [TigerPOMDP(),BabyPOMDP(),RockSamplePOMDP()]
    for pomdp in pomdps
        @info pomdp
        for x in 0.0:0.1:1.0
            @info x
            sample_z_and_test(pomdp,x)
        end
    end
end
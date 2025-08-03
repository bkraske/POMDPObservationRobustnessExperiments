Base.@kwdef struct CancerPOMDP <: POMDP{Symbol,Symbol,Symbol}
    all_states = [:healthy, :ins_cancer, :inv_cancer, :death]
    s_dict = Dict(all_states .=> 1:length(all_states))
    acts = [:wait, :treat, :test]
    a_dict = Dict(acts .=> 1:length(acts))
    i_state = Deterministic(:healthy) #Breaks w/o Deterministic
    disc = 0.999
    ins_prob = 0.02 #Probability of having In-Situ Cancer
    ins_t_prob = 0.6 #Probaility of being healthy after treating In-Situ Cancer
    ins_v_prob = 0.1 #Probaility of invasive cancer after not treating In-Situ Cancer
    inv_th_prob = 0.2 #Probaility of being healthy after treating Invasive Cancer
    inv_td_prob = 0.2 #Probaility of death after treating Invasive Cancer
    inv_d_prob = 0.6 #Probability of death from not treating invasive cancer
    obs_list = [:positive, :negative, :dead]
    o_dict = Dict(obs_list .=> 1:length(obs_list))
    ht_prob = 0.05 #Prob of testing pos if healthy
    inst_prob = 0.80 #Prob of testing pos if in-situ cancer
    invt_prob = 1.00 #Prob of testing pos if invasive cancer
    tr_prob = 1.00 #Prob of finding cancer if treating
end

POMDPs.states(pomdp::CancerPOMDP) = pomdp.all_states
POMDPTools.ordered_states(pomdp::CancerPOMDP) = pomdp.all_states
POMDPs.stateindex(pomdp::CancerPOMDP, s) = pomdp.s_dict[s]

POMDPs.actions(pomdp::CancerPOMDP) = pomdp.acts
POMDPTools.ordered_actions(pomdp::CancerPOMDP) = pomdp.acts
POMDPs.actionindex(pomdp::CancerPOMDP,a) = pomdp.a_dict[a]

POMDPs.observations(pomdp::CancerPOMDP) = pomdp.obs_list
POMDPTools.ordered_observations(pomdp::CancerPOMDP) = pomdp.obs_list
POMDPs.obsindex(pomdp::CancerPOMDP,o) = pomdp.o_dict[o]

POMDPs.discount(pomdp::CancerPOMDP) = pomdp.disc

POMDPs.initialstate(pomdp::CancerPOMDP) = pomdp.i_state

POMDPs.isterminal(pomdp::CancerPOMDP,s) = s==:death

function POMDPs.transition(pomdp::CancerPOMDP,s,a)
    #if healthy
    if s == :healthy
        return SparseCat([:healthy, :ins_cancer],[1-pomdp.ins_prob, pomdp.ins_prob])
    end

    #if in-situ caner
    if s == :ins_cancer
        if a == :treat
            return SparseCat([:healthy, :ins_cancer],[pomdp.ins_t_prob, 1-pomdp.ins_t_prob])
        elseif a == :wait || a == :test
            return SparseCat([:inv_cancer, :ins_cancer],[pomdp.ins_v_prob, 1-pomdp.ins_v_prob])
        end
    end

    #if invasive cancer
    if s == :inv_cancer
        if a == :treat
            return SparseCat([:healthy,:inv_cancer,:death],[pomdp.inv_th_prob, (1-pomdp.inv_th_prob-pomdp.inv_td_prob), pomdp.inv_td_prob])
        elseif a == :wait || a == :test
            return SparseCat([:inv_cancer, :death],[1-pomdp.inv_d_prob, pomdp.inv_d_prob])
        end
    end

    if s == :death
        return Deterministic(:death)
    end
end

function POMDPs.observation(pomdp::CancerPOMDP,a,sp)
    #if action is test
    if a == :test
        if sp == :healthy
            return SparseCat([:positive, :negative],[pomdp.ht_prob, 1-pomdp.ht_prob])
        elseif sp == :ins_cancer
            return SparseCat([:positive, :negative],[pomdp.inst_prob, 1-pomdp.inst_prob])
        elseif sp == :inv_cancer
            return SparseCat([:positive, :negative],[pomdp.invt_prob, 1-pomdp.invt_prob])
        end
    end

    #if action is treat
    if a == :treat && ((sp == :ins_cancer) || (sp == :inv_cancer))
        return SparseCat([:positive, :negative],[pomdp.tr_prob, 1-pomdp.tr_prob])
    end


    if sp == :death
        return Deterministic(:dead)
    end

    return SparseCat([:positive, :negative],[0, 1]) ####OK?
end

function POMDPs.observation(pomdp::CancerPOMDP,s,a,sp)
    POMDPs.observation(pomdp,a,sp)
end

function POMDPs.reward(pomdp,s,a)
    if s == :death
        return 0.0
    else
        if a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end
end
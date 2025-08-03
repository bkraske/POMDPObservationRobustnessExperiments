# Modified from TigerPOMDPs.jl: https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/TigerPOMDPs.jl

mutable struct PartPOMDP <: POMDP{Int64, Int64, Bool}
    r_check::Float64
    r_check_slow::Float64
    r_fail::Float64
    r_pass::Float64
    p_check_correctly::Float64
    p_check_correctly_slow::Float64
    discount_factor::Float64
end
# PartPOMDP() = PartPOMDP(-0.01, -0.02, -1.0, 0.0, 0.75, 0.95, 0.999)
PartPOMDP() = PartPOMDP(0.0, 0.0, -1.0, 0.0, 0.99, 0.95, 1.0)

POMDPs.states(::PartPOMDP) = (0,1,2)
POMDPs.observations(::PartPOMDP) = (false, true)

POMDPs.stateindex(::PartPOMDP, s::Int64) = s + 1
POMDPs.actionindex(::PartPOMDP, a::Int) = a + 1
POMDPs.obsindex(::PartPOMDP, o::Bool) = Int64(o) + 1

const PART_CHECK = 0
const PART_APPROVE = 1
const PART_REJECT = 2
const PART_CHECK_SLOW = 3

const PART_PASS = 1
const PART_FAIL = 0
const TERM = 2

POMDPs.isterminal(pomdp::PartPOMDP,s::Int64) = s==TERM 

function POMDPs.transition(pomdp::PartPOMDP, s::Int64, a::Int64)
    if a == PART_APPROVE || a == PART_REJECT
        return SparseCat(0:2,[0.0,0.0,1.0])
    elseif s == 1
        return SparseCat(0:2,[0.0,1.0,0.0])
    else
        return SparseCat(0:2,[1.0,0.0,0.0])
    end
end

function POMDPs.observation(pomdp::PartPOMDP, a::Int64, sp::Int64)
    pc = pomdp.p_check_correctly
    p = 1.0
    if a == PART_CHECK
        sp==1 ? (p = pc) : (p = 1.0-pc)
    elseif  a == PART_CHECK_SLOW
        sp==1 ? (p = pomdp.p_check_correctly_slow) : (p = 1.0-pomdp.p_check_correctly_slow)
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function POMDPs.observation(pomdp::PartPOMDP, s::Int64, a::Int64, sp::Int64)
    return observation(pomdp, a, sp)
end


function POMDPs.reward(pomdp::PartPOMDP, s::Int64, a::Int64)
    r = 0.0
    a == PART_CHECK && (r+=pomdp.r_check)
    a == PART_CHECK_SLOW && (r+=pomdp.r_check_slow)
    if a == PART_APPROVE
        s == PART_PASS ? (r += pomdp.r_pass) : (r += pomdp.r_fail)
    end
    if a == PART_REJECT
        s == PART_FAIL ? (r += 0.0) : (r += pomdp.r_fail*0.0) #Is this what we want?
    end
    return r
end
POMDPs.reward(pomdp::PartPOMDP, s::Int64, a::Int64, sp::Int64) = POMDPs.reward(pomdp, s, a)


POMDPs.initialstate(pomdp::PartPOMDP) = SparseCat(0:2,[0.5,0.5,0.0])

POMDPs.actions(::PartPOMDP) = 0:3

function upperbound(pomdp::PartPOMDP, s::Int64)
    return pomdp.r_pass
end

POMDPs.discount(pomdp::PartPOMDP) = pomdp.discount_factor

POMDPs.initialobs(p::PartPOMDP, s::Int64) = observation(p, 0, s) # check
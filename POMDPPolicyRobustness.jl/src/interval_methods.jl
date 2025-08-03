# Make Specifications
function make_infinite_spec(pomdp,rew;precision=1e-7)
    prop = InfiniteTimeReward(rew,POMDPs.discount(pomdp)^0.5,precision)
    return Specification(prop, Pessimistic, Minimize)
end

function make_finite_spec(pomdp,rew,horizon)
    prop = FiniteTimeReward(rew,1.0,2*horizon)
    return Specification(prop, Pessimistic, Minimize)
end

#Make MCs
function make_MC(lower_mc,upper_mc,is)
    prob = IntervalProbabilities(;lower=lower_mc,upper=upper_mc)
    return IntervalMarkovChain(prob, [is])
end

#Make infinite horizon problem
function make_infinite_problem(pomdp::POMDP,lower_mc::SparseMatrixCSC,upper_mc::SparseMatrixCSC,rew::SparseVector,is::Int)
   return Problem(make_MC(lower_mc,upper_mc,is),make_infinite_spec(pomdp,rew;precision=1e-7))
end

#Set upper and lower bound matrices
function matrix_from_bound(mc::SparseMatrixCSC, x;eps_pres=0.0)
    lower_mc = copy(mc)
    upper_mc = copy(mc)
    bound = x
    mc_size = size(mc)[1]
    obs_start = obs_start_idx(mc_size)
    row_vals = rowvals(mc)
    for j in obs_start:mc_size
        for idx in nzrange(mc,j)
            i = row_vals[idx]
            if i < obs_start #Don't change cyclical observations (should be unreachable)
                lower_mc[i,j] = clamp(lower_mc[i,j]-bound,eps_pres,1.0)
                upper_mc[i,j] = clamp(upper_mc[i,j]+bound,eps_pres,1.0)
            end
        end
    end
    return lower_mc, upper_mc
end

##Value Calculations
#Get the value of a specification on a set of matrices
function get_value(mc, is, spec, x; include_T=false, eps_pres=0.0)
    if x!=0
        lower_mc,upper_mc = matrix_from_bound(mc, x;eps_pres=eps_pres) 
    else
        lower_mc = mc
        upper_mc = mc
    end

    problem = Problem(make_MC(lower_mc,upper_mc,is), spec)
    V, _, _ = value_iteration(problem;include_T=include_T)
    return V[1]/IntervalMDP.discount(spec.prop)
end

function get_value_withT(mc, is, spec, x; eps_pres=0.0)
    if x!=0
        lower_mc,upper_mc = matrix_from_bound(mc, x;eps_pres=eps_pres) 
    else
        lower_mc = mc
        upper_mc = mc
    end

    problem = Problem(make_MC(lower_mc,upper_mc,is), spec)
    V, _, _, delT = value_iteration(problem;include_T=true)
    return V[1]/IntervalMDP.discount(spec.prop), lower_mc .+ delT
end

#Returns more detailed outputs than the above
function get_value_test(mc, is, spec, x)
    lower_mc,upper_mc = matrix_from_bound(mc, x) 

    problem = Problem(make_MC(lower_mc,upper_mc,is), spec)
    V, k, residual, delT = value_iteration(problem;include_T=true)
    return V, lower_mc, upper_mc, delT
end

##Functions for Finding x given δV
#General
function find_x(mc::SparseMatrixCSC,is::Int,spec,δV::Float64)
    V = get_value(mc, is, spec, 0)
    @info "Initial Policy Value is $V"
    @info "Target Value is $(V-δV)"

    res = upper_bisection_search(x->V-δV-get_value(mc, is, spec, x),0.0,1.0;max_iters=1000,eps=1e-7)
    return res
end

#Assumes infinite horizon
function find_x(pomdp::POMDP,policy_graph::PolicyGraph,δV::Number)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)

    spec = make_infinite_spec(pomdp,rew;precision=1e-7)

    return find_x(mc,is,spec,δV)
end

#Finite horizon
function find_x_udisc(pomdp::POMDP,policy_graph::PolicyGraph,horizon::Int,δV)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)
    spec = make_finite_spec(pomdp,rew,horizon)

    return find_x(mc,is,spec,δV)
end

##Percent-based Code
function find_x_pct(mc::SparseMatrixCSC,is::Int,spec,per_deg::Float64;eps_pres=0.0,eps_bisect=1e-7)
    V = get_value(mc, is, spec, 0)
    δV = per_deg*abs(V)
    @info "Initial Policy Value is $V"
    @info "Target Value is $(V-δV)"

    res = upper_bisection_search(x->V-δV-get_value(mc, is, spec, x;eps_pres=eps_pres),0.0,1.0;max_iters=1000,eps=eps_bisect)
    return res
end

function find_x_pct(pomdp::POMDP,policy_graph::PolicyGraph,per_deg::Float64;eps_pres=0.0,eps_bisect=1e-7)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)
    spec = make_infinite_spec(pomdp,rew;precision=1e-7)

    return find_x_pct(mc,is,spec,per_deg;eps_pres=eps_pres,eps_bisect=eps_bisect)
end

function find_x_pct_udisc(pomdp::POMDP,policy_graph::PolicyGraph,horizon::Int,per_deg::Float64;eps_pres=0.0,eps_bisect=1e-7)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)
    spec = make_finite_spec(pomdp,rew,horizon)
    return find_x_pct(mc,is,spec,per_deg;eps_pres=eps_pres,eps_bisect=eps_bisect)
end

##Worst Case Matrix Code
function worst_case_T_infinite(pomdp::POMDP,mc::SparseMatrixCSC,rew::SparseVector,is::Int,x::Float64)
    lower_mc, upper_mc = matrix_from_bound(mc, x)

    problem = make_infinite_problem(pomdp,lower_mc,upper_mc,rew,is)

    V, k, residual, delT = value_iteration(problem;include_T=true) #Uses IntervalMDPs.jl
    return V[1]/IntervalMDP.discount(problem.spec.prop), lower_mc .+ delT, is, rew
end

function worst_case_T_infinite(pomdp::POMDP,policy_graph::PolicyGraph,x::Float64)
    mc, rew, is = pg_to_mc(pomdp,policy_graph)
    return worst_case_T_infinite(pomdp,mc,rew,is,x)
end

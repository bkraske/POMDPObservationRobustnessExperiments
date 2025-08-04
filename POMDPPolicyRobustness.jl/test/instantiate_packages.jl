Threads.nthreads() == 1 ? nothing : @warn "Run Julia with a single thread."
cur_dir = pwd()
@show bn = basename(cur_dir)

if bn == "POMDPObservationRobustnessExperiments"
    cd(joinpath(cur_dir,"POMDPPolicyRobustness.jl"))
elseif bn == "POMDPPolicyRobustness.jl"
    nothing
elseif bn == "test"
    cd(dirname(@__DIR__))
else
    throw("Directory Not Recongnized. Please run scripts from `POMDPObservationRobustnessExperiments` folder.")
end

using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/bkraske/POMDPPolicyGraphs.jl",rev="bkraske-clean-up")
Pkg.add(url="https://github.com/bkraske/IntervalMDP.jl")
Pkg.instantiate()

Pkg.activate("test")
Pkg.instantiate()
# POMDPObservationRobustnessExperiments

- Install Julia ([Instructions](https://julialang.org/downloads/))- Note v1.9.0 was used for all experiments
- Install STORM ([Instructions](https://www.stormchecker.org/documentation/obtain-storm/docker.html)) - Note v1.9.0 was used for all experiments
- Run experiments located in the `POMDPPolicyRobustness.jl\test` folder from the top level directory (`POMDPObservationRobustnessExperiments`) using `julia <file name>` or `include(<file name>)` in an active Julia session.
    - `POMDPPolicyRobustness.jl\test\ns_validation.jl` for RIS-NS validation results
    - `POMDPPolicyRobustness.jl\test\s_validation.jl` for RIS-S validation results
    - `POMDPPolicyRobustness.jl\test\ns_scaling.jl` for RIS-NS scaling results
    - `POMDPPolicyRobustness.jl\test\s_scaling.jl` for RIS-S scaling results
    - `POMDPPolicyRobustness.jl\test\case_studies.jl` for case studies
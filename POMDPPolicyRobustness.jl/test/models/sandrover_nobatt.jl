#=
Refueling Problem
=#

using StaticArrays
using LinearAlgebra
using Random
using POMDPs
using POMDPModels
using POMDPTools
using Compose
using ColorSchemes
using Parameters
using Printf
using POMDPGifs
using Cairo

struct RoverState
    pos::GWPos
    fuel::Int #fuel is observation
    sand::Int
end

@with_kw struct BatteryPOMDP <: POMDP{RoverState, Int, Int}
    size::Tuple{Int, Int}       = (3,5)
    max_fuel::Int                       = 10
    sand_types::Int                     = 2
    rewards::Dict{GWPos, Float64}  = Dict(GWPos(3,5) =>1.0)
    # science::Set{GWPos}             = create_science()
    # sun::SVector{7*10, GWPos} = create_sun_regions()
    # shadow::Set{GWPos}             = create_shadow_region()
    sand::Set{GWPos}               = create_sand_regions()
    obstacles::Set{GWPos}           = create_obstacles()
    terminate_from::Int                 = 0
    terminal_state::RoverState         = RoverState(GWPos(-1,-1), -1, -1)
    battery_uncertainty::Float64                  = 1.0 #0.7
    sensor_efficiency_stop::Float64               = 0.8
    sensor_efficiency_load::Float64               = 0.6
    sand_sensor_efficiency::Float64               = 1.0
    discount::Float64               = 0.999
    stateindices::Vector{Int}       = cumprod([size[1], size[2], sand_types, max_fuel+1])
end

const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))

function inbounds(m::BatteryPOMDP, s::RoverState)
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2]
end

# [placement] true ->  1: (ax'=0) & (ay'=0); //+ 1/4: (ax'=1) & (ay'=1) + 1/4: (ax'=2) & (ay'=1) + 1/4: (ax'=1) & (ay'=3);
# [west] northenabled -> (1-slippery): (ax'=max(ax-1,axMIN)) + slippery: (ax'=max(ax-2,axMIN)); // 0.9 prob go left, 0.1 prob go two left
# [east] southenabled -> (1-slippery): (ax'=min(ax+1,axMAX)) + slippery: (ax'=min(ax+2,axMAX)); //
# [south]  eastenabled -> (1-slippery): (ay'=min(ay+1,ayMAX)) + slippery: (ay'=min(ay+2,ayMAX));
# [north]  westenabled -> (1-slippery): (ay'=max(ay-1,ayMIN)) + slippery: (ay'=max(ay-2,ayMIN));

function POMDPs.transition(mdp::BatteryPOMDP, s::RoverState, a::Int)
    if s.fuel == 0 || isterminal(mdp, s) || haskey(mdp.rewards, s.pos) || (s.pos ∈ mdp.sand && s.sand == 2)  #|| (s.pos[1] == 10 && s.pos[2] == 10)
        return Deterministic(RoverState(GWPos(-1,-1), -1, -1))
    end

    destinations = MVector{2, RoverState}(undef)
    probs = @MVector(zeros(2))

    probs[1] = mdp.battery_uncertainty
    probs[2] = 1 - mdp.battery_uncertainty

    #this part does the position of the robot
    new_x = s.pos[1]
    new_y = s.pos[2]

    if a == aind[:left]
        new_x = max(s.pos[1] + dir[:left][1], 1)
    elseif a == aind[:right]
        new_x = min(s.pos[1] + dir[:right][1], mdp.size[1])
    elseif a == aind[:up]
        new_y = min(s.pos[2] + dir[:up][2], mdp.size[2])
    elseif a == aind[:down]
        new_y = max(s.pos[2] + dir[:down][2], 1)
    end

    # # this part does the fueling stuff based on sand type and shadow
    # new_fuel_1 = s.fuel
    # new_fuel_2 = s.fuel

    # # if a == aind[:refuel]
    # #    if s.pos[1] == 3 && s.pos[2] == 1
    # #         new_fuel_1 = min(mdp.max_fuel, s.fuel + 2)
    # #         new_fuel_2 = min(mdp.max_fuel, s.fuel + 1)
    # #    end
    # if a == aind[:sense]
    #     if s.pos[1] == 3 && (s.pos[2] == 1 || s.pos[2] == 2)
    #         new_fuel_1 = max(0, s.fuel-1)
    #         new_fuel_2 = max(0, s.fuel-1)
    #     else
    #         new_fuel_1 = 0
    #         new_fuel_2 = 0
    #     end
    # else
    #     if s.pos ∈ mdp.sand
    #         # @show s.sand
    #         if s.sand == 2
    #             new_fuel_1 = max(0, s.fuel - 30)
    #             new_fuel_2 = max(0, s.fuel - 30)
    #         else
    #             new_fuel_1 = max(0, s.fuel - 1)
    #             new_fuel_2 = max(0, s.fuel - 1)
    #         end
    #     else
    #         new_fuel_1 = max(0, s.fuel - 4)
    #         new_fuel_2 = max(0, s.fuel - 1)
    #     end
    #     # new_fuel_1 = max(0, s.fuel - 1)
    #     # new_fuel_2 = max(0, s.fuel - 2)
    # end

    # new_fuel_1 = max(new_fuel_1, 0)
    # new_fuel_2 = max(new_fuel_2, 0)
    # new_fuel_1 = min(new_fuel_1, mdp.max_fuel)
    # new_fuel_2 = min(new_fuel_2, mdp.max_fuel)
    new_fuel_1 = mdp.max_fuel

    sand = s.sand
    
    # if new_fuel_1 == new_fuel_2
        return Deterministic(RoverState(GWPos(new_x, new_y), new_fuel_1, sand))
    # else
    #     destinations[1] = RoverState(GWPos(new_x, new_y), new_fuel_1, sand)
    #     destinations[2] = RoverState(GWPos(new_x, new_y), new_fuel_2, sand)
    #     return SparseCat(convert(SVector, destinations), convert(SVector, probs))
    # end
end

## States 

POMDPs.length(pomdp::BatteryPOMDP) = pomdp.size[1]*pomdp.size[2]*pomdp.sand_types+1 #*(pomdp.max_fuel+1)+1

function POMDPs.states(pomdp::BatteryPOMDP)
    ss = vec([RoverState(GWPos(x, y), z, v) for x in 1:pomdp.size[1], y in 1:pomdp.size[2], z in pomdp.max_fuel, v in 1:pomdp.sand_types])
    push!(ss, RoverState(GWPos(-1,-1), -1, -1))
end

function POMDPs.stateindex(pomdp::BatteryPOMDP, s::RoverState) 
   if isterminal(pomdp, s)
        return length(pomdp)
    end
    return s.pos[1] + pomdp.stateindices[1] * (s.pos[2] -1) + pomdp.stateindices[2]* (s.sand -1 )# + pomdp.stateindices[3] * (s.fuel)
end

function POMDPs.initialstate(pomdp::BatteryPOMDP)
    probs = [0.5, 0.5]
    #concatenate probs twice
    # probs = vcat(probs, probs)

    pos = GWPos(3,1)
    fuels = [pomdp.max_fuel - 1, pomdp.max_fuel - 2]
    sands = [2,1]
    states = []
    # for f in fuels
        for s in sands
            push!(states, RoverState(pos, pomdp.max_fuel, s))
        end
    # end
    return SparseCat(states, probs)
end

## Actions 

const aind = Dict(:sense=>1, :up=>2, :down=>3, :left=>4, :right=>5)

POMDPs.actions(pomdp::BatteryPOMDP) = 1:5
POMDPs.actionindex(mdp::BatteryPOMDP, a::Int) = a

## Transition

POMDPs.isterminal(m::BatteryPOMDP, s) = (s == m.terminal_state)

## Observation

## struct ObservationBools
##     done::Bool
##     crash::Bool
## end

# POMDPs.observations(pomdp::BatteryPOMDP) = 0:(pomdp.max_fuel) #fix this

POMDPs.observations(pomdp::BatteryPOMDP) = 1:((pomdp.sand_types) * (pomdp.max_fuel + 1))


POMDPs.obsindex(::BatteryPOMDP, o::Int) = o
# POMDPTools.ordered_observations(::BatteryPOMDP) = true

function POMDPs.initialobs(pomdp::BatteryPOMDP, s) 
    return observation(pomdp, s, :up, s)
end

# function POMDPs.observation(pomdp::BatteryPOMDP, a::Int, sp)

#     if sp.fuel == -1
#         return Deterministic(0)
#     end

#     efficiency = (a == aind[:refuel]) ? pomdp.sensor_efficiency_stop : pomdp.sensor_efficiency_load
    
#     fuel = sp.fuel
#     lower_fuel = max(fuel - 1, 0)
#     higher_fuel = min(fuel + 1, pomdp.max_fuel)
    
#     if lower_fuel == fuel
#         # At minimum fuel level
#         return SparseCat((fuel, higher_fuel), (0.5 + efficiency/2, 0.5 - efficiency/2))
#     elseif higher_fuel == fuel
#         # At maximum fuel level
#         return SparseCat((lower_fuel, fuel), (0.5 - efficiency/2, 0.5 + efficiency/2))
#     else
#         # Not at minimum or maximum fuel level
#         return SparseCat((lower_fuel, fuel, higher_fuel), 
#                          (0.5 - efficiency/2, efficiency, 0.5 - efficiency/2))
#     end
# end
function POMDPs.observation(pomdp::BatteryPOMDP, a::Int, sp)
    if sp.fuel == -1
        return Deterministic(1)  # Assuming index 1 for failure state
    end
    
    sand_type = sp.sand
    fuel = sp.fuel

    if a == aind[:sense]
        # When sensing, only observe sand type but use correct index in combined space
        obs_indices = Int[]
        probs = Float64[]
        
        for s in 1:pomdp.sand_types
            # Use sand type with default fuel level for indexing
            obs_index = (s - 1) * (pomdp.max_fuel + 1) + (fuel + 1)
            push!(obs_indices, obs_index)
            if s == sand_type
                push!(probs, pomdp.sand_sensor_efficiency)
            else
                push!(probs, (1 - pomdp.sand_sensor_efficiency) / (pomdp.sand_types - 1))
            end
        end
        return SparseCat(obs_indices, probs)
    else
        # For non-sensing actions, only observe fuel but use correct index in combined space
        efficiency = pomdp.sensor_efficiency_load
        lower_fuel = max(fuel - 1, 0)
        higher_fuel = min(fuel + 1, pomdp.max_fuel)
        
        # Use current sand type for indexing
        obs_index = 0 * (pomdp.max_fuel + 1) + (fuel + 1)
        lower_obs_index = 0 * (pomdp.max_fuel + 1) + (lower_fuel + 1)
        higher_obs_index = 0 * (pomdp.max_fuel + 1) + (higher_fuel + 1)
        
        if lower_fuel == fuel
            return SparseCat([obs_index, higher_obs_index], 
                           [0.5 + efficiency/2, 0.5 - efficiency/2])
        elseif higher_fuel == fuel
            return SparseCat([lower_obs_index, obs_index], 
                           [0.5 - efficiency/2, 0.5 + efficiency/2])
        else
            return SparseCat([lower_obs_index, obs_index, higher_obs_index], 
                           [0.5 - efficiency/2, efficiency, 0.5 - efficiency/2])
        end
    end
end


## Reward 
function POMDPs.reward(pomdp::BatteryPOMDP, s, a)
    r = 0
    r+= haskey(pomdp.rewards, s.pos) ? pomdp.rewards[s.pos] : 0   #s.fuel == 0 ? -1 : 0 #s.pos[1] == 10 && s.pos[2] == 10 ? 1 : 0 #haskey(pomdp.rewards, s) ? pomdp.rewards[s] : 0
    r+= s.pos ∈ pomdp.obstacles ? -10.0 : 0
    return r
end
POMDPs.discount(pomdp::BatteryPOMDP) = 0.99

## distributions 
# POMDPs.initialstate(pomdp::BatteryPOMDP) = uniform_belief(pomdp)

## helpers

function deterministic_belief(pomdp, s)
    b = zeros(length(states(pomdp)))
    si = stateindex(pomdp, s)
    b[si] = 1.0
    return DiscreteBelief(pomdp, b)
end

# function create_sun_regions()
#     region = [GWPos(x,y) for x in 1:2 for y in 1:2]
#     append!(region, [GWPos(x,y) for x in ])
#     return  SVector{7*10, GWPos}(
# end

function create_shadow_region()
    region = [GWPos(x,y) for x in 3:4 for y in 3:4]
    append!(region, [GWPos(x,y) for x in 7:9 for y in 2:4])
    append!(region, [GWPos(x,y) for x in 1:3 for y in 7:10])
    append!(region, [GWPos(x,y) for x in 6:8 for y in 6:8])
    return Set(region)
end

function create_sand_regions()
    region = [GWPos(x,y) for x in 3:3 for y in 3:4]
    return Set(region)
end

function create_obstacles()
    obs = [GWPos(x,y) for x in 2:2 for y in 2:4]
    append!(obs, GWPos(x,y) for x in 4:4 for y in 2:5)
    return Set(obs)
end

# function POMDPs.isterminal(pomdp::POMDP, b::DiscreteBelief)
#     belief_support = states(policy.problem)[findall(b.b .> 0)]
#     if ProductState(GWPos(8,5),4) ∈ belief_support || ProductState(GWPos(3,7), 3) ∈ belief_support 
#         return true
#     end
#     for s in belief_support
#         if !isterminal(pomdp, s)
#             return false 
#         end
#     end
#     return true
# end

## Rendering 


begin
    function POMDPTools.render(pomdp::POMDP, step;
        viz_battery_state=true,
        viz_belief=true,
        pre_act_text=""
    )

        background = compose(context(0, 0, 1, 1), Compose.rectangle(), fill("white"))

        if step[:s].pos == GWPos(-1, -1)
            return render_terminal_state(pomdp)
        end
        nx, ny = pomdp.size[1] + 1, pomdp.size[2] + 1
        
        # Calculate sizes
        grid_size = 1.0  # 80% of the total width for the main grid
        chart_size = 0.0  # 20% of the total width for the battery belief chart
        
        # Create main grid context
        grid_ctx = context(0, 0, grid_size, 1)
        
        cells = []
        for x in 1:nx, y in 1:ny
            ctx = cell_ctx((x, y), (nx, ny))
            cell = compose(ctx, Compose.rectangle(), fill("white"))
            push!(cells, cell)
        end

        # special_ctx = context(0,0, grid)

        grid = compose(grid_ctx, linewidth(0.1mm), Compose.stroke("gray"), cells...)
        outline = compose(grid_ctx, linewidth(0.1mm), Compose.rectangle())
        # sun_areas = compose(grid_ctx, render_shadowed_areas(pomdp, (nx, ny)))
        reward_areas = compose(grid_ctx, render_reward_areas(pomdp, (nx, ny)))
        obstacles = compose(grid_ctx, render_obstacles(pomdp, (nx, ny)))

        sand = compose(grid_ctx, render_sand_areas(pomdp, (nx, ny), step[:b]))
        # shields = compose(grid_ctx, render_shield(pomdp, step, (nx, ny)))
        # special_areas = compose(grid_ctx, render_special_reward_areas(pomdp, (nx, ny)))

        agent = nothing
        action = nothing
        if get(step, :s, nothing) !== nothing
            agent_ctx = cell_ctx((step[:s].pos.x, step[:s].pos.y), (nx, ny))
            agent = compose(grid_ctx, render_agent(agent_ctx, step[:s].fuel / pomdp.max_fuel))
            if get(step, :a, nothing) !== nothing
                action = compose(grid_ctx, render_action(pomdp, step))
            end
        end
        action_text = compose(grid_ctx, render_action_text(pomdp, step, pre_act_text))
        
        # Render belief on the grid
        belief = nothing
        if viz_belief && (get(step, :b, nothing) !== nothing)
            belief = compose(grid_ctx, render_belief(pomdp, step))
        end
        

        # @show step[:s].s.sand, step[:s].s.pos, step[:a]

        # get_sand_beliefs(pomdp, step[:b])

        # Render battery belief distribution on the right side
        # battery_belief = nothing
        # if viz_battery_state && (get(step, :b, nothing) !== nothing)
        #     chart_ctx = context(grid_size, 0, chart_size, 1)
        #     battery_belief = compose(chart_ctx, render_battery_belief(pomdp, step))
        # end
        # compose(context(),
                    #    sun_areas, reward_areas, special_areas, obstacles, 
                    #    agent, shields, action, belief, grid, outline, battery_belief, background)
   
        compose(context(), 
                       reward_areas, obstacles, 
                       agent , sand, action, belief, action_text, grid, outline, background)
    end

    function render_shield(pomdp::POMDP, step, size)

        mapping = Dict(2=>:up, 3=>:down, 4=>:left, 5=>:right)

        nx, ny = size
        shielded = []
        
        x = step[:s].pos[1]
        y = step[:s].pos[2]

        for a in 2:5
            direction = mapping[a]
            shield_x = max(0, min(nx, x + dir[direction][1]))
            shield_y = max(0, min(ny, y + dir[direction][2]))
            ctx = cell_ctx((shield_x, shield_y), (nx, ny))
            if step[:shield][a] == 0
                red_box = compose(ctx, Compose.rectangle(), fill("red"), fillopacity(1.0))
                push!(shielded, red_box)
            elseif step[:shield][a] == 2
                orange_box = compose(ctx, Compose.rectangle(), fill("orange"), fillopacity(1.0))
                push!(shielded, orange_box)
            end
        end
        return compose(context(), shielded...)
    end

    # function render_shadowed_areas_simple(pomdp::POMDP, size)
    #     nx, ny = size
    #     shadow_areas = []
    #     for shadow_pos in pomdp.shadow
    #         ctx = cell_ctx((shadow_pos[1], shadow_pos[2]), (nx, ny))
    #         println("Rendering shadow at context: ", ctx)
    #         shadow = compose(ctx, Compose.rectangle(), fill("gray31"), fillopacity(0.5))
    #         push!(shadow_areas, shadow)
    #     end
    #     return compose(context(), shadow_areas...)
    # end

    function render_battery_belief(pomdp::POMDP, step)
        if !haskey(step, :b)
            return nothing
        end
    
        battery_distribution, avg_battery, _ = get_battery_beliefs(pomdp, step[:b])
        
        # Calculate the width and position of the battery belief chart
        chart_height = 0.25  # 80% of the total height
        chart_y = 0.1   # Start at 10% from the top
    
        # Create bars for each battery level
        bars = []
        max_prob = maximum(battery_distribution)
        bar_width = 1 / length(battery_distribution)
        for (level, prob) in enumerate(battery_distribution)
            bar_height = (prob / max_prob) * chart_height
            bar_y = chart_y + chart_height - bar_height
            bar_ctx = context((level-1)*bar_width, bar_y, bar_width, bar_height)
            bar = compose(bar_ctx, Compose.rectangle(), fill("blue"))
            push!(bars, bar)
        end
    
        # Create labels
        labels = []
        label_ctx = context(0, chart_y + chart_height + 0.02, 1, 0.1)
        label = compose(label_ctx, Compose.text(0.05, 0.5, "Battery Level Belief"), Compose.font("Arial"), fontsize(8pt))
        push!(labels, label)
    
        avg_battery_text = string("Avg: ", round(avg_battery, digits=2))
        avg_label_ctx = context(0, chart_y - 0.05, 1, 0.1)
        avg_label = compose(avg_label_ctx, Compose.text(0.25, 0.25, avg_battery_text), Compose.font("Arial"), fontsize(8pt))
        push!(labels, avg_label)
    
        return compose(context(), bars..., labels...)
    end

    function render_belief(pomdp::POMDP, step)
        battery_beliefs, avg_beliefs, terminal_prob = get_battery_beliefs(pomdp, step[:b])
        belief_outlines = []
        belief_fills = []
        nx, ny = pomdp.size[1] + 1, pomdp.size[2] + 1
        x = step[:s].pos.x
        y = step[:s].pos.y

        ctx = cell_ctx((x, y), (nx, ny))
        clr = "black"
        belief_outline = compose(ctx, Compose.rectangle(0.1, 0.87, 0.8, 0.07), Compose.stroke("gray31"), fill("gray31"))
        belief_fill = compose(ctx, Compose.rectangle(0.1, 0.87, (avg_beliefs) * 0.8, 0.07), Compose.stroke("lawngreen"), fill("lawngreen"))
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)

        return compose(context(), belief_fills..., belief_outlines...)
    end

    function get_battery_beliefs(pomdp::POMDP, b)
        max_fuel = pomdp.max_fuel
        battery_beliefs = zeros(Float64, max_fuel + 1)
        terminal_prob = 0.0
        
        for (s, p) in weighted_iterator(b)
            if s.pos.x > 0 && s.pos.y > 0 && 
               s.pos.x <= pomdp.size[1] && s.pos.y <= pomdp.size[2]
                battery_beliefs[s.fuel + 1] += p
            else
                # This is the terminal state
                terminal_prob += p
            end
        end
        
        # Normalize battery beliefs
        total_prob = sum(battery_beliefs)
        if total_prob > 0
            battery_beliefs ./= total_prob
        end
        
        # Calculate average battery level
        avg_battery = sum(battery_beliefs[f] * (f-1) for f in 1:(max_fuel+1)) / max_fuel
        
        return battery_beliefs, avg_battery, terminal_prob
    end

    function get_sand_beliefs(pomdp::POMDP, b)
        sand_beliefs = zeros(Float64, 2)
        terminal_prob = 0.0
        
        for (s, p) in weighted_iterator(b)
            if s.pos.x > 0 && s.pos.y > 0 && 
               s.pos.x <= pomdp.size[1] && s.pos.y <= pomdp.size[2]
               sand_beliefs[s.sand] += p
            else
                # This is the terminal state
                terminal_prob += p
            end
        end
        
        # Normalize battery beliefs
        total_prob = sum(sand_beliefs)
        if total_prob > 0
            sand_beliefs ./= total_prob
        end
        
        # Calculate average battery level
        # avg_battery = sum(sand_beliefs[f] * (f-1) for f in 1:(max_fuel+1)) / max_fuel
        @show sand_beliefs
        return sand_beliefs
        # return battery_beliefs, avg_battery, terminal_prob
    end

    function render_terminal_state(pomdp::POMDP)
        ctx = context()
        background = compose(ctx, Compose.rectangle(), fill("black"))
        text = compose(ctx, text(0.5, 0.5, "Terminal State Reached", hcenter, vcenter),
                    fill("white"), fontsize(30pt))
        return compose(context(), background, text)
    end

    function cell_ctx(xy, size)
        nx, ny = size
        x, y = xy
        return context((x - 1) / nx, (ny - y - 1) / ny, 1 / nx, 1 / ny)
    end

    function render_shadowed_areas(pomdp::POMDP, size)
        nx, ny = size
        shadow_areas = []
        for shadow_pos in pomdp.shadow
            ctx = cell_ctx((shadow_pos[1], shadow_pos[2]), (nx, ny))
            shadow = compose(ctx, Compose.rectangle(), fill("gray31"), fillopacity(0.5))
            push!(shadow_areas, shadow)
        end
        return compose(context(), shadow_areas...)
    end

    function render_sand_areas(pomdp::POMDP, size, b)
        nx, ny = size
        sand_areas = []
        sand_beliefs = get_sand_beliefs(pomdp, b)
        
        for (i, sand) in enumerate(pomdp.sand)
            ctx = cell_ctx((sand[1], sand[2]), (nx, ny))
            
            # Mix colors based on beliefs
            # good (white) = RGB(1,1,1)
            # bad (brown) = RGB(0.6,0.3,0) 
            
            # Linear interpolation between brown and white based on belief of bad sand
            bad_belief = sand_beliefs[2]  # probability of bad sand
            # brown_bias = bad_belief^0.5  
            # color = RGB{Float64}(
            #     1 - 0.4 * brown_bias,  # red component
            #     1 - 0.7 * brown_bias,  # green component
            #     1 - 1.0 * brown_bias   # blue component
            # )
            
            sanded = compose(ctx, 
                            Compose.rectangle(), 
                            fill("brown"), 
                            fillopacity(bad_belief))
            push!(sand_areas, sanded)
        end
        return compose(context(), sand_areas...)
    end

    function render_reward_areas(pomdp::POMDP, size)
        nx, ny = size
        reward_areas = []
        for (state, reward) in pomdp.rewards
            ctx = cell_ctx((state.x, state.y), (nx, ny))
            reward_area = compose(ctx, star(0.5, 0.5, 0.3, 5, 0.5), fill("gold"), Compose.stroke("black"))
            push!(reward_areas, reward_area)
        end
        return compose(context(), reward_areas...)
    end

    # function render_special_reward_areas(pomdp::POMDP, size)
    #     nx, ny = size
    #     reward_areas = []
    #     x = 7
    #     y = 10
    #     ctx = cell_ctx((x, y), (nx, ny))
    #     reward_area = compose(ctx, star(0.5, 0.5, 0.3, 5, 0.5), fill("blue"), Compose.stroke("black"))
    #     push!(reward_areas, reward_area)
    #     x2 = 6
    #     y2 = 10
    #     ctx = cell_ctx((x2, y2), (nx, ny))
    #     reward_area = compose(ctx, star(0.5, 0.5, 0.3, 5, 0.5), fill("blue"), Compose.stroke("black"))
    #     push!(reward_areas, reward_area)
    #     return compose(context(), reward_areas...)
    # end

    function render_obstacles(pomdp::POMDP, size)
        nx, ny = size
        obstacles = []
        for obstacle in pomdp.obstacles
            ctx = cell_ctx((obstacle[1], obstacle[2]), (nx, ny))
            obs = compose(ctx, Compose.rectangle(), fill("red"), fillopacity(0.5))
            push!(obstacles, obs)
        end
        return compose(context(), obstacles...)
    end

    function render_agent(ctx, battery_level)
        battery_color = get(ColorSchemes.RdYlGn, battery_level)
        center = compose(context(), Compose.circle(0.5, 0.5, 0.3), fill(battery_color), Compose.stroke("black"))
        lwheel = compose(context(), ellipse(0.2, 0.5, 0.1, 0.3), fill("gray"), Compose.stroke("black"))
        rwheel = compose(context(), ellipse(0.8, 0.5, 0.1, 0.3), fill("gray"), Compose.stroke("black"))
        return compose(ctx, center, lwheel, rwheel)
    end

    function render_action_text(pomdp::POMDP, step, pre_act_text)
        actions = ["Action: Sense", "Action: Up", "Action: Down", "Action: Left", "Action: Right"]
        

        
        action_text = "Terminal"
        
        if step.a == -5
            action_text = "Waiting for Action"
        elseif get(step, :a, nothing) !== nothing
            action_text = actions[step.a]
        end

        action_text = pre_act_text * action_text


        if step[:s].pos == GWPos(3, 5)
            action_text = ""
        end

        _, ny = pomdp.size
        ny += 1
        ctx = context(0, (ny - 1) / ny, 1, 1 / ny)
        txt = compose(ctx, Compose.text(0.5, 0.5, action_text, hcenter),
            Compose.stroke("black"),
            fill("black"),
            fontsize(20pt))
        return compose(ctx, txt, Compose.rectangle(), fill("white"))
    end

    function render_action(pomdp::POMDP, step)
        if step.a == 1  # Refuel action
            ctx = cell_ctx((step[:s].pos.x, step[:s].pos.y), pomdp.size .+ (1, 1))
            return compose(ctx, star(0.5, 0.5, 0.2, 5, 0.5), Compose.stroke("blue"), fill("yellow"))
        elseif step.a == 6  # Sense action
                ctx = cell_ctx((step[:s].pos.x, step[:s].pos.y), pomdp.size .+ (1, 1))
            return compose(ctx, star(0.5, 0.5, 0.2, 5, 0.5), Compose.stroke("blue"), fill("yellow"))
        elseif step.a in 2:5  # Movement actions
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dir = dirs[step.a - 1]
            ctx = cell_ctx((step[:s].pos.x, step[:s].pos.y), pomdp.size .+ (1, 1))
            return compose(ctx, custom_arrow(0.5, 0.5, 0.5 + 0.3*dir[1], 0.5 + 0.3*dir[2]), 
                        Compose.stroke("blue"), linewidth(2mm), fill("blue"))
        end
        return nothing
    end

    function custom_arrow(x1, y1, x2, y2)
        # Calculate the angle of the line
        angle = atan(y2 - y1, x2 - x1)
        
        # Calculate the points for the arrowhead
        arrowsize = 0.1
        ax1 = x2 - arrowsize * cos(angle - pi/6)
        ay1 = y2 - arrowsize * sin(angle - pi/6)
        ax2 = x2 - arrowsize * cos(angle + pi/6)
        ay2 = y2 - arrowsize * sin(angle + pi/6)
        
        # Create the arrow shape
        arrow_line = line([(x1, y1), (x2, y2)])
        arrowhead = polygon([(x2, y2), (ax1, ay1), (ax2, ay2)])
        
        return (context(), arrow_line, arrowhead)
    end
end
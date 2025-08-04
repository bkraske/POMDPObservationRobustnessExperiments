// SDBatteryPOMDP   size: Tuple{Int64, Int64}   max_fuel: Int64 30   sand_types: Int64 4   rewards: Dict{SVector{2, Int64}, Float64}   sand: Set{SVector{2, Int64}}   obstacles: Set{SVector{2, Int64}}   terminate_from: Int64 0   terminal_state: SDRoverState   battery_uncertainty: Float64 1.0   sensor_efficiency_stop: Float64 0.8   sensor_efficiency_load: Float64 0.6   sand_sensor_efficiency: Float64 0.99   discount: Float64 0.99   damagedstates: Array{Int64}((2,)) [-1, 0]   damaged_sand_sensor_efficiency: Array{Float64}((0,)) Float64[]   stateindices: Array{Int64}((2,)) [4, 4]   candamage: Int64 0   initialdamage: Array{Int64}((1,)) [0]   per_step_rew: Float64 -0.0   bad_sands: Array{Int64}((2,)) [2, 3]  
 dtmc 

//Parameters 
const double p1; 
const double p2; 
const double p3; 
const double p4; 
const double p5; 
const double p6; 
const double p7; 
const double p8; 

//Transitions 
module pomdp_mc 
	 s: [0..25] init 0; 
	 [] s=0 ->  0.25: (s'=1)+ 0.25: (s'=6)+ 0.25: (s'=11)+ 0.25: (s'=16);
	 [] s=1 ->  1.0*p1: (s'=2)+ 1.0*(1.0-p1): (s'=3);
	 [] s=2 ->  1.0*p5: (s'=4)+ 1.0*(1.0-p5): (s'=5);
	 [] s=3 ->  1.0*(1.0-p5): (s'=4)+ 1.0*p5: (s'=5);
	 [] s=4 ->  1.0*(1.0): (s'=24);
	 [] s=5 ->  1.0*(1.0): (s'=25);
	 [] s=6 ->  1.0*p2: (s'=7)+ 1.0*(1.0-p2): (s'=8);
	 [] s=7 ->  1.0*p6: (s'=9)+ 1.0*(1.0-p6): (s'=10);
	 [] s=8 ->  1.0*(1.0-p6): (s'=9)+ 1.0*p6: (s'=10);
	 [] s=9 ->  1.0*(1.0): (s'=24);
	 [] s=10 ->  1.0*(1.0): (s'=25);
	 [] s=11 ->  1.0*p3: (s'=12)+ 1.0*(1.0-p3): (s'=13);
	 [] s=12 ->  1.0*p7: (s'=14)+ 1.0*(1.0-p7): (s'=15);
	 [] s=13 ->  1.0*(1.0-p7): (s'=14)+ 1.0*p7: (s'=15);
	 [] s=14 ->  1.0*(1.0): (s'=24);
	 [] s=15 ->  1.0*(1.0): (s'=25);
	 [] s=16 ->  1.0*p4: (s'=17)+ 1.0*(1.0-p4): (s'=18);
	 [] s=17 ->  1.0*p8: (s'=19)+ 1.0*(1.0-p8): (s'=20);
	 [] s=18 ->  1.0*(1.0-p8): (s'=19)+ 1.0*p8: (s'=20);
	 [] s=19 ->  1.0*(1.0): (s'=24);
	 [] s=20 ->  1.0*(1.0): (s'=25);
	 [] s=21 ->  1.0: (s'=21);
	 [] s=22 ->  1.0: (s'=22);
	 [] s=23 ->  1.0: (s'=23);
	 [] s=24 ->  1.0: (s'=24);
	 [] s=25 ->  1.0: (s'=25);
endmodule 

//Rewards 
rewards 
	 s=4 : 1.0;
	 s=5 : 0.9;
	 s=10 : 0.9;
	 s=15 : 0.9;
	 s=19 : 1.0;
	 s=20 : 0.9;
endrewards
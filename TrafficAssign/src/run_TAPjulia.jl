using DelimitedFiles
include("TrafficAssign.jl")
link_flow, link_travel_time = solve_TAP("tmp", "../../Joint/tmp_jl/net.txt", "../../Joint/tmp_jl/trips.txt", [1,0,0,0,.15,0])
#link_flow, link_travel_time = solve_TAP("tmp", "../../networks/NYC_Uber_net.txt", "../../networks/NYC_Uber_trips.txt", [1,0,0,0,.15,0])

writedlm("../../Joint/tmp_jl/link_flow.txt", link_flow)
writedlm("../../Joint/tmp_jl/link_travel_time.txt", link_travel_time)
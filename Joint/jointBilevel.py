
class transNet:
    def __init__(self, tNet):
        self.incidence_mat


def set_idOD(gDict):
	i =0
	for (s,t), d in gDict.items:
		idOD_dict[(s,t)] = i
		i +=1
	return idOD_dict

def dx_db(TAP, gDict, fcoeffs, delta):

	
	dxdb = [ for i in len(fcoeffs)]
	for i in len(fcoeffs): # this loop can be paralellized






function df_db(demandsDict, fcoeffs, delta, tapFlowVecDict)
	#find derivatives of flows w/respect to changes in the cost function
	tap_flow_dict = Dict()
	tap_vec_dict = Dict()
	dxdb = zeros(length(tapFlowVecDict),0)
	fcoeffs2 = []
	for i = 1:length(fcoeffs) # this loop can be paralelized
	    fcoeffs2 = []
	    append!(fcoeffs2, fcoeffs)
	    fcoeffs2[i] = fcoeffs2[i] + delta
	    #println(demandsDict)
	    #println(fcoeffs2)
	    tap_flow_dict[i], tap_vec_dict[i] = tapMSA(demandsDict, fcoeffs2);
	    dxdb_i = (tap_vec_dict[i]-tapFlowVecDict)/delta;
	    #println(dxdb_i)
	    dxdb = [dxdb  dxdb_i];
	    #println(fcoeffs)
	    #println(fcoeffs2)
	end  
	return dxdb
end
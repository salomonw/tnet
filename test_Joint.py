import os
'''
tnet_dir = os.getcwd()[:-5]
cmd = 'push!(LOAD_PATH, "'+ tnet_dir+ '")'
from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval(cmd)
from julia import TrafficAssign
'''
from joint.utils import *
import joint.tnet as tnet
import joint.msa as msa
import copy
from matplotlib import pyplot as plt


def set_up():
	netFile ="networks/EMA_net.txt"
	gFile = "networks/EMA_trips.txt"
	# Build a ground truth network
	fcoeffs_truth = [1,0,0,0,0.20,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)
	tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
	#tNet.solveMSA()
	#print([tNet.G[s][t]["flow"] for (s,t) in tNet.G.edges()])
	

	G_data = tNet.G.copy()
	g_data = tNet.g.copy()

	# Now, use the data to try to estimate demands, to do so 
	# let us perturb the truth demands. 
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.5)
	fcoeff = [1,0,0,0,0.15,0]
	return G_data, g_data, g_k, fcoeff, tNet



def solve_od_fcoffs(G_data, g_data, g_k, fcoeff, tNet, opt_method, iterations):
	print("--------------------------------------------------------------------------------------")
	print("| \t n  \t | \t Flow norm \t | \t Demand norm \t | \t fcoeffs \t |")
	print("--------------------------------------------------------------------------------------")
	flowNormList = []
	logtime_data = {}
	for i in range(iterations):
		# update network and MSA
		tNet.set_g(g_k)
		tNet.set_fcoeffs(fcoeff)
		tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
		
		# estimate derivatives
		if opt_method=='constant':
			dxdb = False
		else:
			dxdb = tnet.get_dxdb(tNet, delta=0.1, divide=1, log_time=logtime_data)
		
		dxdg = msa.get_dxdg(tNet.G, tNet.g, k =1)
		#print(dxdg)
		# set trust regions
		g_tr = 200/((i+1)**(1/2))
		beta_tr = 0.005/((i+1)**(1/2))
		
		# Optimize:
		if opt_method == "gd": 
		# gradient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
		if opt_method == "joint":
		# solve joint bilevel
			g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e10, c=30, lambda_1=0)
			#g_k, fcoeff = tNet.solve_jointBilevel_julia(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e2, c=30, lambda_1=0)
		if opt_method == "constant":
			dxdb = False
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}			


		fcoeff[0] = 1

		# gathering statistics
		flowNorm = tnet.normFlowDiff(G_data, tNet.G)
		gNorm = tnet.gDiff(g_data, tNet.g)

		flowNormList.append(flowNorm)
		#print(fcoeff)
		#print([tNet.G[s][t]["flow"] for (s,t) in tNet.G.edges()])
		formatted_fcoeffs = ["%.2f"%item for item in fcoeff]
		print("| \t {n:.0f} \t | \t {f:.2f} \t | \t {g:.2f} \t | \t {coeff} \t |".format(\
				n=i, f=flowNorm, g=gNorm, coeff=str(formatted_fcoeffs), flowData=[G_data.get_edge_data(s,t)['flow'] for s,t in G_data.edges()] , gk=tNet.g, \
				flowEstimate = [tNet.G.get_edge_data(s,t)['flow'] for s,t in G_data.edges()]))

	return flowNormList


#to_solve = ["GD", "joint", "constant"]
to_solve = ["GD", "constant"]
#to_solve = ["GD"]
#to_solve = ["constant"]
#to_solve = ["joint"]

iterations = 100

G_data, g_data, g_0, fcoeffs_0, tNet = set_up()
x_axis  = [i+1 for i in range(iterations-1)]
tNet_0 = copy.deepcopy(tNet)
if "GD" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormGD = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "gd", iterations)
	plt.plot(x_axis, flowNormGD[1:], label='GD')
if "joint" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormJOINT = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "joint", iterations)
	plt.plot(x_axis, flowNormJOINT[1:], label='joint')
if "constant" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormConstant = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "constant", iterations)
	plt.plot(x_axis, flowNormConstant[1:], label='Constant f')

plt.xlabel("Iteration (j)")
plt.ylabel("Flow error")
plt.legend()
plt.show()
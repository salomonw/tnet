from utils import *
import tnet
import msa
import copy
from matplotlib import pyplot as plt


def set_up():
	netFile = "../networks/Braess2_net.txt"
	gFile = "../networks/Braess2_trips.txt"

	# Build a ground truth network
	fcoeffs_truth = [1,0,0,0,0.2,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)
	tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
	#tNet.solveMSA()
	print([tNet.G[s][t]["flow"] for (s,t) in tNet.G.edges()])


	G_data = tNet.G.copy()
	g_data = tNet.g.copy()

	# Now, use the data to try to estimate demands, to do so 
	# let us perturb the truth demands. 
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.5)
	fcoeff = [1,0,0,0,0.15,0]
	return G_data, g_data, g_k, fcoeff, tNet



def solve_od_fcoffs(G_data, g_data, g_k, fcoeff, tNet, opt_method):
	print("--------------------------------------------------------------------------------------")
	print("| \t n  \t | \t Flow norm \t | \t Demand norm \t | \t fcoeffs \t |")
	print("--------------------------------------------------------------------------------------")
	flowNormList = []
	for i in range(100):
		# update network and MSA
		tNet.set_g(g_k)
		tNet.set_fcoeffs(fcoeff)
		tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
		# estimate derivatives
		dxdb = tnet.get_dxdb(tNet, delta=0.1, divide=1e1)
		dxdg = msa.get_dxdg(tNet.G, tNet.g, k =1)
		# set trust regions
		g_tr = 300/(i+1)
		beta_tr = 0.05/(i+1)
		
		# Optimize:
		if opt_method == "gd": 
		# gradient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v-Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
		elif opt_method == "joint":
		# solve joint bilevel
			g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e6, c=10, lambda_1=0)
			#g_k, fcoeff = tNet.solve_jointBilevel_julia(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=0, c=1, lambda_1=10)
		else:
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v-Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}			
		fcoeff[0] = 1

		# gathering statistics
		flowNorm = tnet.normFlowDiff(G_data, tNet.G)
		gNorm = tnet.gDiff(g_data, tNet.g)

		flowNormList.append(flowNorm)

		formatted_fcoeffs = ["%.2f"%item for item in tNet.fcoeffs]
		print("| \t {n:.0f} \t | \t {f:.2f} \t | \t {g:.2f} \t | \t {coeff} \t |".format(\
				n=i, f=flowNorm, g=gNorm, coeff=str(formatted_fcoeffs), flowData=[G_data.get_edge_data(s,t)['flow'] for s,t in G_data.edges()] , gk=tNet.g, \
				flowEstimate = [tNet.G.get_edge_data(s,t)['flow'] for s,t in G_data.edges()]))

	return flowNormList


G_data, g_data, g_0, fcoeffs_0, tNet = set_up()
tNet_0 = copy.deepcopy(tNet)
flowNormGD = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "gd")
tNet = copy.deepcopy(tNet_0)
flowNormJOINT = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "joint")
tNet = copy.deepcopy(tNet_0)
flowNormConstant = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "constant")

print("-- flow norm GD --- ")
print(flowNormGD)
print("-- flow norm Joint --- ")
print(flowNormJOINT)
print("-- flow norm Constant --- ")
print(flowNormConstant)

x_axis  = [i+1 for i in range(100)]
plt.plot(x_axis, flowNormGD, label='GD')
plt.plot(x_axis, flowNormJOINT, label='Joint')
plt.plot(x_axis, flowNormConstant, label='Constant f')
plt.xlabel("Iteration (j)")
plt.ylabel("Flow error")
plt.legend()


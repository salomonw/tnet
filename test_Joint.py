import os
from joint.utils import *
import joint.tnet as tnet
import joint.msa as msa
import copy
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

def set_up():
	netFile ="networks/NYC_small_net.txt"
	gFile = "networks/NYC_small_trips.txt"
	# Build a ground truth network
	fcoeffs_truth = [1,0,0,0,0.35,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)
	
	tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
	G_data = tNet.G.copy()
	g_data = tNet.g.copy()

	# Now, use the data to try to estimate demands, to do so 
	# let us perturb the truth demands. 
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.5)
	fcoeff = [1,0,0,0,0.15,0]
	return G_data, g_data, g_k, fcoeff, tNet, fcoeffs_truth



def solve_od_fcoffs(G_data, g_data, g_k, fcoeff, tNet, opt_method, iterations):
	print("--------------------------------------------------------------------------------------")
	print("| \t n  \t | \t Flow norm \t | \t Demand norm \t | \t fcoeffs \t |")
	print("--------------------------------------------------------------------------------------")
	flowNormList = []
	gNormlist = []
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
		g_tr = 150/((i+1))#**(3/4))
		beta_tr = 0.01/((i+1))#**(1/1))
		
		# Optimize:
		if opt_method == "gd": 
		# gradient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
		if opt_method == "Joint":
		# solve joint bilevel
			g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e10, c=30, lambda_1=0)
			#g_k, fcoeff = tNet.solve_jointBilevel_julia(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e2, c=30, lambda_1=0)
		if opt_method == "constant":
			dxdb = False
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}			
		if opt_method =="sequential":
			# move on the cost coefficient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
			tNet.set_fcoeffs(fcoeff)
			tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
			# move step in demand desecent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=False, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}				




		fcoeff[0] = 1

		# gathering statistics
		flowNorm = tnet.normFlowDiff(G_data, tNet.G)
		gNorm = tnet.gDiff(g_data, tNet.g)

		flowNormList.append(flowNorm)
		gNormlist.append(gNorm)
		#print(fcoeff)
		#print([tNet.G[s][t]["flow"] for (s,t) in tNet.G.edges()])
		formatted_fcoeffs = ["%.2f"%item for item in fcoeff]
		print("| \t {n:.0f} \t | \t {f:.2f} \t | \t {g:.2f} \t | \t {coeff} \t |".format(\
				n=i, f=flowNorm, g=gNorm, coeff=str(formatted_fcoeffs), flowData=[G_data.get_edge_data(s,t)['flow'] for s,t in G_data.edges()] , gk=tNet.g, \
				flowEstimate = [tNet.G.get_edge_data(s,t)['flow'] for s,t in G_data.edges()]))

	return flowNormList, gNormlist, fcoeff


to_solve = ["constant", "GD", "sequential", "Joint"]
#to_solve = ["GD", "constant", "sequential"]
#to_solve = ["GD",  "constant", "sequential"]
#to_solve = ["GD", "Joint", "constant"]
#to_solve = ["GD", "constant"]
#to_solve = ["GD"]
#to_solve = ["constant"]
#to_solve = ["Joint"]
#to_solve = ['sequential']

rc('font',**{'family':'Times New Roman', 'size': 16})
rc('text', usetex=True)


iterations = 60
fcoeffs_list = []
G_data, g_data, g_0, fcoeffs_0, tNet, fcoeffs_truth = set_up()
x_axis  = [i+1 for i in range(iterations-1)]
tNet_0 = copy.deepcopy(tNet)
if "GD" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormGD, gNormGD, fcoeffGD = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "gd", iterations)
	fcoeffs_list.append(fcoeffGD)
if "Joint" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormJOINT, gNormJOINT, fcoeffJOINT = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "Joint", iterations)
	fcoeffs_list.append(fcoeffJOINT)
if "constant" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormConstant, gNormConstant, fcoeffConstant = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "constant", iterations)
	fcoeffs_list.append(fcoeffConstant)
if "sequential" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	flowNormSequential, gNormSequential, fcoeffSequential = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "sequential", iterations)
	fcoeffs_list.append(fcoeffSequential)


plt.figure()
plt.plot(x_axis, flowNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
plt.plot(x_axis, flowNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
plt.plot(x_axis, flowNormSequential[1:], label='Sequential', marker=11, markevery=(0,25))
plt.plot(x_axis, flowNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
plt.xlabel("Iteration,$j$")
plt.ylabel("Flow error, $F(\\mathbf{g}, \\boldmath{\\beta})$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig('errorCost_single_'+tNet.netFileName[9:-8]+'.png')


plt.figure()
plt.plot(x_axis, gNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
plt.plot(x_axis, gNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
plt.plot(x_axis, gNormSequential[1:], label='Sequential', marker=11, markevery=(0,25))
plt.plot(x_axis, gNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
plt.xlabel("Iteration,$j$")
plt.ylabel("Demand error, $||(\\mathbf{g} - \\mathbf{g}^{*})||$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig('errorDemand_single_'+tNet.netFileName[9:-8]+'.png')




def f_cost(x, fcoe):
	return sum([fcoe[i]*x**i for i in range(len(fcoe))])

labels = to_solve.copy()
labels.append('ground truth')
labels[labels.index('constant')] = '$f(\\cdot) =$ BPR'
markers = ['o', 'D', 's', '^', 11, '*']
fcoeffs_list.append(fcoeffs_truth)
plt.figure()
x =  np.linspace(0, 1.5, 10, endpoint=True)
i=0
for fcoeffs in fcoeffs_list:
	y = [f_cost(float(j),fcoeffs) for j in x]
	plt.plot(x, y, label=labels[i], marker=markers[i])
	i+=1

plt.xlabel("$x/m$")
plt.ylabel("Travel time function, $f(x/m)$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig('function_'+tNet.netFileName[9:-8]+'.png')
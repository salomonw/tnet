import os
from joint.utils import *
import joint.tnet as tnet
import joint.msa as msa
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from datetime import datetime

def set_up():
	netFile ="networks/EMA_net.txt"
	flowFile = "networks/EMA_flow.txt"
	gFile = "networks/EMA_trips.txt"
	fcoeff = [1,0,0,0,0.15,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeff)
	
	tNet.read_flow_file(flowFile)
	G_data = tNet.G.copy()
	g_data = tNet.g.copy()

	# set iteration start from zero 
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.5)
	
	c1 = 50
	d1 = 0.01

	to_solve = ["constant", "GD", "alternating", "Joint"]
	iterations = 1

	tstamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	dir_out = tstamp + "_test_" + tNet.netFileName[9:-8]
	# Create directories to store results
	mkdir_n('results/')
	mkdir_n('results/joint')
	mkdir_n('results/joint/' + dir_out)
	mkdir_n('results/joint/' + dir_out +'/iterations')
	mkdir_n('results/joint/' + dir_out +'/graphs')
	mkdir_n('results/joint/' + dir_out +'/output')
	dir_out = 'results/joint/' + dir_out 

	return G_data, g_data, g_k, fcoeff, tNet, fcoeff, c1, d1, to_solve, iterations, dir_out



def solve_od_fcoffs(G_data, g_data, g_k, fcoeff, tNet, opt_method, iterations, c1, d1):
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
			dxdb = tnet.get_dxdb(tNet, delta=0.05, divide=1, log_time=logtime_data)
		
		dxdg = msa.get_dxdg(tNet.G, tNet.g, k =1)

		# set trust regions
		g_tr = c1/(((i+1)))**(1/2)
		beta_tr = d1/((i+1)**(1/2))
		
		# Optimize:
		if opt_method == "gd": 
		# gradient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
		if opt_method == "Joint":
		# solve joint bilevel
			g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e2, c=30, lambda_1=0.0)
		if opt_method == "constant":
			dxdb = False
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}			
		if opt_method =="alternating":
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

	return tNet, flowNormList, gNormlist, fcoeff




fcoeffs_list = []
parameters = set_up()
G_data, g_data, g_0, fcoeff_0, tNet, fcoeffs_0, c1, d1, to_solve, iterations, dir_out = parameters
x_axis  = [i+1 for i in range(iterations-1)]
tNet_0 = copy.deepcopy(tNet)
if "constant" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormConstant, gNormConstant, fcoeffConstant = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "constant", iterations, c1, d1)
	fcoeffs_list.append(fcoeffConstant)
	dict2csv(tNet.g, "results/joint/output/"+tNet.netFileName[9:-8]+'_OD_demand'+ '.csv')
	list2txt(tNet.fcoeffs, "results/joint/output/"+tNet.netFileName[9:-8]+'_costFunct'+ '.txt')
if "GD" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormGD, gNormGD, fcoeffGD = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "gd", iterations, c1, d1)
	fcoeffs_list.append(fcoeffGD)
	dict2csv(tNet.g, "results/joint/output/"+tNet.netFileName[9:-8]+'_OD_demand'+ '.csv')
	list2txt(tNet.fcoeffs, "results/joint/output/"+tNet.netFileName[9:-8]+'_costFunct'+ '.txt')
if "alternating" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormAlternating, gNormAlternating, fcoeffAlternating = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "alternating", iterations, c1, d1)
	fcoeffs_list.append(fcoeffAlternating)
	dict2csv(tNet.g, "results/joint/output/"+tNet.netFileName[9:-8]+'_OD_demand'+ '.csv')
	list2txt(tNet.fcoeffs, "results/joint/output/"+tNet.netFileName[9:-8]+'_costFunct'+ '.txt')
if "Joint" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormJOINT, gNormJOINT, fcoeffJOINT = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "Joint", iterations, c1, d1)
	fcoeffs_list.append(fcoeffJOINT)
	dict2csv(tNet.g, "results/joint/output/"+tNet.netFileName[9:-8]+'_OD_demand'+ '.csv')
	list2txt(tNet.fcoeffs, "results/joint/output/"+tNet.netFileName[9:-8]+'_costFunct'+ '.txt')



# Generate Plots

rc('font',**{'family':'Times New Roman', 'size': 16})
rc('text', usetex=True)
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.figure()
plt.plot(x_axis, flowNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
plt.plot(x_axis, flowNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
plt.plot(x_axis, flowNormAlternating[1:], label='Alternating', marker=11, markevery=(0,25))
#plt.plot(x_axis, flowNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
plt.xlabel("Iteration,$j$")
plt.ylabel("Flow error, $F(\\mathbf{g}, \\boldsymbol{\\beta})$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig(dir_out +'/graphs/'+tNet.netFileName[9:-8]+'_errorCost_single.png')


plt.figure()
plt.plot(x_axis, gNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
plt.plot(x_axis, gNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
plt.plot(x_axis, gNormAlternating[1:], label='Alternating', marker=11, markevery=(0,25))
#plt.plot(x_axis, gNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
plt.xlabel("Iteration,$j$")
plt.ylabel("Demand error, $||(\\mathbf{g} - \\mathbf{g}^{*})||$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig(dir_out +'/graphs/'+tNet.netFileName[9:-8]+'_errorDemand_single.png')



def f_cost(x, fcoe):
	return sum([fcoe[i]*x**i for i in range(len(fcoe))])

labels = to_solve.copy()
labels.append('$\\boldsymbol{\\beta}^0$')
labels[labels.index('constant')] = '$f(\\cdot) =$ BPR'
markers = ['o', 'D', 's', '^', '*']
fcoeffs_list.append(fcoeffs_0)
plt.figure()
x =  np.linspace(0, 1.5, 10, endpoint=True)
i=0
for i in range(len(labels)):
	fcoeffs =fcoeffs_list[i]
	y = [f_cost(float(j),fcoeffs) for j in x]
	plt.plot(x, y, label=labels[i], marker=markers[i])
	i+=1

plt.xlabel("$x/m$")
plt.ylabel("Travel time function, $f(x/m)$")
plt.legend(framealpha=1)
plt.grid(linestyle='--', linewidth=1)
plt.savefig(dir_out +'/graphs/'+tNet.netFileName[9:-8]+'_costFunct.png')

# Save results to files

list2txt(flowNormConstant[1:], dir_out + "/iterations/"+tNet.netFileName[9:-8]+'_flowNormConstant.txt')
list2txt(flowNormGD[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormGD.txt')
list2txt(flowNormAlternating[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormAlternating.txt')
list2txt(flowNormJOINT[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormJOINT.txt')
list2txt(gNormConstant[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_gNormConstant.txt')
list2txt(gNormGD[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_gNormGD.txt')
list2txt(gNormAlternating[1:], dir_out + "/iterations/"+tNet.netFileName[9:-8]+'_gNormAlternating.txt')
list2txt(gNormJOINT[1:], dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormJOINT.txt')
list2txt(parameters, dir_out +"/"+tNet.netFileName[9:-8]+'_parameters.txt')

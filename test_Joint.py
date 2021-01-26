import os
from joint.utils import *
import joint.tnet as tnet
import joint.msa as msa
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from datetime import datetime
 

def set_up():
	netFile ="networks/EMA_net.txt"
	gFile = "networks/EMA_trips.txt"
	# Build a ground truth network
	fcoeffs_truth = [1,0,0,0,0.45,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)
	
	tNet = tnet.solveMSA_julia(tNet, tNet.fcoeffs)
	

	G_data = tNet.G.copy()
	g_data = tNet.g.copy()

	c1 = 250
	d1 = 0.02

	# Now, use the data to try to estimate demands, to do so 
	# let us perturb the truth demands. 
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.1)
	fcoeff = [1,0,0,0,0.15,0]

	to_solve = ["constant", "GD", "alternating", "Joint"]
	iterations = 30				

	tstamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	dir_out = tstamp + "_test_" + tNet.netFileName[9:-8]

	generate_plots = True

	# Create directories to store results
	mkdir_n('results/')
	mkdir_n('results/joint')
	mkdir_n('results/joint/' + dir_out)
	mkdir_n('results/joint/' + dir_out +'/iterations')
	mkdir_n('results/joint/' + dir_out +'/graphs')
	mkdir_n('results/joint/' + dir_out +'/output')
	dir_out = 'results/joint/' + dir_out 

	return [G_data, g_data, g_k, fcoeff, tNet, c1, d1, fcoeffs_truth, to_solve, iterations, dir_out, generate_plots]



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
			dxdb = tnet.get_dxdb(tNet, delta=0.1, divide=1, log_time=logtime_data)
		
		dxdg = msa.get_dxdg(tNet.G, tNet.g, k =1)

		# set trust regions
		g_tr = c1/(((i+1)))#**(1/2)
		beta_tr = d1/((i+1))**(1/2)
		
		# Optimize:
		if opt_method == "gd": 
		# gradient descent
			Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
			g_k = {k: max(min(max(v - Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
			fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
		if opt_method == "Joint":
		# solve joint bilevel
			g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=1e10, c=30, lambda_1=0.1)
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


# Run experiments for different algorithms

fcoeffs_list = []
parameters = set_up()
[G_data, g_data, g_0, fcoeffs_0, tNet, c1, d1, fcoeffs_truth, to_solve, iterations, dir_out, generate_plots] = parameters
tNet.write_flow_file(dir_out +"/"+tNet.netFileName[9:-8]+'_initial_flow_j.txt')
x_axis  = [i+1 for i in range(iterations-1)]
tNet_0 = copy.deepcopy(tNet)
if "constant" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormConstant, gNormConstant, fcoeffConstant = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "constant", iterations, c1, d1)
	fcoeffs_list.append(fcoeffConstant)
	dict2csv(tNet.g, dir_out + "/output/"+tNet.netFileName[9:-8]+'_OD_demand_c'+ '.csv')
	list2txt(tNet.fcoeffs, dir_out + "/output/"+tNet.netFileName[9:-8]+'_costFunct_c'+ '.txt')
if "GD" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormGD, gNormGD, fcoeffGD = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "gd", iterations, c1, d1)
	fcoeffs_list.append(fcoeffGD)
	dict2csv(tNet.g, dir_out +"/output/"+tNet.netFileName[9:-8]+'_OD_demand_gd'+ '.csv')
	list2txt(tNet.fcoeffs, dir_out +"/output/"+tNet.netFileName[9:-8]+'_costFunct_gd'+ '.txt')
if "alternating" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormAlternating, gNormAlternating, fcoeffAlternating = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "alternating", iterations, c1, d1)
	fcoeffs_list.append(fcoeffAlternating)
	dict2csv(tNet.g, dir_out + "/output/"+tNet.netFileName[9:-8]+'_OD_demand_a'+ '.csv')
	list2txt(tNet.fcoeffs, dir_out + "/output/"+tNet.netFileName[9:-8]+'_costFunct_a'+ '.txt')
if "Joint" in to_solve:
	tNet = copy.deepcopy(tNet_0)
	tNet, flowNormJOINT, gNormJOINT, fcoeffJOINT = solve_od_fcoffs(G_data, g_data, g_0, fcoeffs_0, tNet, "Joint", iterations, c1, d1)
	fcoeffs_list.append(fcoeffJOINT)
	dict2csv(tNet.g, dir_out + "/output/"+tNet.netFileName[9:-8]+'_OD_demand_j'+ '.csv')
	list2txt(tNet.fcoeffs, dir_out + "/output/"+tNet.netFileName[9:-8]+'_costFunct_j'+ '.txt')
	tNet.write_flow_file(dir_out +"/output/"+tNet.netFileName[9:-8]+'_estimated_flow_j.txt')



# Save results to files

list2txt(flowNormConstant, dir_out + "/iterations/"+tNet.netFileName[9:-8]+'_flowNormConstant.txt')
list2txt(flowNormGD, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormGD.txt')
list2txt(flowNormAlternating, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormAlternating.txt')
list2txt(flowNormJOINT, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_flowNormJOINT.txt')
list2txt(gNormConstant, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_gNormConstant.txt')
list2txt(gNormGD, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_gNormGD.txt')
list2txt(gNormAlternating, dir_out + "/iterations/"+tNet.netFileName[9:-8]+'_gNormAlternating.txt')
list2txt(gNormJOINT, dir_out +"/iterations/"+tNet.netFileName[9:-8]+'_gNormJOINT.txt')
list2txt(parameters, dir_out +"/"+tNet.netFileName[9:-8]+'_parameters.txt')


# Identify if the demand estimation is 'good enough'
if np.mean(flowNormJOINT)/np.mean(flowNormConstant) - 1 < .1:
	list2txt(parameters, dir_out +"/"+tNet.netFileName[9:-8]+'YEY')
else:
	list2txt(parameters, dir_out +"/"+tNet.netFileName[9:-8]+'NEY')


# Generate Plots
if generate_plots:
	mpl.rc('font',**{'family':'Times New Roman', 'size': 16})
	mpl.rc('text', usetex=True)
	mpl.rc('text', usetex=True)
	mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	plt.figure()
	plt.plot(x_axis, flowNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
	plt.plot(x_axis, flowNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
	plt.plot(x_axis, flowNormAlternating[1:], label='Alternating', marker=11, markevery=(0,25))
	plt.plot(x_axis, flowNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
	plt.xlabel("Iteration,$j$")
	plt.ylabel("Flow error, $F(\\mathbf{g}, \\boldsymbol{\\beta})$")
	plt.legend(framealpha=1)
	plt.grid(linestyle='--', linewidth=1)
	plt.savefig(dir_out +'/graphs/'+tNet.netFileName[9:-8]+'_errorCost_single.png')

	plt.figure()
	plt.plot(x_axis, gNormConstant[1:], label='$f(\\cdot) =$ BPR', marker='s', markevery=(0,20))
	plt.plot(x_axis, gNormGD[1:], label='GD', marker = 'o', markevery=(0,10))
	plt.plot(x_axis, gNormAlternating[1:], label='Alternating', marker=11, markevery=(0,25))
	plt.plot(x_axis, gNormJOINT[1:], label='Joint', marker = 'o', markevery=(0,10))
	plt.xlabel("Iteration,$j$")
	plt.ylabel("Demand error, $||(\\mathbf{g} - \\mathbf{g}^{*})||$")
	plt.legend(framealpha=1)
	plt.grid(linestyle='--', linewidth=1)
	plt.savefig(dir_out +'/graphs/'+tNet.netFileName[9:-8]+'_errorDemand_single.png')

	def f_cost(x, fcoe):
		return sum([fcoe[i]*x**i for i in range(len(fcoe))])

	labels = to_solve.copy()
	labels.append('ground truth')
	labels[labels.index('constant')] = '$f(\\cdot) =$ BPR'
	markers = ['o', 'D', 's', '^', '*']
	fcoeffs_list.append(fcoeffs_truth)
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



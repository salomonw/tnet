import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np 

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

plt.style.use(['science', 'ieee', 'high-vis'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
 #   "font.serif": ["Times"],  # specify font here
    "font.size":8,			  # specify font size here
   # "axes.linewidth":0.4,
#	'xtick.major.width' : 0.4,
#	'xtick.minor.width' : 0.2,
#	'ytick.major.width' : 0.4,
#	'ytick.minor.width' : 0.2,
#	'lines.linewidth' : lwidth,
	'patch.linewidth' : 0.2,
	'legend.fancybox' : False,
	'legend.framealpha': 0.92,
	'legend.edgecolor': 'k',
#	'legend.loc': 'upper right',
    })                    


# Read results 
def txt2list(fname):
	a_file = open(fname, "r")
	list_of_lists = [(line.strip()).split() for line in a_file]
	flat_list = [float(item) for sublist in list_of_lists for item in sublist]
	a_file.close()
	return flat_list


def f_cost(x, fcoe):
	return sum([fcoe[i]*x**i for i in range(len(fcoe))])

def plot_f(fcoeffs_list, ax):
	labels = ['$f(\\cdot) =$ BPR', "GD", "Alternating", "Joint"]
	#labels.append('$\\boldsymbol{\\beta}^0$')
	#labels[labels.index('constant')] = '$f(\\cdot) =$ BPR'
	markers = ['o', 'D', 's', '^', '*']
	
	x =  np.linspace(0, 1.5, 10, endpoint=True)
	i=0
	for i in range(len(labels)):
		fcoeffs =fcoeffs_list[i]
		y = [f_cost(float(j),fcoeffs) for j in x]
		if labels[i] == 'Joint':
			ax.plot(x, y, '-', label=labels[i], marker=markers[i], markersize=3)
		else:
			ax.plot(x, y, '--', label=labels[i], marker=markers[i], markersize=3)
		i+=1

	ax.set_xlabel("$x/m$")
	ax.set_ylabel("Travel time function, $f(x/m)$")
	ax.legend(framealpha=0.8, frameon=True)
	ax.set_xlim((0,1.5))
	plt.tight_layout()
	return ax

def read_files(net):
	dir_out = 'joint/' + net
	fn = dir_out + "/iterations/"+net.split('_')[0]
	flowNormConstant=txt2list( fn+'_flowNormConstant.txt')
	flowNormGD=txt2list( fn+'_flowNormGD.txt')
	flowNormAlternating=txt2list( fn+'_flowNormAlternating.txt')
	flowNormJOINT=txt2list( fn+'_flowNormJOINT.txt')
	gNormConstant=txt2list( fn+'_gNormConstant.txt')
	gNormGD=txt2list( fn+'_gNormGD.txt')
	gNormAlternating=txt2list( fn+'_gNormAlternating.txt')
	gNormJOINT=txt2list( fn+'_gNormJOINT.txt')
	#fcoeffs = txt2list( dir_out +'/output/'+net+'_costFunct')
	return flowNormConstant, flowNormGD, flowNormAlternating, flowNormJOINT, gNormConstant, gNormGD, gNormAlternating, gNormJOINT


def plotVecs(ax, constant, gd, alternating, joint):
	x_axis  = [i for i in range(len(constant))]	
	lw=1
	#joint = [i*3 for i in joint]
	ax.plot(x_axis, constant, label='$f(\\cdot) =$ BPR', linestyle='--', linewidth=lw)
	ax.plot(x_axis, gd, label='GD', linestyle='--', linewidth=lw)
	ax.plot(x_axis, alternating, label='Alternating', linestyle='--', linewidth=lw)
	ax.plot(x_axis, joint, label='Joint', linestyle='-', linewidth=lw)

def plotVecsNo(ax, constant, gd, alternating, joint):
	lw=1
	#joint = [i*3 for i in joint]
	#print(len(constant[0]))
	#print(len(constant[1]))
	ax.plot(constant[0], constant[1], label='$f(\\cdot) =$ BPR', linestyle='--', linewidth=lw)
	ax.plot(gd[0], gd[1], label='GD', linestyle='--', linewidth=lw)
	ax.plot(alternating[0], alternating[1], label='Alternating', linestyle='--', linewidth=lw)
	ax.plot(joint[0], joint[1], label='Joint', linestyle='-', linewidth=lw)
	ax.set_ylabel("$F(\\mathbf{g}, \\boldsymbol{\\beta})$")
	ax.set_xlabel("Iteration, $j$")
	ax.set_xlim((0,30))
	ax.set_ylim(top=4e8)
	plt.tight_layout()

def plotFlowsGdiff(netname):
	# Read vectors
	flowNormConstant, flowNormGD, \
	flowNormAlternating, flowNormJOINT, \
	gNormConstant, gNormGD, \
	gNormAlternating, gNormJOINT = read_files(netname)
	
	# Plot Flow Norm
	fig, ax = plt.subplots(1,2, figsize=(4.75,2)) # sharex=True)
	plotVecs(ax[0], flowNormConstant, flowNormGD, flowNormAlternating, flowNormJOINT)
	ax[0].set_ylabel("$F(\\mathbf{g}, \\boldsymbol{\\beta})$")
	ax[0].set_xlabel("Iteration, $j$")
	#ax[0].set_ylim([-.05e7,1e7])
	ax[0].set_xlim([0,30])
	plt.tight_layout()

	# Plot gNorm
	plotVecs(ax[1], gNormConstant, gNormGD, gNormAlternating, gNormJOINT)
	ax[1].set_xlabel("Iteration, $j$")
	ax[1].set_ylabel("$||(\\mathbf{g} - \\mathbf{g}^{*})||$")
	ax[1].legend(frameon=True, framealpha=0.8)
	#ax[1].set_ylim([-.05e7,1e7])
	ax[1].set_xlim([0,30])
	plt.tight_layout()
	plt.savefig(netname+'_norms.pdf')


def plotMulitClass(netname='Braess2_final'):
	fig, ax = plt.subplots(3,2, figsize=(4.5,7.5))
	plt.savefig(netname+'multi.pdf')


def read_files_CaseStudy(net):
	dir_out = 'joint/' + net
	fn = dir_out + "/output/"+net.split('_')[0]
	cfC=txt2list( fn+'_costFunct_c.txt')
	cfGD=txt2list( fn+'_costFunct_gd.txt')
	cfA=txt2list( fn+'_costFunct_a.txt')
	cfJ=txt2list( fn+'_costFunct_j.txt')
	fn = dir_out + "/iterations/"
	colnames = ['x','y']
	data = pd.read_csv(fn+'c.csv', names=colnames).sort_values(by='x')
	objCx = data.x.to_list() 
	objCy = data.y.to_list()
	data = pd.read_csv(fn+'gd.csv', names=colnames).sort_values(by='x')
	objGDx = data.x.to_list()
	objGDy = data.y.to_list()
	data = pd.read_csv(fn+'a.csv', names=colnames).sort_values(by='x')
	objAx = data.x.to_list()
	objAy = data.y.to_list()
	data = pd.read_csv(fn+'j.csv', names=colnames).sort_values(by='x')
	objJx = data.x.to_list()
	objJy = data.y.to_list()

	#fcoeffs = txt2list( dir_out +'/output/'+net+'_costFunct')
	return cfC,cfGD,cfA, cfJ, objCx, objCy, objGDx, objGDy, objAx, objAy, objJx, objJy




def plotFunObjCaseStudy(netname='EMA'):
	cdC,cfGD,cfA, cfJ, objCx, objCy, objGDx, objGDy, objAx, objAy, objJx, objJy = read_files_CaseStudy(net)
	fcoeffs_list = [cdC,cfGD,cfA, cfJ]

	fig, ax = plt.subplots(1,2, figsize=(5.3,2))
	plot_f(fcoeffs_list, ax[0])
	plotVecsNo(ax[1], [objCx, objCy], [objGDx, objGDy], [objAx, objAy], [objJx, objJy] )
	plt.tight_layout()
	plt.savefig(netname+'_cfAndObj.pdf')

# Save results to files

#'EMA_2019-10-13_01-12-51_test_EMA'
#for net in ['Braess2_final']:# ['EMA_single_final', 'NYC_final']:#, 'Braess2_final', 'Braess2','Braess3' , 'NYC', 'EMA']	:#,'EMA','NYC']:
#	plotFlowsGdiff(net)

for net in ['NYC_case_final_v2']:#['EMA_case_final_v2']:#['NYC_case_final_v2']:
	plotFunObjCaseStudy(net)





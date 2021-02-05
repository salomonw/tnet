import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

plt.style.use(['science', 'ieee', 'high-vis'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
 #   "font.serif": ["Times"],  # specify font here
   # "font.size":6,			  # specify font size here
   # "axes.linewidth":0.4,
#	'xtick.major.width' : 0.4,
#	'xtick.minor.width' : 0.2,
#	'ytick.major.width' : 0.4,
#	'ytick.minor.width' : 0.2,
#	'lines.linewidth' : lwidth,
	'patch.linewidth' : 0.1,
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

def plot_f():
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


def plotFlowsGdiff(netname):
	# Read vectors
	flowNormConstant, flowNormGD, \
	flowNormAlternating, flowNormJOINT, \
	gNormConstant, gNormGD, \
	gNormAlternating, gNormJOINT = read_files(netname)
	
	# Plot Flow Norm
	fig, ax = plt.subplots(1,2, figsize=(4.5,2)) # sharex=True)
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

# Save results to files

#plotMulitClass()
#'EMA_2019-10-13_01-12-51_test_EMA'
for net in ['EMA_single_final', 'NYC_final']:#, 'Braess2_final', 'Braess2','Braess3' , 'NYC', 'EMA']	:#,'EMA','NYC']:
	plotFlowsGdiff(net)
	


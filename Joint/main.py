from utils import *
import tnet
import msa

netFile = "../networks/NYC_Uber_net.txt"
gFile = "../networks/NYC_Uber_trips.txt"

opt_method = "gd" # options "joint" and "gd"

# Build a ground truth network
fcoeffs_truth = [1,0,0,0,0.2,0]
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)
print(1)
tNet.solveMSA_julia()
#tNet.solveMSA()
G_data = tNet.G.copy()
g_data = tNet.g.copy()

# Now, use the data to try to estimate demands, to do so 
# let us perturb the truth demands. 
g_k = tnet.perturbDemandConstant(tNet.g, max_var=0.5)
fcoeff = [1,0,0,0,0.1,0]
print(1)
for i in range(200):
	# update network and MSA
	print(1)
	tNet.set_g(g_k)
	print(1)
	tNet.set_fcoeffs(fcoeff)
	print(1)
	tNet.solveMSA_julia()
	print(1)
	# estimate derivatives
	dxdb = tNet.get_dxdb(delta=0.5, divide=1e1)
	print(1)
	dxdg = msa.get_dxdg(tNet.G, tNet.g, k =1)
	print(1)
	# set = trust regions
	g_tr = 500/(i+1)
	beta_tr = 0.05/(i+1)
	

	if opt_method == "gd": 
	# gradient descent
		Delta_g, Delta_fcoeffs = tNet.get_gradient_jointBilevel(G_data, dxdb=dxdb, dxdg=dxdg)
		g_k = {k: max(min(max(v-Delta_g[k], v-g_tr), v+g_tr),0) for k, v in g_k.items()}
		fcoeff = [max(min(max(fcoeff[n] - Delta_fcoeffs[n], fcoeff[n]-beta_tr), fcoeff[n]+beta_tr),0) for n in range(tNet.nPoly)]
	else:
	# solve joint bilvel
		g_k, fcoeff = tNet.solve_jointBilevel(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=0, c=1, lambda_1=0)
		#g_k, fcoeff = tNet.solve_jointBilevel_julia(G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=0, c=1, lambda_1=10)
	
	fcoeff[0] = 1

	# gathering statistics
	flowNorm = tnet.flowDiff(G_data, tNet.G)
	gNorm = tnet.gDiff(g_data, tNet.g)

	#print(tnet.get_FlowDict(tNet.G))
	print(str(int(flowNorm)))
	#print(g_k)
	#print(str(int(flowNorm)) + " | " + str(g_k) + " | " + str(tNet.fcoeffs))
	#print(int(gNorm))


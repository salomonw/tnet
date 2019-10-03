import networkx as nx
from gurobipy import *
from joint.utils import *
import numpy as np
import joint.msa as msa
import trafficAssignment.assign as ta
from scipy.sparse import lil_matrix
from scipy.special import comb
from scipy.linalg import block_diag
from multiprocessing import Pool
import copy 
import os
import time
'''
tnet_dir = os.getcwd()[:-5]
cmd = 'push!(LOAD_PATH, "'+ tnet_dir+ '")'
from julia.api import Julia
jl = Julia(compiled_modules=False)
jl.eval(cmd)
from julia import TrafficAssign
'''
class tNet():
	def __init__(self, netFile=None, gFile=None, G=None, g=None, fcoeffs=[1,0,0,0,0.15,0]):
		"""
        Initialize a Traffic Network object. It requieres networkx and Gurobi libraries

        Parameters
        ----------
        self : a tNet object

        netFile: a network file in the format proposed by Bar Gera

        gFile: an OD demand file in the format proposed by Bar Gera

        Returns
        -------
        A tNet Object.

        """
		if G == None and netFile != None:
			G = readNetFile(netFile)
		elif G !=None and netFile == None:
			G = G

		node_id_map = {k : G.nodes[k]['node name'] for k in G.nodes()}
		if g == None and gFile != None:
			g = readODdemandFile(gFile, node_id_map)
		elif g != None and gFile == None:
			g = g

		self.netFileName = netFile
		self.gFileName = gFile
		self.nLinks = len(G.edges())
		self.nNodes = len(G.nodes())
		self.nOD = len(g)
		self.Zones = getZones(g)
		self.nZones = getNumZones(g)
		self.totalDemand = sum(g.values())
		self.g = g
		self.gGraph = buildDemandGraph(g)
		self.G = G
		self.node_id_map = node_id_map
		self.fcoeffs = fcoeffs
		self.nPoly = len(fcoeffs)
		self.TAP = self.build_TAP()
		self.incidence_matrix, self.link_id_dict = incidence_matrix(self.G)


	def build_TAP(self):
		"""
	    Build a traffic assignment object based on the traffic network 
	    Jurgen Hackl <hackl@ibi.baug.ethz.ch>

	    Parameters
	    ----------

		gdict: OD demands dict

	    Returns
	    -------
	    An nx object.

	    """
		assert (self.nZones>0), "Number of Zones is zero!!"
		assert (self.totalDemand>0), "Total demand is zero!!"
		assert (self.nNodes>0), "No nodes in graph!!"
		assert (self.nLinks>0), "No links in graph!!"
		TAP = msa.TrafficAssignment(self.G, self.gGraph, fcoeffs=self.fcoeffs, iterations=350)
		return TAP
		

	def solveMSA(self):
		"""
	    Solve the MSA flows for a traffic network using the MSA module by 
	    Jurgen Hackl <hackl@ibi.baug.ethz.ch>

	    Parameters
	    ----------

		gdict: OD demands dict

	    Returns
	    -------
	    An nx object.

	    """
		self.TAP.run(fcoeffs=self.fcoeffs, build_t0=False)
		self.G = self.TAP.graph


	def get_solveMSA_julia(self, fcoeffs=False):
		"""
	    Solve the MSA flows for a traffic network using the MSA module by 
	    __ .. __ in Julia

	    Parameters
	    ----------

		tnet object

	    Returns
	    -------
	    An updated nx object.
	    """	
		solveMSA_julia(self, fcoeffs=fcoeffs)


	def set_fcoeffs(self, fcoeff):
		"""
	    set function coefficients of the transportation netwrok object

	    Parameters
	    ----------

		fcoeffs: function coefficients of a polynomial function

	    Returns
	    -------
	    updates the fcoeffs attribute of the tnet object

	    """		
		self.fcoeffs = fcoeff
		#TAP = msa.TrafficAssignment(self.G, self.gGraph, fcoeffs=fcoeff)

	def set_g(self, g):
		"""
	    sets a demand dictionary and updates the the tNet and TAP objects

	    Parameters
	    ----------

		g: an OD demand dictionary

	    Returns
	    -------
	    updates the objects in the tnet where g takes place

	    """		
		self.nOD = len(g)
		self.Zones = getZones(g)
		self.nZones = getNumZones(g)
		self.totalDemand = sum(g.values())
		self.g = g
		self.gGraph = buildDemandGraph(g)
		self.TAP = self.build_TAP()

	@timeit
	def solve_jointBilevel(self, G_data, dxdb, dxdg, g_tr = 45, beta_tr = 0.002, scaling=0, c=10, lambda_1=300):
		"""
	    solve joint bilevel problem using the information of the network

	    Parameters
	    ----------

		self: a tNet object

	    Returns
	    -------
	    the resulting OD demand estimate (g) and the polynomial cost coefficients (beta)

	    """	
		g, fc = jointBilevel(self.G, self.g, self.fcoeffs, self.incidence_matrix, self.link_id_dict,\
	    		 G_data, dxdb, dxdg, g_tr = g_tr, beta_tr = beta_tr, scaling=scaling, c=c, lambda_1=lambda_1)
		return g, fc


	def solve_jointBilevel_julia(self, G_data, dxdb, dxdg, g_tr = 45, beta_tr = 0.002, scaling=0, c=10, lambda_1=300):
		"""
	    solve joint bilevel problem using the information of the network

	    Parameters
	    ----------

		self: a tNet object

	    Returns
	    -------
	    the resulting OD demand estimate (g) and the polynomial cost coefficients (beta)

	    """	
	    # DATA TO JULIA
		A, B, C, h, sort_ld, sort_od = get_jointBilevel_canonical(self.G, self.g, self.fcoeffs, self.incidence_matrix, self.link_id_dict, G_data)
		K = 1
		nPoly = len(self.fcoeffs)
		
		ld = self.link_id_dict
		flow_k = np.transpose(np.array([self.G.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
		data = np.transpose(np.array([G_data.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))

		dxdg_k = []
		dxdb_k = []
		for i in sort_ld:
			dxdg_k.append(np.array([dxdg[w, ld[i]] for w in sort_od]))
			dxdb_k.append(np.array([dxdb[j][ld[i]] for j in range(nPoly)]))
		g_k = np.array([self.g[od] for od in sort_od])
		beta_k = np.array(self.fcoeffs)
		nNodes = len(self.G.nodes())
		nLinks = len(self.G.edges())
		nOD = len(self.g)
		parameters = np.array([g_tr, beta_tr, scaling, c, lambda_1, nNodes, nLinks, nOD])
		self.__save_jBilevel_2_tmp(A,B,C,h, flow_k, data, dxdb_k, dxdg_k, g_k, beta_k, ld, parameters)

		# RUN OPTIMIZATION MODEL
		shell("julia jointBilevel.jl", printOut=True)
		
		# READ JULIA RESULT
		g_k_1 = np.load("tmp_jl/g_k_1.npz")
		beta_k_1 = np.load("tmp_jl/beta_k_1.npz")
		g_k = {sort_od[i]:g_k_1[i] for i in range(nOD)}
		beta_k = [beta_k_1[i] for i in range(nPoly)]

		return g_k, beta_k

	def __save_jBilevel_2_tmp(self, A,B,C,h, flow_k, data, dxdb, dxdg, g_k, beta_k, ld, parameters):
		mkdir_n("tmp_jl")
		np.savez("tmp_jl/A.npz", A)
		np.savez("tmp_jl/B.npz", B)
		np.savez("tmp_jl/C.npz", C)
		np.savez("tmp_jl/h.npz", h)
		np.savez("tmp_jl/flow_k.npz", flow_k)
		np.savez("tmp_jl/data.npz", data)
		np.savez("tmp_jl/dxdb.npz", dxdb)
		np.savez("tmp_jl/dxdg.npz", dxdg)
		np.savez("tmp_jl/gk.npz" , g_k)
		np.savez("tmp_jl/betak.npz", beta_k)
		np.savez("tmp_jl/par.npz", parameters)
		dict2json(ld, "tmp_jl/ld.json")


	@timeit
	def get_gradient_jointBilevel(self, G_data, dxdb=None, dxdg=None ):
		"""
	    build the gradient (OD demand and cost coefficients) with respect to the
	    norm of flows of the joint Bilevel problem, 

	    Parameters
	    ----------

		self: a tNet object
		dxdb: derivative of net flows with respect to the fcoeffs
		dxdg: derivative of net flows with respect to the OD demand
		G_data: nx object with the data flows 

	    Returns
	    -------
	    Delta_g: dictionary with the gradient of the OD pairs
	    Delta_fcoeffs: list with the gradient of the fcoeffs

	    """
		if dxdg == None:
			dxdb = get_dxdb(delta=delta, divide=divide)
			dxdg = msa.get_dxdg(tNet.G, tNet.g, k=1)
		return gradient_jointBilevel(self.G, self.g, self.fcoeffs, dxdb, dxdg, G_data, self.link_id_dict)

def incidence_matrix(G):
	"""
    build incidence matrix and column index dictionary. Note that since 
    node 0 don't exist, the node is the column number plus one

    Parameters
    ----------

	a nx element

    Returns
    -------
    A sparse matrix and a dictionary to match line number and link

    """
	nNodes = len(G.nodes())
	nLinks = len(G.edges())
	N = np.zeros((nNodes, nLinks))
	link_dict = {}
	idx = 0
	for s,t in G.edges():
		link_dict[idx] = (s,t)
		N[s-1, idx] = -1
		N[t-1, idx] = 1
		idx+=1
	return N, link_dict


def gradient_jointBilevel(G, g_k, fcoeffs, dxdb, dxdg, G_data, link_id_dict):
	"""
    build the gradient (OD demand and cost coefficients) with respect to the
    norm of flows of the joint Bilevel problem, 

    Parameters
    ----------

	G: a nx object with flow attribute 
	g_k: a demand dictionary
	fcoeffs: the parameters of the cost function polynomial
	dxdb: derivative of net flows with respect to the fcoeffs
	dxdg: derivative of net flows with respect to the OD demand
	G_data: nx object with the data flows 
	link_id_dict: dictionary with link ids

    Returns
    -------
    Delta_g: dictionary with the gradient of the OD pairs
    Delta_fcoeffs: list with the gradient of the fcoeffs

    """
	# sort dictionaries
	ld = link_id_dict # TODO: Check this is equal to the incidence matrix! !!
	sort_ld = sorted(ld)
	sort_od = sorted(g_k)
	nPoly = len(fcoeffs)

	flow_k = np.transpose(np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	data = np.transpose(np.array([G_data.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	# create objective function
	DeltaF = np.array([[0 for i in range(nPoly)], 0])
	for i in sort_ld:
		if dxdb:
			dxdg_ = np.array([dxdg[w, ld[i]] for w in sort_od])
			dxdb_ = np.array([dxdb[j][ld[i]] for j in range(nPoly)])
			DeltaF = np.add(DeltaF, np.array([2*(flow_k[i] - data[i]) * dxdb_ , 2*(flow_k[i] - data[i]) * dxdg_]))
		else:
			dxdg_ = np.array([dxdg[w, ld[i]] for w in sort_od])
			DeltaF = np.add(DeltaF, np.array([[0 for i in range(nPoly)] , 2*(flow_k[i] - data[i]) * dxdg_]));

	Delta_g = {sort_od[i]: DeltaF[1][i]  for i in range(len(DeltaF[1]))}
	Delta_fcoeffs = DeltaF[0]		
	return Delta_g, Delta_fcoeffs



def get_jointBilevel_canonical(G, g_k, fcoeffs, N, link_id_dict, G_data):
	"""
    b

    Parameters
    ----------

	N : node-link incidence matrix, sparse ...


    Returns
    -------
    A

    """
	K = 1
	nOD = len(g_k.keys())
	nNodes = len(G.nodes())
	nLinks = len(G.edges())
	nPoly = len(fcoeffs)
	beta_k = fcoeffs

	ld = link_id_dict
	sort_ld = sorted(ld)
	sort_od = sorted(g_k)


	# ---- build first constraint ----
	NT = N.T
	A1  = np.array([NT[link,:] for link in sort_ld])
	A1 = block_diag( *[A1]*nOD).T

	B1 = [-G.get_edge_data(ld[idx][0],ld[idx][1])['t_k'] * \
			np.transpose([np.power(G.get_edge_data(ld[idx][0],ld[idx][1])['flow']/ \
			G.get_edge_data(ld[idx][0],ld[idx][1])['capacity'], j) for j in range(nPoly)]) for idx in sort_ld]
	B1 = np.tile(np.array(B1).T, nOD).T
	C1 = np.zeros((1, np.shape(B1)[0]))
	h1 = np.zeros((1, np.shape(B1)[0]))


	# ---- build second constraint ----
	A2 = []
	for (s,t) in sort_od:
		d_w = np.zeros((nNodes,1))
		d_w[s-1,0] = g_k[(s,t)]
		d_w[t-1,0] = -g_k[(s,t)]
		A2.extend(d_w)
	A2 = np.array(A2)

	flow_k = np.transpose(np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	data = np.transpose(np.array([G_data.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	fF_vec = np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['t_k'] for idx in sort_ld])
	a_flow = np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld])
	a_capacity = np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['capacity'] for idx in sort_ld])
	B2 = np.multiply(np.dot(flow_k, fF_vec) , np.array([np.power(sum(np.divide(a_flow,a_capacity)),j) for j in range(nPoly)])).reshape((K,nPoly))
	C2 = -np.ones((1, K))
	h2 = np.zeros((1, K))


	# ---- third constraint ------
	B3 = np.zeros((0, nPoly))
	lspace = np.linspace(-.1,2,30)
	B3 = []
	for i in range(len(lspace)-1):
		B3.append(np.array([lspace[i]**j - lspace[i+1]**j  for j in range(nPoly)] ))
	B3 = np.array(B3)
	A3 = np.zeros((nOD*nNodes, len(lspace)-1))
	C3 = np.zeros((1,  len(lspace)-1))
	h3 = np.zeros((1,  len(lspace)-1))

	# ---- fourth constraint ----
	A4 = np.zeros((nOD*nNodes, K))
	B4 = np.zeros((K,nPoly))
	C4 = -np.eye(K)
	h4 = np.zeros((K,1))


	A = np.concatenate((A1.T, A2.T, A3.T, A4.T), axis=0)
	B = np.concatenate((B1, B2, B3, B4), axis=0)
	C = np.concatenate((C1.T, C2.T, C3.T, C4.T), axis=0)
	h = np.asmatrix(np.concatenate((h1.T, h2.T, h3.T, h4.T), axis=0))
	return A, B, C, h, sort_ld, sort_od

def jointBilevel(G, g_k, fcoeffs, N, link_id_dict, G_data, dxdb, dxdg, g_tr = 45, beta_tr = 0.002, scaling=0, c=10, lambda_1=300):
	"""
    b

    Parameters
    ----------

	N : node-link incidence matrix, sparse ...


    Returns
    -------
    A

    """

	A,B,C,h, sort_ld, sort_od = get_jointBilevel_canonical(G, g_k, fcoeffs, N, link_id_dict, G_data)
	K = 1
	nOD = len(g_k.keys())
	nNodes = len(G.nodes())
	nLinks = len(G.edges())
	nPoly = len(fcoeffs)
	beta_k = fcoeffs
	ld = link_id_dict

	flow_k = np.transpose(np.array([G.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	data = np.transpose(np.array([G_data.get_edge_data(ld[idx][0],ld[idx][1])['flow'] for idx in sort_ld]))
	
	#TODO: convert to sparse matrices in order to make the computation faster
	# ------ Build guroby model -----

	# TODO: add quicksums in order to make computation faster
	m = Model("jointBilevel")
	m.setParam( 'OutputFlag', False )
	# define variables
	y = [m.addVar(name="y_{"+str((i+1))+","+str(j+1)+"}") for i in range(nNodes) for j in range(nOD)]
	g = [m.addVar(lb=0, name="g_{"+str(i) + "}") for i in range(nOD)]   
	eps = [m.addVar(lb=0, name="eps_{"+str(i) + "}") for i in range(K)]
	beta = [m.addVar(lb=0, name="beta_{"+str(i) + "}") for i in range(nPoly)]  
	v = [m.addVar(lb=0, name="v{"+str(i) + "}") for i in range(np.shape(A)[0])] 
	ksi = m.addVar(lb=0,  name="ksi")

	# add constraints
	# first 
	#print(time.time())
	for i in range(len(A)):
		constr = LinExpr(LinExpr(np.dot(A[i,:],y)) +  LinExpr(np.dot(B[i,:],beta)) + LinExpr(np.dot(scaling*C[i,:],eps)) + h[i,:])
		m.addConstr(constr <= 0)

	# second 
	for i in range(len(A.T)):
		m.addConstr(LinExpr(np.dot(A[:,i], v))==0)
	
	#print(time.time())
	# third (primal-dual gap)
	# Creating H1 and H2

	H1 = np.eye(K)
	H2 = np.diag(np.divide(1,setUpFitting(nPoly,c)))

	iH1 = np.linalg.inv(H1)
	iH2 = np.linalg.inv(H2)
	#print(time.time())
	M1 = np.multiply((1/4), np.outer(C, np.outer(iH1,C.T)))
	M1 = np.divide((M1 + M1.T),2)
	M2 = np.multiply((1/4), np.dot(B, np.dot(iH2,B.T)))
	M2 = np.divide((M2 + M2.T),2) + np.multiply(1e-2, np.eye(len(M2)))
	#print(time.time())
	nlconstr = 0
	nlconstr += QuadExpr(H1.dot(eps).dot(eps))
	#print(time.time())  
	nlconstr += QuadExpr(H2.dot(beta).dot(beta))   
	print(time.time())
	'''
	for j in range(len(v)):
	for i in range(len(v)):
		nlconstr +=  QuadExpr(v[j]*v[i]*M1[i,j])
		nlconstr += QuadExpr(v[j]*v[i]*M2[i,j])
	'''
	nlconstr += QuadExpr(M1.dot(v).dot(v) )
	

	print(time.time())
	nlconstr += QuadExpr(M2.dot(v).dot(v) )
	#print(time.time())
	for i in range(len(h)):
		nlconstr -= LinExpr(h[i]*v[i])
	#print(time.time())
	m.addConstr(nlconstr<=ksi)
	#print(time.time())
	# fourth constraint (trust regions)
	idx = 0 
	od_idx = {}
	od_idx_ = {}
	for i in sort_od:	
		od_idx[idx] = i
		od_idx_[i] = idx
		m.addConstr(g[idx] >= g_k[i] - g_tr)
		m.addConstr(g[idx] <= g_k[i] + g_tr)
		idx +=1
	#print(time.time())
	for i in range(nPoly):
		m.addConstr(beta[i] >= beta_k[i] - beta_tr)
		m.addConstr(beta[i] <= beta_k[i] + beta_tr)
	#print(time.time())
	m.addConstr(beta[0]==1)
	#print(3)
	# create objective function
	DeltaF = np.array([0, 0 , 0])
	for i in sort_ld:
		dxdg_ = np.array([dxdg[w, ld[i]] for w in sort_od])
		dxdb_ = np.array([dxdb[j][ld[i]] for j in range(nPoly)])
		DeltaF = np.add(DeltaF, np.array([2*(flow_k[i] - data[i]) * -dxdb_ , 2*(flow_k[i] - data[i]) * -dxdg_ , lambda_1]));
	
	obj = 0
	for b in range(len(beta)):
		obj += LinExpr(np.multiply(DeltaF[0][b] , LinExpr(np.subtract(beta_k[b], beta[b])) )) 
	for w in range(len(g)):
		obj += LinExpr(np.multiply(DeltaF[1][w] , LinExpr(np.subtract(g_k[od_idx[w]], g[w]) ))) 
	obj += LinExpr(np.multiply(DeltaF[2] , ksi )) 

	#print(time.time())
	#m.addConstr(ksi/1e10<=1e6)
	m.setObjective(obj, GRB.MINIMIZE)
	#print(4)
	m.optimize()
	#m.write("file.lp")
	# Retrive values of variables
	g_sol = {}
	for od in sort_od:
		g_sol[od] = g[od_idx_[od]].X

	beta_sol = []
	for i in range(nPoly):
		beta_sol.append(beta[i].X)

	return g_sol, beta_sol




def QuadMult(x,A):
	"""
    compute quadratic expresion x'Ax

    Parameters
    ----------
	
	x: list or numpy vector
	A: numpy Matrix

    Returns
    -------
    value

    """	
	return (np.dot(x.T ,np.dot(A, x)))[0,0]

def setUpFitting(deg, c):
	"""
    find the parameter for the kernel function 

    Parameters
    ----------
	
	deg: polynomial degree
	c: parameter tuned using cross-validation

    Returns
    -------
    norm Coefficientes

    """	
	normCoeffs = []
	for i in range(deg):
		normCoeffs.append(comb(deg, i) * c**(deg-i+2))
	return normCoeffs

def perturbDemandConstant(g, max_var):
	"""
    Perturb demand by a random constant value 

    Parameters
    ----------
	
	g: demand dict
	max_var: maximum percentage of the demands which is goinf to variate either
			increase or decrease

    Returns
    -------
    a perturbed demand dict

    """
	for od,demand in g.items():
		g[od] = 1#max(0, demand+ demand*max_var*np.random.uniform(-1,1))
	return g

def normFlowDiff(G_data, G_estimate):
	"""
    estimate the difference between the flows of two networks
    with identical network topology. 

    Parameters
    ----------

	G_data: nx network with edge attribute "flow"
	G_estimate: nx network with edge attribute "flow"

    Returns
    -------
    a scalar representing the norm of the flows of two netorks

    """
	diff = sum([(G_data.get_edge_data(s,t)['flow'] - G_estimate.get_edge_data(s,t)['flow'])**2 for s,t in G_data.edges()])
	return diff


def get_FlowDict(G):
	"""
    return the flow of a nx as a dictionary

    Parameters
    ----------
	G: nx network with edge attribute "flow"

    Returns
    -------
    a dict with keys=link and values=flow

    """
	return {(s,t): G.get_edge_data(s,t)['flow'] for s,t in G.edges()}



def gDiff(g_data, g_estimate):
	"""
    estimate the difference between the OD demands with identical
    network topology. 

    Parameters
    ----------

	g_data: a dictionary containing OD Demandas
	g_estimate: a dictionary containing OD Demandas

    Returns
    -------
    a scalar representing the norm of the demands

    """
	diff = sum([(g_data[key]-g_estimate[key])**2 for key in g_data.keys()])
	return diff



def buildDemandGraph(g):
	"""
    a nx graph defininf OD demands

    Parameters
    ----------

	gdict: OD demands dict

    Returns
    -------
    An nx object.

    """
	od_graph = nx.DiGraph()
	for (s,t), d in g.items():
		od_graph.add_edge(s, t, demand = d)
	return od_graph

def readNetFile(netFile, sep="\t"):
	"""
    Read the netFile and convert it to a nx object

    Parameters
    ----------

    netFile: a network file in the format proposed by Bar Gera

    Returns
    -------
    An nx object.

    """
    
    # Create a networkx obj
	G = nx.DiGraph()
    # Read the network file
	with open(netFile) as file_flow:
		file_flow_lines = file_flow.readlines()

	for line in file_flow_lines:
		if ";" in line and "~" not in line:
			links = line.split(sep)
			G.add_edge(int(links[1]), int(links[2]), capacity=float(links[3]), \
				length=float(links[4]), t_0=float(links[5]), \
				B=float(links[6]), power=float(links[7]), speedLimit=float(links[8]), \
				toll=float(links[9]), type=float(links[10]))
	G = nx.convert_node_labels_to_integers(G, first_label=1, ordering='sorted', label_attribute='node name')

	return G


def readODdemandFile(gfile, node_id_map):
	"""
	Read the gfile and convert it to a dict

	Parameters
	----------

	gFile: a demand file in the format proposed by Bar Gera

	Returns
	-------
	A dict with OD as key and demand as value

	"""

	with open(gfile) as trips:
		trip_lines = trips.readlines()

	od_d = {}
	od_d_t = {}
	for line in trip_lines:
		if "Origin" in line:
			origin = node_id_map[int(line.split("gin")[1])]
		if ";" in line:
			line_ = line.split(";")
			for j in line_:
				if ":" in j:
					dest = node_id_map[int(j.split(":")[0])]
					d = float(j.split(":")[1])
					if origin != dest:
						od_d[(origin, dest)] = d
	return od_d


@timeit
def writeNetfile(G, g, fname):
	"""
	write net file from G (networkx)

	Parameters
	----------

	G: a neteworkx object

	Returns
	-------
	file
	"""
	nZones = str(len(set([i for i,j in g.keys()])))
	nNodes = str(len(G.nodes()))
	nLinks = str(len(G.edges()))
	header = "<NUMBER OF ZONES> "+nZones+"\n<NUMBER OF NODES> "+nNodes+"\n<FIRST THRU NODE> 1\n<NUMBER OF LINKS> "+nLinks+"\n<END OF METADATA>\n~  Init node  Term node  Capacity  Length  Free Flow Time  B  Power  Speed limit  Toll  Type  ;\n"
	text = ""
	idx = 0
	link_id = {}
	for (s,t) in G.edges():
		idx += 1
		link_id[idx] = (s,t)
		text += "\t"+str(s)+"\t"+str(t)+"\t"+str(G[s][t]["capacity"])+"\t"+str(G[s][t]["length"])+"\t"+str(G[s][t]["t_0"]) \
		+"\t"+str(G[s][t]["B"])+"\t"+str(G[s][t]["power"])+"\t"+str(G[s][t]["speedLimit"])+"\t"+str(G[s][t]["toll"]) \
		+"\t"+str(G[s][t]["type"])+"\t;\n"
	write_file(header+text, fname)
	return link_id, header+text


@timeit
def writeTripsfile(g, fname):
	"""
	write trips file from dict 

	Parameters
	----------
]
	g dict

	Returns
	-------
	file
	"""
	nZones = str(len(set([i for i,j in g.keys()])))
	totalFlow = sum([d for d in g.values()])
	header = "<NUMBER OF ZONES> "+nZones+"\n<TOTAL OD FLOW> "+str(totalFlow)+"\n<END OF METADATA>\n\n"

	text = ""
	nodes = list(set([s for s,t in g.keys()]))
	nodes.extend(list(set([t for s,t in g.keys()])))
	nodes = list(set(nodes))
	for o in nodes:
		txt = ""
		txt = "Origin " + str(o) + "\n"
		demandTxt = ""
		for d in nodes:
			try:
				gk = str(g[(o,d)])
			except:
				gk = str(0)
			demandTxt += str(d) + "\t : \t" + str(gk) + ";\n"
		text += txt + demandTxt + "\n\n"
	write_file(header+text, fname)
	return header+text


def getZones(gdict):
	"""
    Returns Zones in a OD file

    Parameters
    ----------

    gdict: a demand dictiornary

    Returns
    -------
    a list with the Zones of the network

    """
	sources = [s for (s,t) in gdict.keys()]
	targets = [t for (s,t) in gdict.keys()]
	sources.extend(targets)
	return set(sources)

def getNumZones(gdict):
	"""
    Finds the number of zones in a network

    Parameters
    ----------

    gdict: a demand dictiornary

    Returns
    -------
    number of zones

    """
	return len(getZones(gdict))

def flow_conservation_adjustment(G):
	"""
    Returns the same graph with conserved flows minimizng the quadratic error between 
    the original flows and the solution.

    Parameters
    ----------

    G: a networkx file with flows

    Returns
    -------
    G: a networx file with conserved flows

    """
	y = {key : G.get_edge_data(key[0], key[1])[G.get_edge_data(key[0], key[1]).keys()[0]]['flow'] for key in list(G.edges())}
	y_0 = y.values()

	## Gurobi minimization problem
	model = Model("Flow_conservation_adjustment")
	l = len(y)
	x = []
	# Define variables (adjusted flows)
	for i in range(l):
		x.append(model.addVar(name = str(y.keys()[i])))
	model.update()
	# Set objective ||x-y||2
	obj = 0
	for i in range(l):
		obj += (x[i] - y_0[i] ) * (x[i] - y_0[i])
	model.setObjective(obj)
	# Set constraints
	# non-negativity
	for i in range(l):
		model.addConstr( x[i] >= 0 )
	# conservation of flow
	for node in G.nodes():
		in_ = list(G.in_edges(nbunch = node,data=False))
		out_ = list(G.out_edges(nbunch = node,data=False))
		if len(in_)>0 and len(out_)>0:
			model.addConstr(quicksum(model.getVarByName(str(incoming_edge)) for incoming_edge in in_) == \
				quicksum(model.getVarByName(str(outgoing_edge)) for outgoing_edge in out_))
	model.update()
	model.setParam('OutputFlag', False)
	model.optimize()
    ##

	u = []
	res = {}
	for v in model.getVars():
		res[v.VarName] = v.x
		s = int((v.VarName.split("(")[1].split(")")[0].split(","))[0])
		t = int((v.VarName.split("(")[1].split(")")[0].split(","))[1])
		G[s][t][G.get_edge_data(s, t).keys()[0]]['flow_conserved'] = v.x
	return G

def greenshieldFlow(speed, capacity, free_flow_speed):
	"""
    Returns the flow of a link following the Fundamental Diagram (Greenshield's Model).

    Parameters
    ----------

    speed: miles/hr or km/hr
    capacity: link's capacity in veh/hr
    free_flow_speed: link's free flow speed in miles/hr or km/hr

    Returns
    -------
    flow: resulting flow 

    """
	if speed > free_flow_speed or capacity < 0:
		return 0
	x = 4 * capacity * speed / free_flow_speed - 4 * capacity * (speed ** 2) / (free_flow_speed ** 2)
	return x



@timeit
def solveMSA_julia(tnet, fcoeffs=False, net_fname="tmp_jl/net.txt", trips_fname="tmp_jl/trips.txt"):
	"""
    Solve the MSA flows for a traffic network using the MSA module by 
    __ .. __ in Julia

    Parameters
    ----------

	tnet object

    Returns
    -------
    An updated nx object.
    """	
	if fcoeffs==False:
		fcoeffs = tnet.fcoeffs
	#pwd = os.getcwd()
	#link_id, text_net = writeNetfile(tnet.G, tnet.g, net_fname)
	#text_trips = writeTripsfile(tnet.g, trips_fname)
	new_G = ta.assignment(tnet.G, tnet.g, tnet.fcoeffs, flow=False, method='MSA', accuracy=0.0001, max_iter=1000)
	tnet.G = new_G
	#flow, t_k = TrafficAssign.solve_TAP("tmp", net_fname, trips_fname, fcoeffs)
	'''
	idx = 0
	for (s,t) in link_id.values():
		tnet.G[s][t]["flow"] = flow[idx]
		tnet.G[s][t]["t_k"] = t_k[idx]
		idx +=1
	'''
	return tnet

@timeit
def get_dxdb(tnet_, delta=0.05, divide=1, num_cores=False, **kwargs):
	"""
    Get the derviatives of the link flows wiht resepect to the 
    cost coefficient parameters.

    Parameters
    ----------

	gdict: OD demands dict

    Returns
    -------
    An nx object.

    """
	tnet_copy = copy.deepcopy(tnet_)
	if num_cores == False:
		num_cores = len(tnet_.fcoeffs)
	fcoeffs_vec = []
	for i in range(len(tnet_copy.fcoeffs)):
		fcoeffs_i = tnet_copy.fcoeffs.copy()
		fcoeffs_i[i] = fcoeffs_i[i] + delta
		fcoeffs_vec.append(fcoeffs_i)
	dxdb = []
	''
	for i in range(len(tnet_copy.fcoeffs)):
		dxdb.append(get_get_dxdb_single(tnet_copy, fcoeffs_vec[i], delta=delta, divide=divide))
	
	'''
	pool = Pool(num_cores)
	dxdb = [pool.apply(get_get_dxdb_single, (tnet_, coeff, delta, \
		divide, "tmp_jl/"+str(fcoeffs_vec.index(coeff))+ \
		"net.txt", "tmp_jl/"+str(fcoeffs_vec.index(coeff))+"trips.txt" ) ) for coeff in fcoeffs_vec]
	pool.close()
	'''
	return dxdb



def get_get_dxdb_single(tNet, fcoeffs, delta=0.05, divide=1, net_fname="tmp_jl/net.txt", trips_fname="tmp_jl/trips.txt"):
	"""
	Derivative of the flow distribution with respect to a single beta coefficient
	in the cost function. Uses a finite-difference approach

	parameters
	----------

	fcoeffs: the actual coefficient vector
	delta: how big/small is the step

	Returns
	-------
	a dictionary (key=edge, value=derivative)
	"""

	G_orig = tNet.G.copy()
	tNet.set_fcoeffs(fcoeffs)
	solveMSA_julia(tNet, fcoeffs=tNet.fcoeffs, net_fname=net_fname, trips_fname=trips_fname)
	G_new = tNet.G.copy()
	#print([tNet.G[s][t]["flow"] for (s,t) in tNet.G.edges()])
	flowDiff_ = flowDiff(G_new, G_orig)
	dxdb_i = {k : v/delta/divide for k,v in flowDiff_.items()}
	return dxdb_i 



def flowDiff(graph1, graph2):
    """
    Return the difference between the flows of two graphs.

    Parameters
    ----------

    graph1: netwrorkx obj
    graph2: networkx obj

    Returns
    -------
    a dictionary with name od edge and diff in flows. graph1-graph2

    """
    flowDiff_ = {}
    for edge in graph1.edges():
        flow1 = graph1[edge[0]][edge[1]]['flow']
        flow2 = graph2[edge[0]][edge[1]]['flow']
        flowDiff_[edge] = flow1-flow2
    return flowDiff_


def get_totalTravelTime(G):
    """
    Return the total travel time of a network

    Parameters
    ----------

    G: networkx obj

    Returns
    -------
    a float 

    """
    return sum([G[u][v]["t_k"]*G[u][v]["flow"] for (u,v) in G.edges()])

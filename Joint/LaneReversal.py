
from gurobipy import *
import tnet
import networkx as nx
import numpy as np
import operator


def set_numLanes(G, carLength=4.572):
	'''
	set the number of lanes of a link based on it's capacity and lenght

	Parameters:
	____________

	G: a nx object with ____
	dict: a dictionary with link 'a' as key and link 'a_' as value
	
	Returns
	____________
	
	G: a netwrokx object with attribute lanes
	'''
	for (u,v) in G.edges():
		G[u][v]['lanes'] = 4#round(G[u][v]["capacity"]*2/((G[u][v]["length"]*1600/carLength)))
		G[u][v]['capacityPerLane'] = G[u][v]['capacity']/G[u][v]['lanes']
	return G

def add_oppositeLinks(G):
	'''
	set opposite Links to the network 

	Parameters
	__________

	G: a nx object with ____
	dict: a dictionary with link 'a' as key and link 'a_' as value

	'''
	idx=0
	edges = list(G.edges())
	for (u,v) in edges:
		if (v,u) in edges:
			a=1
		else:
			G.add_edge(v,u, capacity=1, lanes=0.01, t_0=G[u][v]['t_0'], \
				length=G[u][v]['length'],  B=G[u][v]['B'], power=G[u][v]['power'],\
				speedLimit=G[u][v]['speedLimit'], toll=G[u][v]['toll'], \
				type=G[u][v]['type'], capacityPerLane=G[u][v]['capacityPerLane'] )
		idx+=1
	return G

def update_capcity(G):
	a=1

def LaneReversal_gurobi(G, g, fcoeffs,  carLength=4.572, method="branch_and_bound"):
	N, link_dict = tnet.incidence_matrix(G)
	sort_ld = sorted(link_dict)
	sort_od = sorted(g)
	nLinks = len(G.edges())
	nNodes = len(G.nodes())
	nOD = len(g)
	nPoly = len(fcoeffs)

	m = Model("LaneReversal")
	m.setParam( 'OutputFlag', False )
	
	x_t = [m.addVar(lb=0, name="x_{"+str(a)+"}") for a in sort_ld]
	x = []
	d = []
	g_id = {}
	w_id = 0
	for (s,t) in sort_od:
		# define flow variables
		x.append([m.addVar(lb=0, name="x_{"+str(a)+","+str(w_id)+"}") for a in sort_ld])
		# set demands
		d.append(np.zeros((nNodes,1)))
		d[w_id][s-1] = g[(s,t)]
		d[w_id][t-1] = -g[(s,t)]
		# add flow conservation and demand comply
		m.update()
		constr = np.dot(N,x[w_id])
		for n in range(nNodes):
			m.addConstr(constr[n] == d[w_id][n][0])
		g_id[w_id] = (s,t)
		w_id +=1
		
	m.update()
	# second constrains (sum of od flows is overall flow)
	for a in range(nLinks):
		m.addConstr(x_t[a] == sum([x[w][a] for w in g_id.keys()]))
	m.update()

	# define lanes variables
	l = {}
	for a in sort_ld:
		u = link_dict[a][0]
		v = link_dict[a][1]
		l[u,v] = m.addVar(vtype=GRB.INTEGER, lb=0, name="l_{"+str(u)+","+str(v)+"}")
		l[v,u] = m.addVar(vtype=GRB.INTEGER, lb=0, name="l_{"+str(v)+","+str(u)+"}")
		m.addConstr(l[u,v]+l[v,u] <= G[u][v]["lanes"]+G[v][u]["lanes"])


	obj = 0
	for a in sort_ld:
		u = link_dict[a][0]
		v = link_dict[a][1]		
		obj += x_t[a] * G[u][v]['t_0'] + (sum([fcoeffs[i]*(x_t[a]^i/(l[u,v]*G[u][v]["length"]*carLength/2)^i)for i in range(nPoly)]))


def choose_flipping_lane(G, t_diff):
	chosen = False
	while chosen == False:
		(u,v) = max(t_diff.items(), key=operator.itemgetter(1))[0]
		if G[u][v]['t_k']/G[u][v]['t_0'] >=  G[v][u]['t_k']/G[v][u]['t_0']:
			if G[v][u]['lanes']<=0.01 :
				del t_diff[(u,v)]
				del t_diff[(v,u)]
				continue	
			else:
				chosen =(u,v)
		else:
			if G[u][v]['lanes']<=0.01 :
				del t_diff[(v,u)]
				del t_diff[(u,v)]
				continue	
			else:
				chosen = (v,u)
	return chosen


def flip_lane(G, l):
	(u,v) = l
	G[u][v]['lanes'] = round(G[u][v]['lanes']+1,0)
	G[v][u]['lanes'] = round(G[v][u]['lanes']-1,0)
	if G[v][u]['lanes'] == 0:
		G[v][u]['lanes'] = 0.01
	if G[v][u]['lanes'] == 0:
		G[v][u]['lanes'] = 0.01
	
	G[u][v]['capacity'] = G[u][v]['lanes']*G[u][v]['capacityPerLane']
	G[v][u]['capacity'] = G[v][u]['lanes']*G[u][v]['capacityPerLane']
	return G

def LaneReversal_greedy(G, g, fcoeffs, carLength=4.572):
	net = tnet.tNet(G=G, g=g, fcoeffs=fcoeff)
	net = tnet.solveMSA_julia(net)
	G = net.G

	obj = tnet.get_totalTravelTime(G)

	print("Initial sol: " + str(obj))
	print("------------------------------------------------------")
	print("|  n   |    obj    |  a+  |  a-  |  l(a+)  |  l(a-)  |")
	print("------------------------------------------------------")
	for i in range(40):
		t_diff = {(u,v):(G[u][v]['t_k']/G[u][v]['t_0'] - G[v][u]['t_k']/G[v][u]['t_0'])**2 for (u,v) in G.edges()}
		(u,v) = choose_flipping_lane(G, t_diff)
		G = flip_lane(G, (u,v))
		net = tnet.tNet(G=G, g=g, fcoeffs=fcoeff)
		net = tnet.solveMSA_julia(net)
		G = net.G
		obj = tnet.get_totalTravelTime(G)
		
		print("|  {n:.0f}   |   {obj:.0f}   |  {a}  |  {a_}  |   {la:.0f}   |   {la_:.0f}    |".format(\
		n=i, obj=obj,  a=str((u,v)), a_=str((v,u)), la=G[u][v]['lanes'], la_=G[v][u]['lanes']))
	
	print(tnet.get_totalTravelTime(G))
		#print(G[v][u])


def solve_LaneReversal(G, g, fcoeffs,  method="branch_and_bound"):
	G = set_numLanes(G)
	G = add_oppositeLinks(G)
	LaneReversal_greedy(G, g, fcoeffs)



if __name__ == "__main__":
	netFile = "../networks/EMA_net.txt"
	gFile = "../networks/EMA_trips.txt"
	fcoeff = [1,0,0,0,0.15,0]
	tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeff)
	g_k = tnet.perturbDemandConstant(tNet.g, max_var=0)
	solve_LaneReversal(tNet.G, g_k, tNet.fcoeffs)

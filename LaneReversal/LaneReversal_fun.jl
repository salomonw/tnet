

function LaneReversal(N, link_dict, g, fcoeffs,  carLength=4.572, method="branch_and_bound")

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

end

using NPZ
using JuMP
using Gurobi
using LinearAlgebra

function facto(m::Int64)
    factori = 1
    for j = 1:m
        factori = factori * j
    end
    return factori
end

function comb(n::Int64, m::Int64)
    combi = facto(n) / (facto(m) * facto(n-m))
    return convert(Int, combi)
end

function setUpFitting(deg::Int64, c::Float64)
	normCoeffs = Array{Int}(undef, deg+1)
	for i=1:deg + 1
		normCoeffs[i] = comb(deg, i-1) * c^(deg-i+1)
	end
	return normCoeffs
end


function read_np_data()
	A = npzread("tmp_jl/A.npz")["arr_0"]
	B = npzread("tmp_jl/B.npz")["arr_0"]
	C = npzread("tmp_jl/C.npz")["arr_0"]
	h = npzread("tmp_jl/h.npz")["arr_0"]
	flow_k = npzread("tmp_jl/flow_k.npz")["arr_0"]
	data = npzread("tmp_jl/data.npz")["arr_0"]
	dxdb_k = npzread("tmp_jl/dxdb.npz")["arr_0"]
	dxdg_k = npzread("tmp_jl/dxdg.npz")["arr_0"]
	g_k = npzread("tmp_jl/gk.npz")["arr_0"]
	beta_k = npzread("tmp_jl/betak.npz")["arr_0"]
	pars = npzread("tmp_jl/par.npz")["arr_0"]
	return A, B, C, h, flow_k, data, dxdg_k, dxdb_k, g_k, beta_k, pars
end

function jointBilevel()
	
	# Read optimization data
	A, B, C, h, flow_k, data, dxdg_k, dxdb_k, g_k, beta_k, pars = read_np_data()
	trust_region_g = pars[1,1]
	trust_region_beta = pars[2,1]
	scaling = pars[3,1]
	c = pars[4,1]
	lambda_1 = pars[5,1]
	numNodes = Int(pars[6,1])
	numLinks = Int(pars[7,1])
	numOD = Int(pars[8,1])
	numPoly = size(B,2)
	K = 1

	# create variables
	m  =  Model(with_optimizer(Gurobi.Optimizer, Presolve=0, OutputFlag=0))
	@variable(m, y[1:numOD,1:numNodes])
	@variable(m, g[1:numOD]>=0)
	@variable(m, epsilon[1:K]>=0)
	@variable(m, beta[1:numPoly])# >=0 )

	y_ = [y[i,j] for i = 1:numOD for j = 1:numNodes]

	# build IP-1 constraint
	constr = A*y_ + B*beta + C*epsilon.*scaling + h
	@constraint(m, constr .<= 0);
	@constraint(m, beta[1] == 1);

	# Dual variables constraint
	@variable(m, v[1:size(A,1)]>=0)
	@variable(m, ksi>=0)
	@constraint(m, A'*v .==0);

	# Primal Dual gap constraint
	# Creating H1 and H2
	H1 = Matrix(I, K, K);
	H2 = Matrix(Diagonal((1 ./setUpFitting(numPoly-1, c))));

	iH1 =  inv(H1)

	iH2 = inv(H2)

	M1 = (1/4)*C*iH1*C'  ;
	M1 = (M1 + M1')/2;
	M2 =  (1/4)*B*iH2*B'  ;
	M2 = (M2 + M2')/2 +1e-3*Matrix(I, size(M2,1), size(M2,1));

	qd = dot(beta, H2*beta)+ dot(v, M1*v) + dot(v, M2*v)
	ln = dot(-h,v)
	nlconstr = qd+ln
	@constraint(m, nlconstr <= ksi)

	# Creating Objective function
	DeltaF = zeros(numOD+numPoly+length(lambda_1) , 1);
	for i = 1 :numLinks
		DeltaF += [2*(flow_k[i] - data[i]).*dxdb_k[i,:] ; 
					2*(flow_k[i] - data[i]).*dxdg_k[i,:] ;
					lambda_1];
	end

	# Constraints on trust region
	k=0
	for g_ka in g_k
	 	k=k+1
		@constraint(m, g[k] <= g_ka + trust_region_g)
		@constraint(m, g[k] >= g_ka - trust_region_g)
	end

	@constraint(m, beta_k - trust_region_beta .<= beta[1:numPoly] )
	@constraint(m, beta[1:numPoly] .<= beta_k + trust_region_beta)


	obj = DeltaF'*[(beta-beta_k) ; (g -g_k) ; ksi];
	#obj = DeltaF'*[(beta-beta_k) ; (g -g_k) ; (0)];
	@objective(m, Min, obj[1])

	optimize!(m)

	beta_k_1 = [value(beta[i]) for i =1:length(beta)];
	g_k_1 = [value(g[i]) for i =1:length(g)];
	ksi_k_1 = [value(ksi)];
	obj_k_1 = objective_value(m)

	npzwrite("tmp_jl/beta_k_1.npz", beta_k_1)
	npzwrite("tmp_jl/g_k_1.npz", g_k_1)
	npzwrite("tmp_jl/ksi_k_1.npz", ksi_k_1)
end

jointBilevel()
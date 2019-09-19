import sys
import joint.tnet as tnet
import os 
import trafficAssignment.assign as ta



netFile = "networks/NYC_full_net.txt"
gFile = "networks/NYC_full_trips.txt"

#netFile = "networks/EMA_net.txt"
#gFile = "networks/EMA_trips.txt"

#netFile = "networks/Braess1_net.txt"
#gFile = "networks/Braess1_trips.txt"

fcoeffs_truth = [1,0,0,0,0.15,0]
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs_truth)

G = ta.assignment(G=tNet.G, g=tNet.g, fcoeffs=tNet.fcoeffs, flow=False, method='FW', accuracy=0.0001, max_iter=20)
print('-----------------')
tNet.G = G
tNet.fcoeffs = [1,0,0,0,0.20,0]
G = ta.assignment(G=tNet.G, g=tNet.g, fcoeffs=tNet.fcoeffs, flow=True, method='FW', accuracy=0.0001, max_iter=20)
print([G[i][j]['flow'] for i,j in G.edges()])
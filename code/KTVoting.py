import re
import time
import math
import numpy as np
import networkx as nx
from effectivecoverage import effective_cov, read_graph

from config import get_config
params, _ = get_config()

n_subarea = 100
n_users = params.user_no
n_seed_list = [params.seed_no]
n_samplefile = 10
input_file_path = r'../dataset/data_1'
output_result_file_prefix = '../Result/Result_KTVoting/Result_KTVoting_5000'


#read info from txt file into a graph
def readGraph(G,nodefilename,edgefilename, n_subarea):
    nodefile = open(nodefilename)
    newnode = nodefile.readline()
    while newnode:
        nodeId = int(newnode.split('\t')[0])
        nodeWeight = list()
        for i in range(0, n_subarea):
            nodeWeight.append(float(newnode.split('\t')[i+1]))
        G.add_node(nodeId, weight=nodeWeight)
        newnode = nodefile.readline()
    edgefile = open(edgefilename)
    newedge = edgefile.readline()
    while newedge:
        node1 = int(newedge.split('\t')[0])
        node2 = int(newedge.split('\t')[1])
        edgeWeight =  float(newedge.split('\t')[2])
        G.add_weighted_edges_from([(node1, node2, edgeWeight)])
        G.add_weighted_edges_from([(node2, node1, edgeWeight)])
        newedge = edgefile.readline()
    return G


def KTVoting(seed_num, subarea_num, beta, T, G):
    S = set()
    dvote0 = dict()
    dvote1 = dict()
    vote = dict()
    ntmp = math.floor(subarea_num/seed_num)
    for v in set(G._node):
        dvote0[v] = list()
        for i in range(0, seed_num):
            vtmp = 0
            for j in range(0, ntmp):
                if (i*ntmp+j)<subarea_num:
                    vtmp += G._node[v]['weight'][i*ntmp+j]
            dvote0[v].append(vtmp)
        vote[v] = [0 for i in range(0, seed_num)]
        dvote1[v] = [0 for i in range(0, seed_num)]
    for t in range(0, T):
        for v in set(G._node):
            for i in range(0, seed_num):
                dvote1[v][i] = 0
                for u in set(nx.neighbors(G, v)):
                    dvote1[v][i] = dvote1[v][i] + G[v][u]['weight'] * dvote0[u][i]
                vote[v][i] = vote[v][i] + dvote1[v][i]
        for v in set(G._node):
            for i in range(0, seed_num):
                dvote0[v][i] = dvote1[v][i]
    Dm = dict()
    finalV = dict()
    for v in set(G._node):
        Dm[v] = set()
        maxV = 0
        for i in range(0, seed_num):
            if vote[v][i] > maxV:
                maxV = vote[v][i]
        finalV[v] = 0
        for i in range(0, seed_num):
            if vote[v][i] == maxV:
                Dm[v].add(i)
                finalV[v] = finalV[v] + beta * vote[v][i]
            else:
                finalV[v] = finalV[v] + (1-beta) * vote[v][i]
    for i in range(0, seed_num):
        maxs = 0
        maxV = 0
        for v in set(G._node)-S:
            if i in Dm[v] and finalV[v] > maxV:
                maxs = v
                maxV = finalV[v]
        S.add(maxs)
    while len(S)<seed_num:
        maxs = 0
        maxV = 0
        for v in set(G._node)-S:
            if finalV[v] >= maxV:
                maxs = v
                maxV = finalV[v]
        S.add(maxs)

    return S

def KTVoting2(seed_num, subarea_num, beta, T, G):
    S = set()
    dvote0 = dict()
    dvote1 = dict()
    vote = dict()
    for v in set(G._node):
        dvote0[v] = list()
        for i in range(0, subarea_num):
            dvote0[v].append(G._node[v]['weight'][i])
        vote[v] = [0 for i in range(0, subarea_num)]
        dvote1[v] = [0 for i in range(0, subarea_num)]
    for t in range(0, T):
        for v in set(G._node):
            for i in range(0, subarea_num):
                dvote1[v][i] = 0
                for u in set(nx.neighbors(G, v)):
                    dvote1[v][i] = dvote1[v][i] + G[v][u]['weight'] * dvote0[u][i]
                vote[v][i] = vote[v][i] + dvote1[v][i]
        for v in set(G._node):
            for i in range(0, subarea_num):
                dvote0[v][i] = dvote1[v][i]
    Dm = dict()
    finalV = dict()
    for v in set(G._node):
        Dm[v] = set()
        maxV = 0
        for i in range(0, subarea_num):
            if vote[v][i] > maxV:
                maxV = vote[v][i]
        finalV[v] = 0
        for i in range(0, subarea_num):
            if vote[v][i] == maxV:
                Dm[v].add(i)
                finalV[v] = finalV[v] + beta * vote[v][i]
            else:
                finalV[v] = finalV[v] + (1-beta) * vote[v][i]
    for i in range(0, subarea_num):
        maxs = 0
        maxV = 0
        for v in set(G._node)-S:
            if i in Dm[v] and finalV[v] > maxV:
                maxs = v
                maxV = finalV[v]
        S.add(maxs)
    while len(S)<seed_num:
        maxs = 0
        maxV = 0
        for v in set(G._node)-S:
            if finalV[v] >= maxV:
                maxs = v
                maxV = finalV[v]
        S.add(maxs)

    return S

def resultKTVoting(n_subarea, n_users, n_samplefile,
                    input_file_path, output_result_file):

    n = len(n_seed_list)
    writefile_EC = open(output_result_file_prefix+'_EC.txt', 'w')
    writefile_stdEC = open(output_result_file_prefix+'_stdEC.txt', 'w')
    writefile_runtime = open(output_result_file_prefix+'_runtime.txt', 'w')
    
    first_line = 'seed_num'
    for k in n_seed_list:
        first_line += ('\t'+str(k))
    first_line = first_line+'\n'
    writefile_EC.write(first_line)
    writefile_stdEC.write(first_line)
    writefile_runtime.write(first_line)
    g = dict()
    g_quality = dict()
    sample_list = [params.sample_file_no]
    for i in sample_list:

        
        nodefilename = input_file_path+'/input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path+'/input_edge_' + str(n_users) + '_' + str(i) + '.txt'
        
        ECresult_line = 'Input '+str(i)
        stdECresult_line = 'Input '+str(i)
        runtimeresult_line = 'Input '+str(i)
        
        # g.clear()
        # g_quality.clear()
        # g, g_quality = read_from_txt(nodefilename, edgefilename, n_subarea)

        for j in range(0, n):
            G = nx.DiGraph()
            G = readGraph(G, nodefilename, edgefilename, n_subarea)
            n_seed = n_seed_list[j]
            start = time.process_time()
            S = KTVoting2(n_seed,n_subarea, 0.9, 4, G)
            end = time.process_time()
            t = end - start
            # EC, stdEC, allEC = probeffectivecoverage(g, g_quality, S, n_subarea)
            nx_G = read_graph(nodefilename, edgefilename, n_subarea)
            EC, stdEC = effective_cov(nx_G, S, n_subarea)
            ECresult_line += ('\t'+str(format(EC, '.4f')))
            stdECresult_line += ('\t'+str(format(stdEC, '.4f')))
            runtimeresult_line += ('\t'+str(format(t,'.4f')))
            print('Sample: ',i,' Seed number: ', n_seed, ' EC: ', format(EC, '.4f'), ' stdEC: ', format(stdEC, '.4f'), ' t: ', format(t,'.4f'))
        
        ECresult_line += '\n'
        stdECresult_line += '\n'
        runtimeresult_line += '\n'
        writefile_EC.write(ECresult_line)
        writefile_stdEC.write(stdECresult_line)
        writefile_runtime.write(runtimeresult_line)
        del G

    writefile_EC.close()
    writefile_stdEC.close()
    writefile_runtime.close()

    
# resultKTVoting(n_subarea, n_users, n_samplefile,input_file_path, output_result_file_prefix)
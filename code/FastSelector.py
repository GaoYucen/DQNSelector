import re
import time
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
output_result_file_prefix = '../Result/Result_FastSelector/Result_FastSelector_sample1allEC_5000'

def readGraph(G, nodefilename, edgefilename, n_subarea):
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


def cos_sim(x,y):
    a = np.mat(x)
    b = np.mat(y)
    num = float(a * b.T)
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    if denom == 0:
        return 0
    else:
        sim = 0.5 + 0.5 * (num / denom)
        return sim


def FastSelector(k, beta, h, G):
    # k is the number of seeds
    # beta is the paramter of DegreeRank and TriDiffRank
    # h is the number of subareas
    S = set()
    DegreeAll = dict(G.degree())
    apVector = np.zeros(h)
    while len(S)<k:
        # Calclate arg max R(u)
        TriCosSim = dict()
        Degree = dict()
        DegreeRank = dict()
        TriDiffRank = dict()
        for v in set(G._node):
            if v not in S:
                ap = np.array(G._node[v]['weight'])
                TriCosSim[v] = cos_sim(ap,apVector)
                Degree[v] = DegreeAll[v]
        DegreeRankInd = sorted(Degree.items(), key=lambda x: x[1], reverse=True)
        TriDiffRankInd = sorted(TriCosSim.items(), key=lambda x: x[1], reverse=True)
        for i in range(0,len(DegreeRankInd)):
            DegreeRank[DegreeRankInd[i][0]] = i+1
            TriDiffRank[TriDiffRankInd[i][0]] = i+1
        minR = 2*len(DegreeRank)
        u = 0
        for v in DegreeRank.keys():
            R = beta*DegreeRank[v]+(1-beta)*TriDiffRank[v]
            if R<minR:
                minR = R
                u = v
        S.add(u)
    return S


def resultFastSelector(n_subarea, n_users, n_samplefile, input_file_path, output_result_file):

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
    samplefilelist = [params.sample_file_no]
    for i in samplefilelist: #range(0, n_samplefile):
        # G = nx.Graph()
        nodefilename = input_file_path+'/input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path+'/input_edge_' + str(n_users) + '_' + str(i) + '.txt'
        
        ECresult_line = 'Input '+str(i)
        stdECresult_line = 'Input '+str(i)
        runtimeresult_line = 'Input '+str(i)
        
        # g.clear()
        # g_quality.clear()
        
        # g, g_quality = read_from_txt(nodefilename, edgefilename, n_subarea)
        # G = readGraph(G, nodefilename, edgefilename, n_subarea)
        for j in range(0, n):
            G = nx.DiGraph()
            G = readGraph(G, nodefilename, edgefilename, n_subarea)
            n_seed =n_seed_list[j]
            start = time.process_time()
            S = FastSelector(n_seed, 0.56, n_subarea, G)
            end = time.process_time()
            t = end - start
            # EC, stdEC, allEC = probeffectivecoverage(g, g_quality, S, n_subarea)
            # ECresult_line = ''
            # for k in range(0,len(allEC)):
            #     if (k+1)%10==0:
            #         ECresult_line += '\t'+str(allEC[k])
            #         ECresult_line += '\n'
            #         writefile_EC.write(ECresult_line)
            #         # print(ECresult_line)
            #         ECresult_line = ''
            #     else:
            #         ECresult_line += '\t'+str(allEC[k])
            nx_G = read_graph(nodefilename, edgefilename, n_subarea)
            EC, stdEC = effective_cov(nx_G, S, n_subarea)
            # print(EC)
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


# resultFastSelector(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)

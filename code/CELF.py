import re
import time
import networkx as nx
import numpy as np
from effectivecoverage import effective_cov, read_graph

from config import get_config
params, _ = get_config()

n_subarea = 100
n_users = params.user_no
n_seed_list = [params.seed_no]
n_samplefile =10

input_file_path = r'../dataset/data_1'
output_result_file_prefix = r'../Result/Result_CELF/Result_CELF_sample1_allEC_3000'

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

def CELF(G, k):
    marg_gain = [effective_cov(G, [node], n_subarea)[0] for node in G.nodes()]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(G.nodes(), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q = Q[1:]

    for _ in range(k - 1):

        check, node_lookup = False, 0

        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, effective_cov(G, S+[current], n_subarea)[0] - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)

        # Remove the selected node from the list
        Q = Q[1:]

    return S, SPREAD


def resultCELF(n_subarea, n_users, n_samplefile,
                    input_file_path, output_result_file):
    n = len(n_seed_list)
    writefile_EC = open(output_result_file_prefix + '_EC.txt', 'w')
    writefile_stdEC = open(output_result_file_prefix + '_stdEC.txt', 'w')
    writefile_runtime = open(output_result_file_prefix + '_runtime.txt', 'w')

    first_line = 'seed_num'
    for k in n_seed_list:
        first_line += ('\t' + str(k))
    first_line = first_line + '\n'
    writefile_EC.write(first_line)
    writefile_stdEC.write(first_line)
    writefile_runtime.write(first_line)
    g = dict()
    g_quality = dict()
    sample_file_list = [params.sample_file_no]
    # sample_file_list = [3]
    for i in sample_file_list:

        nodefilename = input_file_path + '/input_node_' + str(n_users) + '_' + str(i) + '.txt'
        edgefilename = input_file_path + '/input_edge_' + str(n_users) + '_' + str(i) + '.txt'

        ECresult_line = 'Input ' + str(i)
        stdECresult_line = 'Input ' + str(i)
        runtimeresult_line = 'Input ' + str(i)

        # g.clear()
        # g_quality.clear()
        # g, g_quality = read_from_txt(nodefilename, edgefilename, n_subarea)
        for j in range(0, n):
            G = nx.DiGraph()
            G = readGraph(G, nodefilename, edgefilename, n_subarea)
            n_seed = n_seed_list[j]
            start = time.process_time()
            S = CELF(G, n_seed)[0]
            end = time.process_time()
            t = end - start
            nx_G = read_graph(nodefilename, edgefilename, n_subarea)
            EC, stdEC = effective_cov(nx_G, S, n_subarea)
            ECresult_line += ('\t' + str(format(EC, '.4f')))
            stdECresult_line += ('\t' + str(format(stdEC, '.4f')))
            runtimeresult_line += ('\t' + str(format(t, '.4f')))
            print('Sample: ', i, ' Seed number: ', n_seed, ' EC: ', format(EC, '.4f'), ' stdEC: ', format(stdEC, '.4f'),
                  ' t: ', format(t, '.4f'))

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


# resultCELF(n_subarea, n_users, n_samplefile, input_file_path, output_result_file_prefix)

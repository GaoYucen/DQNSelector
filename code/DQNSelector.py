# %% md

## 1. Import package

import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random
import time
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# %% md

## 2. Function definitions

# %% md

#### 2.1 Graph reader

# %%

def read_graph(node_file_name, edge_file_name, n_subarea):
    G = nx.DiGraph()

    nodefile = open(node_file_name)
    newnode = nodefile.readline()
    while newnode:
        nodeId = int(newnode.split('\t')[0])
        nodeWeight = list()
        for i in range(0, n_subarea):
            nodeWeight.append(float(newnode.split('\t')[i + 1]))
        G.add_node(nodeId, weight=nodeWeight)
        newnode = nodefile.readline()
    nodefile.close()

    edgefile = open(edge_file_name)
    newedge = edgefile.readline()
    while newedge:
        node1 = int(newedge.split('\t')[0])
        node2 = int(newedge.split('\t')[1])
        edgeWeight = float(newedge.split('\t')[2])
        G.add_weighted_edges_from([(node1, node2, edgeWeight)])
        newedge = edgefile.readline()
    edgefile.close()

    return G


# %% md

#### 2.2 Get embeddings

# %% md

##### 2.2.1 Mean square error-based influence embedding

# %%

def one_time_monte_carlo_simulation(nxG, start_node):
    all_activated_node = set([start_node])
    curr_active_node = set([start_node])
    next_active_node = set()
    mc_pairs = []

    for node in curr_active_node:
        for nbr in nxG.neighbors(node):
            randval = random.random()
            if (nbr not in all_activated_node) and randval < nxG[node][nbr]['weight']:
                next_active_node.add(nbr)

    while len(curr_active_node) != 0 and len(next_active_node) != 0:
        curr_active_node = next_active_node.copy()
        all_activated_node = all_activated_node | curr_active_node
        next_active_node = set()
        for node in curr_active_node:
            for nbr in nxG.neighbors(node):
                randval = random.random()
                if (nbr not in all_activated_node) and randval < nxG[node][nbr]['weight']:
                    next_active_node.add(nbr)

    for node in all_activated_node:
        if node != start_node:
            mc_pairs.append((start_node, node))

    return mc_pairs


# Get monte carlo probability
def monte_carlo_simulations(MC_times, nxG, start_node):
    times_dict = dict()
    for i in range(0, MC_times):
        mc_pairs = one_time_monte_carlo_simulation(nxG, start_node)
        for pair in mc_pairs:
            if pair[1] not in times_dict.keys():
                times_dict[pair[1]] = 1
            else:
                times_dict[pair[1]] = times_dict[pair[1]] + 1

    for node in times_dict.keys():
        times_dict[node] = times_dict[node] / MC_times
    # Return the probability of monte carlo simulation
    return times_dict


def directed_embedding_learning_mse_balanced(nxG, embedding_dim=32, MC_times=10, lr=0.01, iter_times=100):
    # 初始化
    allnodes = list(nxG.nodes())
    seed_embedding_dict = dict()
    influenced_embedding_dict = dict()
    for node in allnodes:
        seed_embedding_dict[node] = np.random.rand(embedding_dim)
        influenced_embedding_dict[node] = np.random.rand(embedding_dim)

    for l in range(0, iter_times):
        # if (l+1)%10==0:
        print('Embedding learning: %d/%d' % (l + 1, iter_times))

        random.shuffle(allnodes)

        for node in allnodes:
            prob_dict = monte_carlo_simulations(MC_times, nxG, node)
            pos_samples = list(prob_dict.keys())
            n_pos_sample = len(pos_samples)
            neg_samples = []
            neg_samples_set = set(allnodes.copy()) - set(prob_dict.keys())
            while len(neg_samples) < n_pos_sample:
                influenced_node = random.choice(list(neg_samples_set))
                neg_samples.append(influenced_node)

            samples = pos_samples + neg_samples
            random.shuffle(samples)

            prob_true = [0 for i in range(0, len(samples))]

            for i in range(0, len(samples)):
                if samples[i] in pos_samples:
                    prob_true[i] = prob_dict[samples[i]]

            v = node
            for i in range(0, len(samples)):
                u = samples[i]
                tmp_val = seed_embedding_dict[v] @ influenced_embedding_dict[u]
                tmp = seed_embedding_dict[v] - 2 * lr * (tmp_val - prob_true[i]) * influenced_embedding_dict[u]
                influenced_embedding_dict[u] = influenced_embedding_dict[u] - 2 * lr * (tmp_val - prob_true[i]) * \
                                               seed_embedding_dict[v]
                seed_embedding_dict[v] = tmp

    return seed_embedding_dict, influenced_embedding_dict


def directed_embedding_learning_mse_balanced_2(nxG, embedding_dim=32, lr=0.01, MC_times=10, MC_iter_times=10,
                                               train_iter_times=30):
    allnodes = list(nxG.nodes())
    seed_embedding_dict = dict()
    influenced_embedding_dict = dict()
    for node in allnodes:
        seed_embedding_dict[node] = np.random.uniform(low=-1 / embedding_dim, high=1 / embedding_dim,
                                                      size=embedding_dim)
        influenced_embedding_dict[node] = np.random.uniform(low=-1 / embedding_dim, high=1 / embedding_dim,
                                                            size=embedding_dim)

    samples = []
    for l in range(0, MC_iter_times):

        print('Embedding learning: %d/%d' % (l, MC_iter_times))

        for node in allnodes:
            prob_dict = monte_carlo_simulations(MC_times, nxG, node)
            pos_samples = list(prob_dict.keys())
            n_pos_sample = len(pos_samples)
            neg_samples = []
            neg_samples_set = set(allnodes.copy()) - set(prob_dict.keys())
            while len(neg_samples) < n_pos_sample:
                influenced_node = random.choice(list(neg_samples_set))
                neg_samples.append(influenced_node)

            for u in pos_samples:
                samples.append([node, u, prob_dict[u]])
            for u in neg_samples:
                samples.append((node, u, 0.0))

    for l in range(0, train_iter_times):
        random.shuffle(samples)
        print('Train learning: %d/%d' % (l, train_iter_times))
        for i in range(0, len(samples)):
            v = samples[i][0]
            u = samples[i][1]
            p = samples[i][2]
            tmp_val = seed_embedding_dict[v] @ influenced_embedding_dict[u]
            tmp = seed_embedding_dict[v] - 2 * lr * (tmp_val - p) * influenced_embedding_dict[u]
            influenced_embedding_dict[u] = influenced_embedding_dict[u] - 2 * lr * (tmp_val - p) * seed_embedding_dict[
                v]
            seed_embedding_dict[v] = tmp
    return seed_embedding_dict, influenced_embedding_dict


# %% md

##### 2.2.2 Coverage embedding

# %%
def coverage_step_embedding(Graph, cov_step, seed, influence):
    step_cov_embedding = {}
    all_nodes = list(Graph.nodes())

    for node in all_nodes:
        n_dim = len(np.array(Graph.node[node]['weight']))
        step_cov_embedding[node] = np.zeros(n_dim)

    for node in all_nodes:
        step_cov_embedding_chain1 = Graph.neighbors(node)
        for i in range(0, cov_step):
            step_cov_embedding_chain2 = []
            for nbr in step_cov_embedding_chain1:
                step_cov_embedding[node] += np.dot(seed[node],influence[nbr].reshape(-1,1))*Graph.node[nbr]['weight']
                step_cov_embedding_chain2 += Graph.neighbors(nbr)
                step_cov_embedding_chain2 = list(set(step_cov_embedding_chain2))
            step_cov_embedding_chain1 = step_cov_embedding_chain2

    maxval = np.zeros(n_dim)
    for i in range(0, n_dim):
        for node in step_cov_embedding.keys():
            maxval[i] = max(step_cov_embedding[node][i], maxval[i])

    for node in step_cov_embedding.keys():
        step_cov_embedding[node] = np.true_divide(step_cov_embedding[node], maxval)
    return step_cov_embedding


def coverage_step(Graph, cov_step=4):
    step_cov_embedding = {}
    step_cov_embedding_chain1 = {} #neighbor node weight
    step_cov_embedding_chain2 = {}
    all_nodes = list(Graph.nodes())

    for node in all_nodes:
        step_cov_embedding_chain2[node] = np.array(Graph._node[node]['weight'])
        n_dim = len(np.array(Graph._node[node]['weight']))
        step_cov_embedding[node] = np.zeros(n_dim)
        step_cov_embedding_chain1[node] = np.zeros(n_dim)

    for i in range(0, cov_step):
        for node in all_nodes:
            for nbr in Graph.neighbors(node):
                step_cov_embedding_chain1[node] += step_cov_embedding_chain2[nbr] * Graph[node][nbr]['weight'] # edge weight

        for node in all_nodes:
            step_cov_embedding[node] += step_cov_embedding_chain1[node]
            step_cov_embedding_chain2[node] = step_cov_embedding_chain1[node]
            step_cov_embedding_chain1[node] = np.zeros(n_dim)

    del step_cov_embedding_chain1
    del step_cov_embedding_chain2

    maxval = np.zeros(n_dim)
    for i in range(0, n_dim):
        for node in step_cov_embedding.keys():
            maxval[i] = max(step_cov_embedding[node][i], maxval[i])

    for node in step_cov_embedding.keys():
        step_cov_embedding[node] = np.true_divide(step_cov_embedding[node], maxval)
    return step_cov_embedding


def get_node2vec_cov_embeddings(nx_G, node2vec_p=0.5, node2vec_q=0.5, num_walk=100, walk_len=8,
                                word2vec_embedding_dim=100, word2vec_window=4, word2vec_iter=5, cov_step=4):
    # Get node2vec embeddings
    G = node2vec.Graph(nx_G, p=node2vec_p, q=node2vec_q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=num_walk, walk_length=walk_len)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(sentences=walks, vector_size=word2vec_embedding_dim, window=word2vec_window, workers=1)
    node2vec_embedding = {}
    for index, word in enumerate(model.wv.index_to_key):
        node2vec_embedding[int(word)] = model.wv.vectors[index]

    # Get step coverage value
    step_cov_embedding = coverage_step(nx_G, cov_step=cov_step)

    return node2vec_embedding, step_cov_embedding


# %% md

#### 2.3 Deep Q Network

# %%

class DeepQNetwork:
    def __init__(
            self,
            n_actions=1,
            n_features=100,
            n_l1=20,
            n_l2=20,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.n_l1 = n_l1
        self.n_l2 = n_l2

        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 4 + 1))

        self._build_net()
        self.saver = tf.train.Saver(max_to_keep=300)

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features * 2], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_l1, \
                tf.random_normal_initializer(0., 0.5), tf.constant_initializer(0.1)  # config of layers

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features * 2, self.n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features * 2], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features * 2, self.n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w2', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, action_list, action_embedding, env_embedding):
        # to have batch dimension when feed into tf placeholder
        env_embedding_batch = env_embedding[np.newaxis, :].repeat(action_embedding.shape[0], axis=0)
        observation = np.concatenate([env_embedding_batch, action_embedding], axis=1)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_ind = np.argmax(actions_value)
            action = action_list[action_ind]
            # print("Selected seed (max Q-Value): ", action, 'max_val: ', np.amax(actions_value), ' min_val: ', np.amin(actions_value))
        else:
            action = random.choice(action_list)
            # print("Selected seed (random): ", action, flush=True)
        return action

    def choose_action_no_ran(self, action_list, action_embedding, env_embedding):
        env_embedding_batch = env_embedding[np.newaxis, :].repeat(action_embedding.shape[0], axis=0)
        observation = np.concatenate([env_embedding_batch, action_embedding], axis=1)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action_ind = np.argmax(actions_value)
        action = action_list[action_ind]
        # print("Selected seed (max Q-Value): ", action, flush=True)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        # check to replace target parameters
        # print('========learning========')
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -2 * self.n_features:],  # fixed params
                self.s: batch_memory[:, :2 * self.n_features],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, 2*self.n_features].astype(int)

        reward = batch_memory[:, 2 * self.n_features]
        reward = reward[:, np.newaxis]
        q_target = reward + self.gamma * q_next
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :2 * self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def get_cost_history(self):
        return self.cost

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)

    def load_model(self, load_path):
        self.saver.restore(self.sess, load_path)


# %% md

#### 2.4 Effective Coverage

# %%

def effective_cov(nx_G, seedset, n_subareas):
    """
    calculate a node's effective coverage in IC（Independent Cascade）model by Monte Carlo
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """

    paticipate_prob = dict()
    quality = dict()
    all_nodes = list(nx_G.nodes())
    for node in all_nodes:
        quality[node] = np.array(nx_G._node[node]['weight'])
        if node in seedset:
            paticipate_prob[node] = 1
        else:
            paticipate_prob[node] = 0

    all_activated_node = set(seedset)
    current_active_node = set(seedset)
    next_active_node = set()
    for node in current_active_node:
        for nbr in nx_G.neighbors(node):
            if nbr not in all_activated_node:
                next_active_node.add(nbr)

    while len(current_active_node) != 0 and len(next_active_node) != 0:
        for node in current_active_node:
            for nbr in nx_G.neighbors(node):
                if nbr in next_active_node:
                    paticipate_prob[nbr] = 1 - (1 - paticipate_prob[nbr]) * (
                                1 - paticipate_prob[node] * nx_G[node][nbr]['weight'])
        current_active_node = next_active_node.copy()
        all_activated_node = all_activated_node | current_active_node
        next_active_node = set()
        for node in current_active_node:
            for nbr in nx_G.neighbors(node):
                if nbr not in all_activated_node:
                    next_active_node.add(nbr)

    EC = np.zeros(n_subareas)
    for node in paticipate_prob.keys():
        if node in quality.keys():
            EC += paticipate_prob[node] * quality[node]

    return np.mean(EC), np.std(EC)


# %% md

#### 2.5 Train and testing

# %%

def train(RL, nx_G, node2vec_embedding, step_cov_embedding, checkpoints_path_prefix,
          batchsize=32, n_l1=10, n_l2=20, episode=100, seed_number_training=4, n=4, word2vec_embedding_dim=20,
          n_subarea=1):
    print('========training========')
    node_embedding = dict()
    node_deg = list()
    for node in node2vec_embedding.keys():
        tmp_embed = np.concatenate([step_cov_embedding[node], node2vec_embedding[node]])
        if np.sum(np.isnan(tmp_embed)) == 0:
            node_embedding[node] = tmp_embed
        node_deg.append((node, nx_G.degree(node)))
    node_deg.sort(key=lambda x: x[1], reverse=True)

    seed_initial = set()
    seed_initial_set_embedding = np.zeros(n_subarea + word2vec_embedding_dim)

    # Define a RL
    # RL = DeepQNetwork(n_actions=1, n_features=(n_subarea+word2vec_embedding_dim),
    #                  n_l1=n_l1,
    #                  n_l2=n_l2,
    #                  batch_size=batchsize)

    all_nodes = set(node2vec_embedding.keys()) - seed_initial

    for i in range(0, episode):
        seed_set = seed_initial.copy()
        seed_set_embedding = seed_initial_set_embedding
        for t in range(1, seed_number_training + 1):
            candidate_seed = list(all_nodes - seed_set)
            candidate_seed_embedding = np.array([node_embedding[node] for node in candidate_seed])
            selected_seed = RL.choose_action(candidate_seed, candidate_seed_embedding, seed_set_embedding)

            reward = effective_cov(nx_G, seed_set | {selected_seed}, n_subarea)[0] - \
                     effective_cov(nx_G, seed_set, n_subarea)[0]

            seed_set_embedding_ = seed_set_embedding + node_embedding[selected_seed]
            next_seed_set = seed_set | {selected_seed}
            next_candidate_seed = list(all_nodes - next_seed_set)
            next_candidate_seed_embedding = np.array([node_embedding[node] for node in next_candidate_seed])
            next_selected_seed = RL.choose_action_no_ran(next_candidate_seed,
                                                         next_candidate_seed_embedding, seed_set_embedding_)

            RL.store_transition(seed_set_embedding, node_embedding[selected_seed], reward,
                                np.hstack((seed_set_embedding_, node_embedding[next_selected_seed])))

            seed_set.add(selected_seed)
            seed_set_embedding = seed_set_embedding + node_embedding[selected_seed]
            if t % 50 == 0:
                print('Episode %d/%d, step %d/%d' % (i + 1, episode, t, seed_number_training))
                print(' Selected_seed: ', selected_seed, ' reward: ', reward, ' eff_cov: ',
                      effective_cov(nx_G, seed_set, n_subarea)[0])

            if t % n == 0:
                RL.learn()
        RL.save_model(checkpoints_path_prefix + 'episode_' + str(i) + '_ec_' + str(
            format(effective_cov(nx_G, seed_set, n_subarea)[0], '.4f')) + '.ckpt')
    return RL


def test(nx_G, node2vec_embedding, step_cov_embedding, RL, k,
         n_subarea=10, word2vec_embedding_dim=10):
    print('========testing========')
    node_embedding = dict()
    node_deg = []
    for node in node2vec_embedding.keys():
        tmp_embed = np.concatenate([step_cov_embedding[node], node2vec_embedding[node]])
        if np.sum(np.isnan(tmp_embed)) == 0:
            node_embedding[node] = tmp_embed
        node_deg.append((node, nx_G.degree(node)))
    node_deg.sort(key=lambda x: x[1], reverse=True)

    seed_initial = set()
    seed_initial_set_embedding = np.zeros(n_subarea + word2vec_embedding_dim)

    all_nodes = set(node_embedding.keys()) - seed_initial
    seed_set = seed_initial.copy()
    seed_set_embedding = seed_initial_set_embedding
    while len(seed_set) < k:
        candidate_seed = list(all_nodes - seed_set)
        candidate_seed_embedding = np.array([node_embedding[node] for node in candidate_seed])
        selected_seed = RL.choose_action_no_ran(candidate_seed, candidate_seed_embedding, seed_set_embedding)
        seed_set.add(selected_seed)
        seed_set_embedding = seed_set_embedding + node_embedding[selected_seed]

    return seed_set


# %% md

## 3. Experiments

# %%

n_subarea = 100
n_users = 5000
n_samplefile = 10
input_file_path = r'../dataset/data_1'
input_index = 1
output_result_file_prefix = '../Result/Result_DQNSelector_n5000_trainingseed500_trainingsample4/Result_DQNSelector_5000'

# %%

# RL Agent (Deep Q Network Setting)
batchsize = 128
n_l1 = 50
n_l2 = 20
episode = 200
seed_number_training = 500
n = 10

# %%

nodefilename = input_file_path + '/input_node_' + str(n_users) + '_' + str(input_index) + '.txt'
edgefilename = input_file_path + '/input_edge_' + str(n_users) + '_' + str(input_index) + '.txt'
nx_G = read_graph(nodefilename, edgefilename, n_subarea)

# %%

# seed_embedding_dict_mse_balanced, influenced_embedding_dict_mse_balanced = directed_embedding_learning_mse_balanced_2(
#     nx_G, embedding_dim=128,
#     lr=0.01, MC_times=10, MC_iter_times=5, train_iter_times=5)
# step_cov_embedding = coverage_step(nx_G, cov_step=4)
# step_cov_embedding = coverage_step_embedding(nx_G, cov_step=4, seed=seed_embedding_dict_mse_balanced, influence=influenced_embedding_dict_mse_balanced)

# %%

# save embedding
# np.save('../DQNSelector_checkpoints_5000_seed500_sample1/seed_embedding_dict_mse_balanced',
#         seed_embedding_dict_mse_balanced)
# np.save('../DQNSelector_checkpoints_5000_seed500_sample1/influenced_embedding_dict_mse_balanced',
#         influenced_embedding_dict_mse_balanced)
# np.save('../DQNSelector_checkpoints_5000_seed500_sample1/step_cov_embedding', step_cov_embedding)

# %%

seed_embedding_dict_mse_balanced = np.load(
    '../DQNSelector_checkpoints_5000_seed500_sample1/seed_embedding_dict_mse_balanced.npy',
    allow_pickle=True).item()
step_cov_embedding = np.load('../DQNSelector_checkpoints_5000_seed500_sample1/step_cov_embedding.npy',
                             allow_pickle=True).item()

# %%

RL = DeepQNetwork(n_actions=1, n_features=(n_subarea + 128),
                  n_l1=n_l1,
                  n_l2=n_l2,
                  batch_size=batchsize)


# %%

RL = train(RL, nx_G, seed_embedding_dict_mse_balanced, step_cov_embedding,
           '../DQNSelector_checkpoints_5000_seed500_sample1/',
           batchsize=batchsize, n_l1=n_l1, n_l2=n_l2, episode=episode, seed_number_training=seed_number_training, n=n,
           word2vec_embedding_dim=128, n_subarea=n_subarea)

# %%

RL.save_model('../test.ckpt')
#
# # %% md
#
# ##### Single sample testing
#
# # %%
#
# RL = DeepQNetwork(n_actions=1, n_features=(n_subarea + 128),
#                   n_l1=n_l1,
#                   n_l2=n_l2,
#                   batch_size=batchsize)
#
# # %%
#
# seed_embedding_dict_mse_balanced = np.load(
#     '../DQNSelector_checkpoints_5000_seed500_sample4/seed_embedding_dict_mse_balanced.npy',
#     allow_pickle=True).item()
# step_cov_embedding = np.load('../DQNSelector_checkpoints_5000_seed500_sample4/step_cov_embedding.npy',
#                              allow_pickle=True).item()
#
# # %%
#
# RL.load_model('../DQNSelector_checkpoints_5000_seed500_sample4/episode_2_ec_645.0761.ckpt')
#
# # %%
#
# n_seed_list = [1000]
# writefile_EC = open(output_result_file_prefix + '_EC.txt', 'w')
# writefile_stdEC = open(output_result_file_prefix + '_stdEC.txt', 'w')
# writefile_runtime = open(output_result_file_prefix + '_runtime.txt', 'w')
#
# first_line = 'seed_num'
# for k in n_seed_list:
#     first_line += ('\t' + str(k))
# first_line = first_line + '\n'
# writefile_EC.write(first_line)
# writefile_stdEC.write(first_line)
# writefile_runtime.write(first_line)
#
# nodefilename = input_file_path + '/input_node_' + str(n_users) + '_' + str(input_index) + '.txt'
# edgefilename = input_file_path + '/input_edge_' + str(n_users) + '_' + str(input_index) + '.txt'
# nx_G = read_graph(nodefilename, edgefilename, n_subarea)
#
# ECresult_line = 'Input ' + str(input_index)
# stdECresult_line = 'Input ' + str(input_index)
# runtimeresult_line = 'Input ' + str(input_index)
#
# for k in n_seed_list:
#     start = time.process_time()
#     seed_set = test(nx_G, seed_embedding_dict_mse_balanced, step_cov_embedding, RL, k,
#                     n_subarea=n_subarea, word2vec_embedding_dim=128)
#     end = time.process_time()
#     EC, stdEC = effective_cov(nx_G, seed_set, n_subarea)
#     t = end - start
#     ECresult_line += ('\t' + str(format(EC, '.4f')))
#     stdECresult_line += ('\t' + str(format(stdEC, '.4f')))
#     runtimeresult_line += ('\t' + str(format(t, '.4f')))
#     print('Sample: ', input_index, ' Seed number: ', k, ' EC: ', format(EC, '.4f'), ' stdEC: ', format(stdEC, '.4f'),
#           ' t: ', format(t, '.4f'))
#
# ECresult_line += '\n'
# stdECresult_line += '\n'
# runtimeresult_line += '\n'
# writefile_EC.write(ECresult_line)
# writefile_stdEC.write(stdECresult_line)
# writefile_runtime.write(runtimeresult_line)
#
# del nx_G
#
# writefile_EC.close()
# writefile_stdEC.close()
# writefile_runtime.close()
#
# #%%
# import numpy as np
#
#
#

#%%
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random
import time

seed_embedding_dict_mse_balanced = np.load(
    'DQNSelector_checkpoints_5000_seed500_sample1/seed_embedding_dict_mse_balanced.npy',
    allow_pickle=True).item()
influenced_embedding_dict_mse_balanced = np.load(
    'DQNSelector_checkpoints_5000_seed500_sample1/influenced_embedding_dict_mse_balanced.npy',
    allow_pickle=True).item()

n_subarea = 100
n_users = 5000
input_file_path = r'dataset/data_1'
input_index = 1

def read_graph(node_file_name, edge_file_name, n_subarea):
    G = nx.DiGraph()

    nodefile = open(node_file_name)
    newnode = nodefile.readline()
    while newnode:
        nodeId = int(newnode.split('\t')[0])
        nodeWeight = list()
        for i in range(0, n_subarea):
            nodeWeight.append(float(newnode.split('\t')[i + 1]))
        G.add_node(nodeId, weight=nodeWeight)
        newnode = nodefile.readline()
    nodefile.close()

    edgefile = open(edge_file_name)
    newedge = edgefile.readline()
    while newedge:
        node1 = int(newedge.split('\t')[0])
        node2 = int(newedge.split('\t')[1])
        edgeWeight = float(newedge.split('\t')[2])
        G.add_weighted_edges_from([(node1, node2, edgeWeight)])
        newedge = edgefile.readline()
    edgefile.close()

    return G

def coverage_step_embedding(Graph, cov_step, seed, influence):
    step_cov_embedding = {}
    all_nodes = list(Graph.nodes())

    for node in all_nodes:
        n_dim = len(np.array(Graph.node[node]['weight']))
        step_cov_embedding[node] = np.zeros(n_dim)

    for node in all_nodes:
        step_cov_embedding_chain1 = Graph.neighbors(node)
        for i in range(0, cov_step):
            step_cov_embedding_chain2 = []
            for nbr in step_cov_embedding_chain1:
                step_cov_embedding[node] += np.dot(seed[node],influence[nbr].reshape(-1,1))*Graph.node[nbr]['weight']
                step_cov_embedding_chain2 += Graph.neighbors(nbr)
                step_cov_embedding_chain2 = list(set(step_cov_embedding_chain2))
            step_cov_embedding_chain1 = step_cov_embedding_chain2

    maxval = np.zeros(n_dim)
    for i in range(0, n_dim):
        for node in step_cov_embedding.keys():
            maxval[i] = max(step_cov_embedding[node][i], maxval[i])

    for node in step_cov_embedding.keys():
        step_cov_embedding[node] = np.true_divide(step_cov_embedding[node], maxval)
    return step_cov_embedding

nodefilename = input_file_path + '/input_node_' + str(n_users) + '_' + str(input_index) + '.txt'
edgefilename = input_file_path + '/input_edge_' + str(n_users) + '_' + str(input_index) + '.txt'
nx_G = read_graph(nodefilename, edgefilename, n_subarea)

step_cov_embedding = coverage_step_embedding(nx_G, cov_step=4, seed=seed_embedding_dict_mse_balanced, influence=influenced_embedding_dict_mse_balanced)

print(step_cov_embedding)

#%%
np.save('DQNSelector_checkpoints_5000_seed500_sample1/step_cov_embedding', step_cov_embedding)



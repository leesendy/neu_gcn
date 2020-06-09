# -*- coding: utf-8 -*-
import sys, os, time
sys.path.insert(0, '..')
from lib import graph, utils, model
import tensorflow as tf
import numpy as np

t_start1 = time.process_time()

"""# Get the laplacian """
coords = utils.get_atlas_coords()
dist, idx = utils.distance_scipy_spatial(coords, k=10, metric='euclidean')
A = graph.adjacency(dist, idx).astype(np.float32)
graphs = []
for i in range(3):
    graphs.append(A)
L = [graph.laplacian(A, normalized=True) for A in graphs]
del A

"""# Input data and labels """
#kind = 'correlation'
#subject_IDs = utils.get_ids()
#networks = utils.load_all_networks(subject_IDs, kind)
#X = np.array(networks)
X = np.zeros(shape=(871, 116, 116))

label_dict = utils.get_subject_label(subject_IDs, label_name='DX_GROUP')
Y = np.array([int(label_dict[x]) - 1 for x in sorted(label_dict)])  # sorted是自动排序，默认升序

# Split the data
train_idx, test_idx, val_idx = utils.give_idx(871, 0.8, 0.1)

train_data = X[train_idx, :, :]
val_data = X[val_idx, :, :]
test_data = X[test_idx, :, :]
train_labels = Y[train_idx]
val_labels = Y[val_idx]
test_labels = Y[test_idx]

t_start = time.process_time()
print('Execution time: {:.2f}s'.format(time.process_time() - t_start1))

""" Neural networks"""
params = {}
params['dir_name'] = 'neu_gcn/'
params['num_epochs'] = 50
params['batch_size'] = 100
params['decay_steps'] = train_data.shape[0] / params['batch_size']
params['eval_frequency'] = int(len(train_idx) / params['batch_size'])
params['filter'] = 'chebyshev5'
params['brelu'] = 'b1relu'
params['pool'] = 'mpool1'
C = 2  # number of classes
params['regularization'] = 5e-4
params['dropout'] = 0.5
params['learning_rate'] = 0.01  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
params['decay_rate'] = 0.95
params['momentum'] = 0.9
params['F'] = [32, 64]  # the number of channel
params['K'] = [3, 3]  # K-hop neighborhood
params['p'] = [1, 1]  # 每次卷积后的池化大小,池化次数与卷积次数一致
params['M'] = [512, C]  # 全连接层输出 M[-1] is the length of output vector
params['len_signal'] = 116  # the length of signal

# Run the model 
model = model.DTI_gcn(L, **params)
accuracy, loss, t_step = model.fit(train_data, train_labels, val_data, val_labels)
# Evaluate the test data

print("Test accuracy is:")
res = model.evaluate(test_data, test_labels)
print(res[0])

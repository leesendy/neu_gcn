# neu_gcn
CNNs on graphs with fast localized spectral filtering code revision
The code can be used in classification the whole graph

# Detail：
https://blog.csdn.net/weixin_43279911/article/details/102920457
by Chinese

Change：
1.All functions are integrated into on one class,named neu_gcn
2.The length of node signal is changed to be greater than one

# Data：
1.X：the data of one subject,size：(871, 116, 116)
871:the numbers of subjects,including training sets and testing sets
116x116:the feature matrix of one subject,each subject has one feature matrix(functional brain networks，.mat file)

2.Y: the labels of all subjects,size:(871,)
871: the numbers of all subjects 

3.L:the adjacency matrix of nodes,size(116, 116)
116: the numbers of nodes, each subject has 116 nodes,each node has a feature vector(the length of the vector is 116)

# Results
Classificating the graph of datasets(one subject one graph),and get the roc/acc.

# Use in your datasets:
The model is not pre-training,so you can only use the model in your data.
1.Change the utils.py for load your datasets
2.Change the main.py for setting the X,Y,L
3.python main.py and get the results

# Requirements
1.TensorFlow 1.12.0
2.Python3.x


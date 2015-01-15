import networkx as nx
import random
import sys
import math
from scipy.stats import rv_discrete  
import numpy as np

def Statistic():

    ave_tag = 0
    ave_pub = 0
    num_user = 0
    for u in graph.nodes():
        if graph.node[u]['types'] == 'user':
            num_user += 1
            for n in graph.neighbors(u):
                if graph.node[n]['types'] == 'tag':
                    ave_tag += 1
                if graph.node[n]['types'] == 'publication':
                    ave_pub += 1

    print 'num user = ',num_user
    print 'total tag = ',ave_tag,'total pub = ',ave_pub

    max_tlink = 0
    max_plink = 0
    Gpub_link = 0
    Gtag_link = 0
    for u in graph.nodes():
        if graph.node[u]['types'] == 'user':
            tag_link = 0
            pub_link = 0
            numP = 0
            numT = 0
            for n in graph.neighbors(u):
                if graph.node[n]['types'] == 'tag':
                    numT += 1
                    tag_link += float(len(graph.edge[u][n]))
                    if len(graph.edge[u][n]) > max_tlink:
                        max_tlink = len(graph.edge[u][n])
                else:
                    numP += 1
                    pub_link += float(len(graph.edge[u][n]))
                    if len(graph.edge[u][n]) > max_plink:
                        max_plink = len(graph.edge[u][n])
            if not numP == 0:
                pub_link = pub_link/float(numP)
            if not numT == 0:
                tag_link = tag_link/float(numT)
            Gpub_link += pub_link
            Gtag_link += tag_link

    Gpub_link = Gpub_link/float(num_user)
    Gtag_link = Gtag_link/float(num_user)
    print 'ave pub link =',Gpub_link
    print 'max pub link =',max_plink
    print 'ave tag link =',Gtag_link
    print 'max tag link =',max_tlink

    return

def SplitData(graph):

    edges_list = []
    for n in graph.nodes():
        if graph.node[n]['types'] == 'user':
            for nn in graph.neighbors(n):
                if not graph.node[nn]['types'] == 'user':
                    edges_list.append((n,nn))
    sample_size = int(float(len(edges_list))*0.1)
    sample_edges = random.sample(edges_list, sample_size)
    
    for e in sample_edges:
        for edge in graph.edge[e[0]][e[1]]:
            graph.edge[e[0]][e[1]][edge]['belongs'] = 'test'
    
    return

def RMSE(graph,edges):
    rmse = 0
    for t in edges:
        if graph.node[t[0]]['types'] == 'tag' or graph.node[t[1]]['types'] == 'tag':
            rmse += math.pow(graph.edge[t[0]][t[1]][0]['y']-graph.edge[t[0]][t[1]][0]['y_p'],2)
    rmse = math.sqrt(rmse/float(len(edges)))
    print 'rmse =',rmse
    return

def Learning(graph, update_iter, sample_iter):
    
    train_edges = []
    for node in graph.nodes():
        if graph.node[node]['types'] == 'user':
            for nn in graph.neighbors(node):
                if graph.node[nn]['types'] == 'tag':
                    for edge in graph.edge[nn][node]:
                        if graph.edge[nn][node][edge]['belongs'] == 'train':
                            train_edges.append((nn,node))
                            break
    print 'train size=',len(train_edges)
    tag_list = []
    for node in graph.nodes():
        if graph.node[node]['types'] == 'tag':
            tag_list.append(node)
    # initial parameter
    print 'initial...'
    lda = 3*[-15] # to do: we can design appropriate initial value of lambda
    m_value = 800
    for e in train_edges:
        y = len(graph.edge[e[0]][e[1]])
        y_p = random.randint(0,m_value)
        for edge in graph.edge[e[0]][e[1]]:
            graph.edge[e[0]][e[1]][edge]['y_p'] = y_p # predicted y
            graph.edge[e[0]][e[1]][edge]['y'] = y # ground truth y
    print 'ramdom guess'
    RMSE(graph,train_edges)
    print 'training...'
    # calculate P(Y|X)
    learning_rate = 0.1
    P_Y_X = 0.0
    exp_lda = math.exp(lda[0])
    sameTag = 0
    for t in tag_list:# two user link to same tag
        user = []
        for n in graph.neighbors(t):
            if graph.node[n]['types'] == 'user' and graph.edge[n][t][0]['belongs'] == 'train':
                user.append(n)
        if len(user) > 1:
            samePub += len(user)
        for u in user:
            for v in user:
                if not v == u:
                    P_Y_X += -0.5*exp_lda*math.pow(graph.edge[t][v][0]['y_p']-graph.edge[t][u][0]['y_p'],2)
    print 'number of users in same publication =',samePub
    record = []
    for i in range(len(train_edges)):
        record.append([])
    for i in range(update_iter):
        exp_lda = math.exp(lda[0])
        for k in train_edges:
            for j in range(sample_iter):
                #sample y_t when y_k is known
                P = [0]*m_value
                if graph.node[k[0]]['types'] == 'tag':
                    tag = k[0]
                    user = k[1]
                elif graph.node[k[1]]['types'] == 'tag':
                    tag = k[1]
                    user = k[0]
                else:
                    continue
                for t in graph.neighbors(tag):
                    if graph.node[t]['types'] == 'user' and (not t == user) and graph.edge[t][tag][0]['belongs'] == 'train':
                        z = 0
                        for v in graph.neighbors(tag):
                            if graph.node[v]['types'] == 'user' and (not v == t) and graph.edge[v][tag][0]['belongs'] == 'train':
                                P_Y_X = P_Y_X + 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-graph.edge[tag][t][0]['y_p'],2)
                        for l in range(m_value):
                            P[l] = P_Y_X

                        for v in graph.neighbors(tag):
                            if graph.node[v]['types'] == 'user' and (not v == t) and graph.edge[v][tag][0]['belongs'] == 'train':
                                if v == user:
                                    for l in range(m_value):
                                        P[l] = P[l] - 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y']-l,2)
                                else:
                                    for l in range(m_value):
                                        P[l] = P[l] - 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-l,2)
                        Sum = 0
                        for l in range(m_value):
                            P[l] = math.exp(P[l])
                            Sum += P[l]
                        for l in range(m_value):
                            if not Sum == 0:
                                P[l] = P[l]/Sum
                            else:
                                P[random.randint(0,m_value-1)] = 1.0
                        distrib = rv_discrete( values=(np.arange(m_value), P) )
                        pre = distrib.rvs(size=1) # sample a value from ditribution
                        for edge in graph.edge[tag][t]:
                            graph.edge[tag][t][edge]['y_p'] = pre[0]
                        #update P(Y|X)
                        temp = 0.0
                        for v in graph.neighbors(tag):
                            if graph.node[v]['types'] == 'user' and (not v == t) and graph.edge[v][tag][0]['belongs'] == 'train':
                                P_Y_X += -0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-graph.edge[tag][t][0]['y_p'],2)
                                temp += -0.5*math.pow(graph.edge[tag][v][0]['y_p']-graph.edge[tag][t][0]['y_p'],2)
                        record[ train_edges.index((tag,t)) ].append( temp )

        E = [0]*len(train_edges)
        for t,k in enumerate(train_edges):
            for s in range(len(record[t])):
                E[t] += record[t][s]
            if not len(record[t]) == 0:
                E[t] = E[t]/float(len(record[t]))
            record[t] = []
        delta = [0]
        for t,k in enumerate( train_edges):
            if graph.node[k[0]]['types'] == 'tag':
                tag = k[0]
                user = k[1]
            elif graph.node[k[1]]['types'] == 'tag':
                tag = k[1]
                user = k[0]
            else:
                continue
            for v in graph.neighbors(tag):
                if graph.node[v]['types'] == 'user' and (not v == user) and graph.edge[v][tag][0]['belongs'] == 'train':
                    delta[0] += -0.5*math.pow(graph.edge[tag][v][0]['y']-graph.edge[tag][user][0]['y'],2)
            delta[0] = delta[0] - E[t]
        delta[0] = exp_lda*delta[0]
        lda[0] = lda[0] + learning_rate*delta[0]
        print 'delta =',delta[0],'lambda =',lda

        RMSE(graph,train_edges)

    return lda[0]

def Inference(graph, sample_iter, lda):

    temperature = 100.0
    coolingRate = 0.9
    absoluteTemperature = 0.1
    exp_lda = math.exp(lda)
    
    #initial prediction
    test_edges = []
    for node in graph.nodes():
        if graph.node[node]['types'] == 'user':
            for nn in graph.neighbors(node):
                if graph.node[nn]['types'] == 'tag':
                    for edge in graph.edge[nn][node]:
                        if graph.edge[nn][node][edge]['belongs'] == 'test':
                            test_edges.append((nn,node))
                            break
    print 'test size=',len(test_edges)
    m_value = 800
    tag_list = []
    for node in graph.nodes():
        if graph.node[node]['types'] == 'tag':
            tag_list.append(node)
    for e in test_edges:
        y = len(graph.edge[e[0]][e[1]])
        y_p = random.randint(0,value)
        for edge in graph.edge[e[0]][e[1]]:
            graph.edge[e[0]][e[1]][edge]['y_p'] = y_p # predicted y
            graph.edge[e[0]][e[1]][edge]['y'] = y # ground truth y

    #Gibbs sampling initialization
    P_Y_X = 0
    sameTag = 0
    for t in tag_list:# two user link to same tag
        user = []
        for n in graph.neighbors(t):
            if graph.node[n]['types'] == 'user':
                user.append(n)
        if len(user) > 1:
            sameTag += len(user)
        for u in user:
            for v in user:
                if not v == u:
                    if graph.edge[u][t][0]['belongs'] == 'train':
                        if graph.edge[v][t][0]['belongs'] == 'train':
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[t][v][0]['y']-graph.edge[t][u][0]['y'],2)
                        else: # test
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[t][v][0]['y_p']-graph.edge[t][u][0]['y'],2)
                    else: #test
                        if graph.edge[v][t][0]['belongs'] == 'train':
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[t][v][0]['y']-graph.edge[t][u][0]['y_p'],2)
                        else:
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[t][v][0]['y_p']-graph.edge[t][u][0]['y_p'],2)
    print 'number of users in same publication =',samePub
    print P_Y_X
    #simulated annealing
    print 'simulated annealing :)'
    iteration = 0
    while temperature > absoluteTemperature:
        for i in range(sample_iter):
            for t in test_edges:
                P = [0]*m_value
                if graph.node[t[0]]['types'] == 'tag':
                    tag = t[0]
                    user = t[1]
                elif graph.node[t[1]]['types'] == 'tag':
                    tag = t[1]
                    user = t[0]
                else:
                    continue
                z = 0
                oldLikelihood = P_Y_X
                for v in graph.neighbors(tag):
                    if graph.node[v]['types'] == 'user' and (not v == user):
                        if graph.edge[v][tag][0]['belongs'] == 'train':
                            P_Y_X = P_Y_X + 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y']-graph.edge[tag][user][0]['y_p'],2)
                        else: # test
                            P_Y_X = P_Y_X + 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-graph.edge[tag][user][0]['y_p'],2)
                for l in range(m_value):
                    P[l] = P_Y_X
                for v in graph.neighbors(tag):
                    if graph.node[v]['types'] == 'user' and (not v == user):
                        if graph.edge[v][tag][0]['belongs'] == 'train':
                            for l in range(m_value):
                                P[l] = P[l] - 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y']-l,2)
                        else: # test
                            for l in range(m_value):
                                P[l] = P[l] - 0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-l,2)
                Sum = 0
                for l in range(m_value):
                    P[l] = math.exp(P[l])
                    Sum += P[l]
                for l in range(m_value):
                    if not Sum == 0:
                        P[l] = P[l]/Sum
                Sum = 0
                for l in range(m_value):
                    P[l] = math.pow(P[l],1.0/temperature)
                    Sum += P[l]
                for l in range(m_value):
                    if Sum == 0:
                        P[random.randint(0,m_value-1)] = 1.0
                    else:
                        P[l] = P[l]/Sum
                distrib = rv_discrete( values=(np.arange(m_value), P) )
                pre = distrib.rvs(size=1) # sample a value from ditribution

                #calculate delta F
                Likelihood = P_Y_X
                for v in graph.neighbors(pub):
                    if graph.node[v]['types'] == 'user' and (not v == user):
                        if graph.edge[v][pub][0]['belongs'] == 'train':
                            Likelihood = Likelihood - 0.5*exp_lda*math.pow(graph.edge[pub][v][0]['y']-pre[0],2)
                        else:
                            Likelihood = Likelihood - 0.5*exp_lda*math.pow(graph.edge[pub][v][0]['y_p']-pre[0],2)
                F = Likelihood - oldLikelihood
                
                if min(1.0,math.exp(F/temperature)) > random.uniform(0.0,1.0):
                    for edge in graph.edge[tag][user]:
                        graph.edge[tag][user][edge]['y_p'] = pre[0]

                #update P(Y|X)
                for v in graph.neighbors(tag):
                    if graph.node[v]['types'] == 'user' and (not v == user):
                        if graph.edge[v][tag][0]['belongs'] == 'train':
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y']-graph.edge[tag][user][0]['y_p'],2)
                        else: # test
                            P_Y_X += -0.5*exp_lda*math.pow(graph.edge[tag][v][0]['y_p']-graph.edge[tag][user][0]['y_p'],2)


        temperature *= coolingRate
        iteration += 1
        print iteration, temperature
        RMSE(graph,test_edges)
    return

if __name__ == '__main__':
   
    # read data
    graph = nx.MultiGraph()
    dirPath = '../../Data/small_data_1000/'
    f = open(dirPath+'user-tag/out.bibsonomy-2ut','r')
    for line in f:
        l = line.strip('\n').split(' ')
        graph.add_node('user'+l[0], types = 'user')
        graph.add_node('tag'+l[1], types = 'tag')
        graph.add_edge('user'+l[0], 'tag'+l[1], weight=l[2], time=l[3], belongs='train')# belong = train or test
    f.close();

    f = open(dirPath+'user-publication/out.bibsonomy-2ui','r')
    for line in f:
        l = line.strip('\n').split(' ')
        graph.add_node('user'+l[0], types = 'user')
        graph.add_node('publication'+l[1], types = 'publication')
        graph.add_edge('user'+l[0], 'publication'+l[1], weight=l[2], time=l[3], belongs='train')
    f.close();

    f = open(dirPath+'tag-publication/out.bibsonomy-2ti','r')
    for line in f:
        l = line.strip('\n').split(' ')
        graph.add_node('tag'+l[0], types = 'tag')
        graph.add_node('publication'+l[1], types = 'publication')
        graph.add_edge('tag'+l[0], 'publication'+l[1], weight=l[2], time=l[3])
    f.close();

    update_iter = 10
    sample_iter = 10
    print 'num of nodes =',len(graph.nodes())
    print 'num of edges =',len(graph.edges())
    Statistic()
    print 'split data..'
    SplitData(graph)
    print 'begin to learning'
    lda = Learning(graph, update_iter, sample_iter)
    print 'inference testing data'
    #Inference(graph, sample_iter, lda)


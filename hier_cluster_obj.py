import numpy as np
from numpy import save,load
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['font.size'] = 17
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from scipy.interpolate import interp1d
import scipy.sparse as sps
from scipy.sparse import linalg as sps_linalg
import scipy.linalg as scipy_linalg
from importlib import reload
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import PCA
import queue
import sys
import os

class HierCluster:
    def __init__(self,memberlist,address,is_leaf=True):
        self.memberlist = memberlist
        self.n_members = len(self.memberlist)
        self.address = address
        self.is_leaf = is_leaf
        return
    def __eq__(self,other):
        return (self.is_leaf == other.is_leaf) and (self.n_members == other.n_members)
    def __ne__(self,other):
        return not self.__eq__(other)
    def __gt__(self,other):
        if self.is_leaf != other.is_leaf:
            return other.is_leaf
        return self.n_members < other.n_members
    def __lt__(self,other):
        if self.is_leaf != other.is_leaf:
            return self.is_leaf
        return self.n_members > other.n_members
    def __ge__(self,other):
        return (self.is_leaf == other.is_leaf) and (self.n_members <= other.n_members)
    def __le__(self,other):
        return (self.is_leaf == other.is_leaf) and (self.n_members >= other.n_members)
    def subcluster(self,x,nsubclust,min_clust_size):
        stop = False
        print("nsubclust = ",end="")
        while not stop:
            print("{}, ".format(nsubclust),end="")
            if len(self.memberlist) < nsubclust or nsubclust == 0:
                sys.exit("ERROR in hier_cluster: len(memberlist) < nsubclust")
            self.km = MiniBatchKMeans(n_clusters=nsubclust).fit(x[self.memberlist])
            counts = np.zeros(self.km.n_clusters)
            for i in range(self.km.n_clusters):
                counts[i] = np.sum(self.km.labels_ == i)
            stop = (np.min(counts) >= min_clust_size) # was 6, then 3, 
            if not stop:
                #sys.exit("MiniBatchKMeans has empty clusters")
                nsubclust = nsubclust//2
            self.is_leaf = False
            self.children = []
        print("")
        return

def nested_kmeans(X,K,mcpl=2,min_clust_size=10):
    N,d = X.shape
    hc0 = HierCluster(np.arange(N),[])
    hc0.subcluster(X,min(K,mcpl),min_clust_size)
    q = queue.PriorityQueue()
    x_addresses = [[] for i in range(N)]
    for i in range(hc0.km.n_clusters):
        memberlist = np.where(hc0.km.labels_ == i)[0]
        for j in memberlist:
            x_addresses[j] = [i,]
        hc0.children += [HierCluster(memberlist,[i,]),]
        q.put(hc0.children[-1])
    total = hc0.km.n_clusters
    while total < K:
        clust = q.get()
        #print("Is new clust a leaf? {}".format(clust.is_leaf))
        num_new_clust = min([K-total+1,mcpl,clust.n_members])
        if num_new_clust == 0:
            print("Zero new clusters prescribed")
        clust.subcluster(X,min([K-total+1,mcpl,clust.n_members]),min_clust_size)
        for i in range(clust.km.n_clusters):
            submemberlist = np.where(clust.km.labels_ == i)[0]
            if len(submemberlist) == 0:
                print("Submemberlist is empty. num_new_clust={}".format(num_new_clust))
            subaddress = [i,]
            clust.children += [HierCluster(clust.memberlist[submemberlist],clust.address+subaddress),]
            q.put(clust.children[-1])
            for j in clust.memberlist[submemberlist]:
                x_addresses[j] += subaddress
        q.put(clust) # After it has been downgraded
        total += clust.km.n_clusters-1
        #print("n_clust = {}; total = {}; K = {}".format(clust.km.n_clusters,total,K))
    # Count leaves
    num_leaves = 0
    for i in range(len(q.queue)):
        num_leaves += (q.queue[i].is_leaf)
    print("num_leaves = {}; K = {}".format(num_leaves,K))
    # Get flat addresses and cluster centers
    k = 0
    centers = np.zeros([K,d])
    for i in range(len(q.queue)):
        if q.queue[i].is_leaf:
            q.queue[i].global_address = k
            if len(q.queue[i].memberlist) == 0:
                print("PROBLEM! Empty memberlist")
            #print("X.shape = {}".format(X.shape))
            #print("memberlist in ({},{})".format(np.min(q.queue[i].memberlist),np.max(q.queue[i].memberlist)))
            #print("centers.shape = {}".format(centers.shape))
            #print("k = {}; K = {}".format(k,K))
            centers[k] = np.mean(X[q.queue[i].memberlist], 0)
            k += 1    
    return q,x_addresses,hc0,centers

def nested_kmeans_predict(Y,hc0):
    hc = hc0
    address = []
    while not hc.is_leaf:
        address += [hc.km.predict(np.array([Y]))[0],]
        hc = hc.children[address[-1]]
    global_address = hc.global_address
    return address,global_address

def nested_kmeans_predict_batch(Y,hc):
    # Put a lot of data points at once into a cluster
    # This is a recursive function
    N = len(Y)
    global_addresses = np.zeros(N,dtype=int)
    addresses = []
    for i in range(N):
        addresses += [hc.address.copy(),]
    if not hc.is_leaf:
        assn = hc.km.predict(Y)
        # print("assn = {}".format(assn))
        for j in range(hc.km.n_clusters):
            idx = np.where(assn==j)[0]
            if len(idx) > 0:
                subaddresses,sub_global_addresses = nested_kmeans_predict_batch(Y[idx],hc.children[j])
                for i in range(len(idx)):
                    addresses[idx[i]] = subaddresses[i].copy()
                    global_addresses[idx[i]] = sub_global_addresses[i]
                    
    else:
        for i in range(N):
            global_addresses[i] = hc.global_address
    return addresses,global_addresses

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random
import os
import tensorflow as tf
import numpy as np
import copy
import math
import operator
random.seed(0)
np.random.seed(0)
flags = tf.app.flags

flags.DEFINE_string("data_dir", 'Datasets/Linkedin', "data directory.")
flags.DEFINE_integer("max_epoch", 5000, "max number of epochs.")
flags.DEFINE_integer("emb_dim", 128, "embedding dimension.")
flags.DEFINE_integer("batch_size_first", 2000, "batch size of edges in 1st")
flags.DEFINE_integer("batch_size_second", 500, "batch size of edges in 2nd") # this is the # positive examples per batch
flags.DEFINE_integer("order", 2, "proximity to use") # if 3, use both.


flags.DEFINE_integer("disp_freq", 100, "frequency to output.")
flags.DEFINE_integer("save_freq", 10000, "frequency to save.")
flags.DEFINE_float("lr", 0.01, "initial learning rate.")
flags.DEFINE_float("number_neg_samples", 5, "# of negative samples per positive example")
flags.DEFINE_boolean("reload_model", 0, "whether to reuse saved model.") # Note : this is for saved model
flags.DEFINE_boolean("train", 1, "whether to train model.")

FLAGS = flags.FLAGS

class Options(object):
    """options used by LINE model."""
    def __init__(self):
        # model options.
        self.emb_dim = FLAGS.emb_dim
        self.order = FLAGS.order
        self.batch_size_first = FLAGS.batch_size_first
        self.batch_size_second = FLAGS.batch_size_second
        self.network_file = os.path.join(FLAGS.data_dir, 'network_edgelist.txt')
        self.save_path = os.path.join(FLAGS.data_dir, 'line.ckpt')
        self.max_epoch = FLAGS.max_epoch
        self.lr = FLAGS.lr
        self.number_neg_samples = FLAGS.number_neg_samples
        self.disp_freq = FLAGS.disp_freq
        self.save_freq = FLAGS.save_freq
        self.reload_model = FLAGS.reload_model

class LINE(object):
    """LINE Embedding model."""
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._u2idx = {}
        self._idx2u = []
        self._network = self._readFromFile(options.network_file)
        self._options.num_nodes = len(self._u2idx)
        self._Pn = self._getDegreeDist() # This is actually d_v ^(0.75)
        self.buildGraph()
        self.buildGraphSecond()
        if options.reload_model:
            self.saver.restore(session, options.save_path)

    def _readFromFile(self, filename):
        edges = []
        nodeCounter = 0
        adj = {}
        for line in open(filename):
            a = int(line.split()[0])
            b = int(line.split()[1])
            if a not in self._u2idx:
                self._u2idx[a] = nodeCounter
                self._idx2u.append(a)
                nodeCounter += 1
            if b not in self._u2idx:
                self._u2idx[b] = nodeCounter
                self._idx2u.append(b)
                nodeCounter += 1
            a = self._u2idx[a]
            b = self._u2idx[b]
            if a not in adj:
                adj[a] = []
            if b not in adj:
                adj[b] = []
            adj[a].append(b)
            adj[b].append(a)
            if a<b :
                edges.append((a,b))
            else:
                edges.append((b,a))
        print ("edges = ", len(edges))
        #node_Degree = {}
        #for i in range(0, len(self._u2idx)):
        #    node_Degree[i] = len(adj[i])
        #deg_sorted = sorted(node_Degree.items(), key = operator.itemgetter(1), reverse = True)
        #print (deg_sorted[0:int(0.2* len(self._u2idx))])
        for i in range(0, len(self._u2idx)):
            if len(adj[i]) < 10:
                for j in adj[i]:
                    for k in adj[j]:
                        if k < i :
                            edges.append((i,k))
                        else:
                            edges.append((k,i))
        edges = list(set(edges))
        print("edges after densify = ", len(edges))
        return edges

    def _getDegreeDist(self):
        opts = self._options
        degree_dist = [0.0]* opts.num_nodes
        for e in self._network:
            degree_dist[e[0]] += 1
            degree_dist[e[1]] += 1
        for i in range(0, opts.num_nodes):
            degree_dist[i] = math.pow(degree_dist[i], 0.75)
        degree_dist = [i/sum(degree_dist) for i in degree_dist]
        return degree_dist   

    def buildGraph(self):
        opts = self._options
        U_batch = tf.placeholder(tf.int32,[opts.batch_size_first])
        V_batch = tf.placeholder(tf.int32,[opts.batch_size_first])
        embedding = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding")
        embeddings_U = tf.nn.embedding_lookup(embedding, U_batch)
        embeddings_V = tf.nn.embedding_lookup(embedding, V_batch)
        X = tf.diag_part(tf.matmul(embeddings_U, tf.transpose(embeddings_V)))
        loss = -1*tf.reduce_sum(tf.log(tf.nn.sigmoid(X)))
        #optimizer = tf.train.GradientDescentOptimizer(opts.lr).minimize(loss)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        optimizer = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads, tvars))
        self.embedding = embedding
        self.loss = loss
        self.optimizer = optimizer
        self.U_batch = U_batch
        self.V_batch = V_batch
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def buildGraphSecond(self):
        opts = self._options
        U_batch_second = tf.placeholder(tf.int32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        V_batch_second = tf.placeholder(tf.int32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        label_batch = tf.placeholder(tf.float32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        embedding_target = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding_target")
        embedding_context = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding_context")
        embeddings_U = tf.nn.embedding_lookup(embedding_target, U_batch_second)
        embeddings_V = tf.nn.embedding_lookup(embedding_context, V_batch_second)
        X = tf.multiply(label_batch, tf.diag_part(tf.matmul(embeddings_U, tf.transpose(embeddings_V))))
        loss_second = -1*tf.reduce_sum(tf.log(tf.nn.sigmoid(X)))
        #optimizer = tf.train.GradientDescentOptimizer(opts.lr).minimize(loss)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss_second, tvars)
        optimizer_second = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads, tvars))
        self.embedding_context = embedding_context
        self.embedding_target = embedding_target
        self.U_batch_second = U_batch_second
        self.V_batch_second = V_batch_second
        self.label_batch = label_batch
        self.optimizer_second = optimizer_second
        self.loss_second = loss_second
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def createBatchSecondOrder(self):
        opts = self._options
        edges_list = np.random.permutation(len(self._network))[0:opts.batch_size_second]
        batch_edges = map(lambda x: self._network[x], edges_list)
        labels = [1] * len(batch_edges)
        batch_negative_edges = []
        for e in batch_edges:
            a = e[0]
            b = e[1]
            negative_nodes = np.random.choice(range(0, opts.num_nodes), opts.number_neg_samples, p=self._Pn)
            for x in negative_nodes:
                if random.random() < 0.5:
                    batch_negative_edges.append((a,x))
                else:
                    batch_negative_edges.append((b,x))
        labels.extend([-1]* len(batch_negative_edges))
        batch_edges.extend(batch_negative_edges)
        return (batch_edges , labels)

    def trainSecond(self, batches_2nd_order):
        opts = self._options
        n_iter = 0        
        print ("Starting training")
        for epoch in xrange(opts.max_epoch):
            #print ("Creating batch")
            (batch_edges , labels) = batches_2nd_order[epoch]
            #(batch_edges , labels) = self.createBatchSecondOrder()
            #print ("Created batch")
            train_U = map(lambda x : x[0],batch_edges)
            train_V = map(lambda x : x[1],batch_edges)
            self._session.run(self.optimizer_second, feed_dict={self.U_batch_second: train_U, self.V_batch_second: train_V, self.label_batch: labels})
            n_iter += 1
            if n_iter % opts.disp_freq == 0:
                loss_second = self._session.run(self.loss_second, feed_dict={self.U_batch_second: train_U, self.V_batch_second: train_V, self.label_batch:labels})
                print('step %d, loss=%f' % (n_iter, loss_second))
            if n_iter % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)
        current_embedddings = self._session.run(self.embedding_target)
        return current_embedddings

    def trainFirst(self):
        opts = self._options
        n_iter = 0
        print ("Starting training")
        for _ in xrange(opts.max_epoch):
            edges_list = np.random.permutation(len(self._network))[0:opts.batch_size_first]
            batch_edges = map(lambda x: self._network[x], edges_list)
            #batch_edges = self._network[0:opts.batch_size_first]
            train_U = map(lambda x : x[0],batch_edges)
            train_V = map(lambda x : x[1],batch_edges)
            self._session.run(self.optimizer, feed_dict={self.U_batch: train_U, self.V_batch: train_V})
            n_iter += 1
            if n_iter % opts.disp_freq == 0:
                loss = self._session.run(self.loss, feed_dict={self.U_batch: train_U, self.V_batch: train_V})
                print('step %d, loss=%f' % (n_iter, loss))
            if n_iter % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)
        current_embedddings = self._session.run(self.embedding)
        return current_embedddings
        #print (current_embedddings.shape)
    def readBatches(self):
        batches_2nd_order = {}
        f = open('Batches_2nd.txt','r')
        i = 0
        ctr = 0
        batch_edges = []
        batch_labels = []
        for line in f:
            line = line.split("\n")[0]
            a = int(line.split(",")[0])
            b = int(line.split(",")[1])
            a = self._u2idx[a]
            b = self._u2idx[b]
            label = int(line.split(",")[2])
            i += 1
            batch_edges.append((a,b))
            batch_labels.append(label)
            if i % (self._options.batch_size_second*(1+self._options.number_neg_samples)) == 0:
                batches_2nd_order[ctr] = (batch_edges, batch_labels)
                batch_edges = []
                batch_labels = []
                print (len(batches_2nd_order), len(batches_2nd_order[ctr][0]))
                ctr += 1
            if ctr == self._options.max_epoch:
                break
        f.close()
        return batches_2nd_order
        

def main(_):
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = LINE(options, session)
        if FLAGS.train:
            if FLAGS.order == 1:
                emb_first = model.trainFirst()
                emb = emb_first
            elif FLAGS.order == 2:
                batches_2nd_order = []
                batches_2nd_order = model.readBatches()
                emb_second = model.trainSecond(batches_2nd_order)
                emb = emb_second
            else:
                emb_first = model.trainFirst()
                batches_2nd_order = []
                batches_2nd_order = model.readBatches()
                emb_second = model.trainSecond(batches_2nd_order)
                emb = np.concatenate((emb_first, emb_second), axis=1)

            f = open('linkedin.LINE.emb.txt', 'w')
            for i in range(0, options.num_nodes):
                f.write(str(model._idx2u[i])+" ")
                for j in range(0, len(emb[i])):
                    f.write(str(emb[i][j])+" ")
                f.write("\n")
            f.close()

if __name__ == "__main__":
    tf.app.run()

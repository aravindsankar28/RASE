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
import json
import pickle
random.seed(0)
np.random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("data_dir", 'Datasets/Linkedin', "data directory.")
flags.DEFINE_integer("max_epoch", 5000, "max number of epochs.")
flags.DEFINE_integer("emb_dim", 128, "embedding dimension.")
flags.DEFINE_integer("batch_size_first", 1000, "batch size of edges in 1st")
flags.DEFINE_integer("batch_size_second", 100, "batch size of edges in 2nd") # this is the # positive examples per batch
flags.DEFINE_integer("relation_matrix_dim", 128, "relation matrix dimension")

flags.DEFINE_integer("order", 1, "frequency to output.")
flags.DEFINE_integer("disp_freq", 100, "frequency to output.")
flags.DEFINE_integer("save_freq", 10000, "frequency to save.")
flags.DEFINE_float("lr", 0.005, "initial learning rate.")
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
        self.relation_matrix_dim = FLAGS.relation_matrix_dim
        self.network_file = os.path.join(FLAGS.data_dir, 'network.txt')
        self.save_path = os.path.join(FLAGS.data_dir, 'line.ckpt')
        self.max_epoch = FLAGS.max_epoch
        self.lr = FLAGS.lr
        self.number_neg_samples = FLAGS.number_neg_samples
        self.disp_freq = FLAGS.disp_freq
        self.save_freq = FLAGS.save_freq
        self.reload_model = FLAGS.reload_model

class RLINE(object):
    """LINE Embedding model."""
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._u2idx = {}
        self._idx2u = []
        self._rtoidx = {}
        self._idx2r = []
        self._network, self._relations = self._readFromFile(options.network_file)
        self._options.num_nodes = len(self._u2idx)
        self._options.num_relations = len(self._rtoidx)
        self._Pn = self._getDegreeDist() # This is actually d_v ^(0.75)
        self._line_embs = self._readInitialEmb("linkedin.deepwalk.emb.txt").astype(np.float32)
        self.inital_R_matrices = pickle.load( open( "R_matrices.p", "rb" ) ).astype(np.float32)
        self.buildGraph()
        self.buildGraphSecond()

        if options.reload_model:
            self.saver.restore(session, options.save_path)

    def _readInitialEmb(self, filename):
        user_embeddings = {}
        initialEmbs = []
        with open(filename) as f:
            lines = f.read().splitlines()
            for l in lines:
                split_line = l.split()
                user = int(split_line[0])
                idx = self._u2idx[user]
                emb = map(lambda x : float(x) , split_line[1:len(split_line)])
                user_embeddings[idx] = emb
        for i in range(0, self._options.num_nodes):
            initialEmbs.append(user_embeddings[i])
        return np.array(initialEmbs)

    def _readFromFile(self, filename):
        edges = []
        nodeCounter=0
        network_dict = json.load(open(filename))
        relation_set = set()
        for line in network_dict:
            a = int(line.split()[0])
            b = int(line.split()[1])
            relations = network_dict[line]
            relation_set.update(set(relations))
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
            if a<b :
                edges.append((a,b))
            else:
                edges.append((b,a))

        relationCounter =0 
        for r in relation_set:
            self._rtoidx[r] = relationCounter
            self._idx2r.append(r)
            relationCounter += 1

        edges = list(set(edges))
        relations = []

        for e in edges:
            edgeStr = str(self._idx2u[e[0]])+" "+str(self._idx2u[e[1]])
            if edgeStr not in network_dict:
                edgeStr = str(self._idx2u[e[1]])+" "+str(self._idx2u[e[0]])
            relation_list = map(lambda x: self._rtoidx[x], network_dict[edgeStr])
            relations.append(relation_list)
        return (edges, relations)

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
        relation_labels = tf.placeholder(tf.int32, [opts.batch_size_first]) # relation labels
        embedding = tf.Variable(self._line_embs, name = "embedding")
        #embedding = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding")
        R_matrices = tf.Variable(self.inital_R_matrices, name = "R_matrices")
        #R_matrices = tf.Variable(tf.random_uniform([opts.num_relations, opts.relation_matrix_dim, opts.emb_dim], -0.1, 0.1), name="r_matrices")
        embeddings_U = tf.nn.embedding_lookup(embedding, U_batch)
        embeddings_V = tf.nn.embedding_lookup(embedding, V_batch)
        R_matrix = tf.nn.embedding_lookup(R_matrices, relation_labels)
        embeddings_U_projected = tf.matmul(R_matrix, tf.expand_dims(embeddings_U,2))
        embeddings_V_projected = tf.matmul(R_matrix, tf.expand_dims(embeddings_V,2))
        
        embeddings_U_projected = tf.reduce_sum(embeddings_U_projected, 2)
        embeddings_V_projected = tf.reduce_sum(embeddings_V_projected, 2)
        #print (embeddings_U.shape, embeddings_U_projected.shape)
        X = tf.diag_part(tf.matmul(embeddings_U_projected, tf.transpose(embeddings_V_projected)))
        loss = -1*tf.reduce_sum(tf.log(tf.nn.sigmoid(X)))
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        optimizer = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads, tvars))

        X_unlabled = tf.diag_part(tf.matmul(embeddings_U, tf.transpose(embeddings_V)))
        loss_unlabeled = -1*tf.reduce_sum(tf.log(tf.nn.sigmoid(X_unlabled)))
        optimizer_unlabeled = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(tf.gradients(loss_unlabeled, tf.trainable_variables()), tf.trainable_variables()))        

        self.R_matrices = R_matrices
        self.embedding = embedding
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_unlabeled = optimizer_unlabeled
        self.loss_unlabeled = loss_unlabeled
        self.relation_labels = relation_labels
        self.U_batch = U_batch
        self.V_batch = V_batch
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def buildGraphSecond(self):
        opts = self._options
        U_batch_second = tf.placeholder(tf.int32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        V_batch_second = tf.placeholder(tf.int32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        label_batch = tf.placeholder(tf.float32,[opts.batch_size_second*(1+opts.number_neg_samples)])
        relation_labels_second = tf.placeholder(tf.int32,[opts.batch_size_second*(1+opts.number_neg_samples)])

        embedding_target = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding_target")
        embedding_context = tf.Variable(tf.random_uniform([opts.num_nodes, opts.emb_dim], -0.1, 0.1), name="embedding_context")
        R_matrices = tf.Variable(tf.random_uniform([opts.num_relations, opts.relation_matrix_dim, opts.emb_dim], -0.1, 0.1), name="r_matrices")
        embeddings_U = tf.nn.embedding_lookup(embedding_target, U_batch_second)
        embeddings_V = tf.nn.embedding_lookup(embedding_context, V_batch_second)
        R_matrix = tf.nn.embedding_lookup(R_matrices, relation_labels_second)
        
        embeddings_U_projected = tf.matmul(R_matrix, tf.expand_dims(embeddings_U,2))
        embeddings_V_projected = tf.matmul(R_matrix, tf.expand_dims(embeddings_V,2))
        embeddings_U_projected = tf.reduce_sum(embeddings_U_projected, 2)
        embeddings_V_projected = tf.reduce_sum(embeddings_V_projected, 2)
        #print (R_matrix.shape)
        #print (tf.expand_dims(embeddings_U,2).shape)
        #print (embeddings_U_projected.shape)
        #print (embeddings_U.shape, embeddings_U_projected.shape)
        X = tf.multiply(label_batch, tf.diag_part(tf.matmul(embeddings_U_projected, tf.transpose(embeddings_V_projected))))
        loss_second = -1*tf.reduce_sum(tf.log(tf.nn.sigmoid(X)))
        #optimizer = tf.train.GradientDescentOptimizer(opts.lr).minimize(loss)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss_second, tvars)
        optimizer_second = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads, tvars))
        self.embedding_context = embedding_context
        self.embedding_target = embedding_target
        self.U_batch_second = U_batch_second
        self.V_batch_second = V_batch_second
        self.relation_labels_second = relation_labels_second
        self.label_batch = label_batch
        self.optimizer_second = optimizer_second
        self.loss_second = loss_second
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def trainSecond(self):
        opts = self._options
        n_iter = 0    
        (batch_edges , labels, rel_labels) = self.createBatchLabeledSecond()
        print (len(batch_edges))
        print ("Starting training")
        for _ in xrange(opts.max_epoch):
            train_U = map(lambda x : x[0],batch_edges)
            train_V = map(lambda x : x[1],batch_edges)
            self._session.run(self.optimizer_second, feed_dict={self.U_batch_second: train_U, self.V_batch_second: train_V, self.label_batch: labels, self.relation_labels: rel_labels})
            n_iter += 1
            if n_iter % opts.disp_freq == 0:
                loss_second = self._session.run(self.loss_second, feed_dict={self.U_batch_second: train_U, self.V_batch_second: train_V, self.label_batch:labels, self.relation_labels:rel_labels})
                print('step %d, loss=%f' % (n_iter, loss_second))
            if n_iter % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)    

    def createBatchUnlabeledSecond(self):
        opts = self._options
        edges_list = []
        for i in range(0, len(self._network)):
            if len(self._relations[i]) == 0:
                edges_list.append(self._network[i])
        random.shuffle(edges_list)
        batch_edges = edges_list[0:opts.batch_size_second]
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

    def createBatchLabeledSecond(self):
        # TOOD : incomplete
        opts = self._options
        indices = []
        for i in range(0, len(self._network)):
            if len(self._relations[i]) > 0:
                indices.append(i)
        random.shuffle(indices)
        indices = indices[0:opts.batch_size_second]
        batch_edges = map(lambda x: self._network[x], indices)
        rel_labels = map(lambda x: random.choice(self._relations[x]), indices)
        labels = [1]*len(batch_edges)
        batch_negative_edges = []
        for i in range(0, len(batch_edges)):
            e = batch_edges[i]
            rel_label = rel_labels[i]
            a = e[0]
            b = e[1]
            negative_nodes = np.random.choice(range(0, opts.num_nodes), opts.number_neg_samples, p=self._Pn)
            for x in negative_nodes:
                batch_negative_edges.append((a,x))
                rel_labels.append(rel_label)
        labels.extend([-1]* len(batch_negative_edges))

        batch_edges.extend(batch_negative_edges)
        return (batch_edges , labels, rel_labels)

    def createBatchUnlabeledFirst(self):
        opts = self._options
        edges_list = []
        for i in range(0, len(self._network)):
            if len(self._relations[i]) == 0:
                edges_list.append(self._network[i])
        random.shuffle(edges_list)
        batch_edges = edges_list[0:opts.batch_size_first]
        return batch_edges

    # return edges and relation labels.
    def createBatchLabeledFirst(self):
        opts = self._options
        indices = []
        for i in range(0, len(self._network)):
            if len(self._relations[i]) > 0:
                indices.append(i)
        random.shuffle(indices)
        indices = indices[0:opts.batch_size_first]
        batch_edges = map(lambda x: self._network[x], indices)
        labels = map(lambda x: random.choice(self._relations[x]), indices)
        return (batch_edges, labels)

    def trainFirst(self):
        opts = self._options
        n_iter = 0
        print ("Starting training")
        rel_labels= np.array([1]* opts.batch_size_first) # TODO : need to fix - dummy for now.
        for _ in xrange(opts.max_epoch):
            if random.random() < 0.5:
                (batch_edges, rel_labels) = self.createBatchLabeledFirst()
                train_U_labeled = map(lambda x : x[0],batch_edges)
                train_V_labeled = map(lambda x : x[1],batch_edges)
                self._session.run(self.optimizer, feed_dict={self.U_batch: train_U_labeled, self.V_batch: train_V_labeled, self.relation_labels: rel_labels})
            else:
                batch_edges = self.createBatchUnlabeledFirst()
                train_U_unlabeled = map(lambda x : x[0],batch_edges)
                train_V_unlabeled = map(lambda x : x[1],batch_edges)
                self._session.run(self.optimizer_unlabeled, feed_dict={self.U_batch: train_U_unlabeled, self.V_batch: train_V_unlabeled})           
            n_iter += 1
            if n_iter % opts.disp_freq == 0:
                loss_unlabeled = self._session.run(self.loss_unlabeled, feed_dict={self.U_batch: train_U_unlabeled, self.V_batch: train_V_unlabeled})
                loss_labeled = self._session.run(self.loss, feed_dict={self.U_batch: train_U_labeled, self.V_batch: train_V_labeled, self.relation_labels: rel_labels})                
                print('step %d, loss_label =%f  loss_unlabeled=%f' % (n_iter, loss_labeled, loss_unlabeled))
            if n_iter % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)
        current_embedddings = self._session.run(self.embedding)
        R_matrices = self._session.run(self.R_matrices)
        #r = current_embedddings
        #r = np.matmul(current_embedddings, np.transpose(R_matrices[0]))
        # print (r.shape)
        projections = []
        for i in range(0, opts.num_relations):
            emb_r = np.matmul(np.transpose(current_embedddings), R_matrices[i])
            projections.append(emb_r)
            #print (emb_r.shape)
            #r = np.concatenate((r, emb_r), axis=1)
            #print(emb_r.shape, r.shape)
        #print(r.shape)
        return (current_embedddings, projections)
        #return r

def main(_):
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = RLINE(options, session)
        if FLAGS.train:
            if FLAGS.order == 1:
                (emb_first, projections) = model.trainFirst()
                emb = emb_first
            elif FLAGS.order == 2:
                batches_2nd_order = []
                #batches_2nd_order = model.readBatches()
                emb_second = model.trainSecond(batches_2nd_order)
                emb = emb_second
            else:
                emb_first = model.trainFirst()
                batches_2nd_order = []
                #batches_2nd_order = model.readBatches()
                emb_second = model.trainSecond(batches_2nd_order)
                emb = np.concatenate((emb_first, emb_second), axis=1)

            f = open('linkedin.RLINE.emb.txt', 'w')
            for i in range(0, options.num_nodes):
                f.write(str(model._idx2u[i])+" ")
                for j in range(0, len(emb[i])):
                    f.write(str(emb[i][j])+" ")
                f.write("\n")
            f.close()

            for r in range(0, options.num_relations):
                f = open('linkedin.RLINE.'+model._idx2r[r]+'.emb.txt', 'w')
                for i in range(0, options.num_nodes):
                    f.write(str(model._idx2u[i])+" ")
                    for j in range(0, len(emb[i])):
                        f.write(str(projections[r][i][j])+" ")
                    f.write("\n")
                f.close()


if __name__ == "__main__":
    tf.app.run()

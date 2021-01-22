### PROGRAM TO RUN A MODEL FOR 10 DIFFERENT RANDOM SEEDS SELECTED AT RANDOM

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN_2_layer, GCNDense, GCN_1_layer, GCN_3_layer, GCNDenseDense 

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.8   , 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3 , 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('s', 1, 'Hyper-parameter s for diffusion and GraphHeat filters')
flags.DEFINE_float('a', 1.00, 'Value of hyper-parameter a for p-step RW filter')
flags.DEFINE_integer('p', 3 , 'Value of hyper-parameter a for p-step RW filter.')
arch = GCN_2_layer

#Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
features = preprocess_features(features)
Accuracy=[]
Time=[]
seeds=[363, 542, 183, 342, 852, 965, 537, 888, 750, 140]
count=0
for seedstf in seeds:    
    count+=1
    np.random.seed(seedstf)
    tf.set_random_seed(seedstf)
    
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = arch
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = arch
    elif FLAGS.model == 'cosine':
        angle=np.pi/4  
        support= cosine_filter_single_param(adj, 2*FLAGS.max_degree, angle)
        num_supports = 1 
        model_func = arch
    elif FLAGS.model == 'psteprandomwalk':
        support = psteprandomwalk(adj, FLAGS.a,  FLAGS.p)
        num_supports = 1 
        model_func = arch
    elif FLAGS.model == 'graphheat':
        support = graphheat(adj, FLAGS.max_degree,  FLAGS.s)
        num_supports = 2
        model_func = arch
    elif FLAGS.model == 'diffusion':
        support = diffusion_SP_with_param(adj, FLAGS.max_degree,  FLAGS.s)
        num_supports = 1 
        model_func = arch
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
    sess = tf.Session()


# Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

# Train model
    sumtime=0
    for epoch in range(FLAGS.epochs):

        t = time.time()
    # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        sumtime=sumtime + (time.time() - t)

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
             
            break
    Time.append(sumtime/epoch+1)
# Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results for iteration:", "{:.0f}".format(count), "- cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    Accuracy.append(test_acc)

print("Final result:", "accuracy=", "{:.5f}".format(np.mean(Accuracy)), 
            "+-", "{:.5f}".format(np.std(Accuracy)), "time=", "{:.5f}".format(np.mean(Time)))

import tensorflow as tf
import os
import numpy as np

from module.misc import scatter_nd_add_diff, v_check_control
from module.load import parse_file


class GNS(object):
    """
    """

    def __init__(self,
                 case, 
                 train_params,
                 model_params,
                 noise_params,
                 name,
                 sess,
                 model_to_reload=None):

        # Get session and params
        self.sess = sess
        self.train_params = train_params
        self.model_params = model_params

        # Store params into tf variables
        self._params_to_variables()

        # Define the type of non linearity that will be used
        self.non_lin = tf.nn.leaky_relu
        #self.non_lin = tf.tanh

        # Define discount for the loss
        self.discount = 0.5

        # Neural network output normalization. This factor helps the learning process by starting with small
        # reasonable updates at the start of the learning process
        self.neural_net_output_coeff = 1e-3

        # Maximum value for the gradient during learning. Used during the gradient clipping process
        self.max_grad = 1e-3

        self._build_dicts()

        # NOTE : ça doit créer des variables qui seront modifiées par un assign
        self.import_power_grid(case)

        # NOTE : ça ne crée que des tenseurs qui sont engendrés par le case et des variables aléatoires
        self.perturbate_power_grid(noise_params)

        self._build()

        self._summary()
        self.summary_writer = tf.summary.FileWriter(
            name, 
            graph=tf.get_default_graph()
        )

        if model_to_reload is not None:
            self._load(model_to_reload)


    def _load(self, model_to_reload):
        """
        """

        print("Restoring model.")
        self.saver.restore(self.sess, os.path.join(model_to_reload, 'tmp/model.ckpt'))
        print("Model restored.")


    def save(self, path):
        """
        """

        print("Saving model.")
        self.saver.save(self.sess, os.path.join(path, "tmp/model.ckpt"))
        print("Model saved.")


    def _build(self):

        #### Build variables that will usefull later ####

        # Get line characteristics in polar coordinates
        self.y_ij = 1. / tf.sqrt(self.lines['r']**2 + self.lines['x']**2)
        self.delta_ij = tf.math.atan2(self.lines['r'], self.lines['x'])

        # Build indices
        self.linspace = tf.expand_dims(tf.range(0, self.n_samples, 1),-1)
        self.one_tensor = tf.ones([1], tf.int32)
        self.n_lines_tensor = tf.reshape(self.n_lines,[1])
        self.n_gens_tensor = tf.reshape(self.n_gens,[1])
        self.shape_lines_indices = tf.concat([self.one_tensor, self.n_lines_tensor], axis=0)
        self.shape_gens_indices = tf.concat([self.one_tensor, self.n_gens_tensor], axis=0)

        self.indices_from = tf.reshape(tf.tile(self.linspace, self.shape_lines_indices), [-1])
        self.indices_from = tf.stack([self.indices_from, tf.reshape(self.lines['from'], [-1])], 1)

        self.indices_to = tf.reshape(tf.tile(self.linspace, self.shape_lines_indices), [-1])
        self.indices_to = tf.stack([self.indices_to, tf.reshape(self.lines['to'], [-1])], 1)

        self.indices_gens = tf.reshape(tf.tile(self.linspace, self.shape_gens_indices), [-1])
        self.indices_gens = tf.stack([self.indices_gens, tf.reshape(self.gens['bus'], [-1])], 1)

        # Initialize dummy variables that will be useful for the scatter_nd_add_diff function
        self.n_samples_tensor = tf.reshape(self.n_samples,[1])
        self.n_nodes_tensor = tf.reshape(self.n_nodes,[1])
        self.latent_dim_tensor = tf.reshape(self.latent_dim,[1])
        self.shape_latent_message = tf.concat([self.n_samples_tensor,
                                               self.n_nodes_tensor,
                                               self.latent_dim_tensor], axis=0)
        self.zero_input = tf.reshape(0.,[1,1,1])
        self.dummy_message = tf.tile(self.zero_input, self.shape_latent_message)

        self.dummy = tf.zeros_like(self.buses['baseKV'])

        # Build indices matrix for "from" indices
        #self.indices_from = tf.expand_dims(tf.range(0, self.n_samples, 1),-1)
        #temp_from_1 = tf.ones([1], tf.int32)
        #temp_from_2 = tf.reshape(self.n_lines,[1])
        #temp_from_dim = tf.concat([temp_from_1, temp_from_2], axis=0)
        #self.indices_from = tf.reshape(tf.tile(self.indices_from, temp_from_dim), [-1])
        #self.indices_from = tf.stack([self.indices_from, tf.reshape(self.lines['from'], [-1])], 1)

        # Build indices matrix for "to" indices
        #self.indices_to = tf.expand_dims(tf.range(0, self.n_samples, 1),-1)
        #temp_to_1 = tf.ones([1], tf.int32)
        #temp_to_2 = tf.reshape(self.n_lines,[1])
        #temp_to_dim = tf.concat([temp_to_1, temp_to_2], axis=0)
        #self.indices_to = tf.reshape(tf.tile(self.indices_to, temp_to_dim), [-1])
        #self.indices_to = tf.stack([self.indices_to, tf.reshape(self.lines['to'], [-1])], 1)

        # Build indices matrix for "gen" indices
        #self.indices_gen = tf.expand_dims(tf.range(0, self.n_samples, 1),-1)
        #temp_gen_1 = tf.ones([1], tf.int32)
        #temp_gen_2 = tf.reshape(self.n_gens,[1])
        #temp_gen_dim = tf.concat([temp_gen_1, temp_gen_2], axis=0)
        #self.indices_gen = tf.reshape(tf.tile(self.indices_gen,temp_gen_dim), [-1])
        #self.indices_gen = tf.stack([self.indices_gen, tf.reshape(self.gens['bus'], [-1])], 1)

        #################################################

        #### INITIALIZATION ####
        # Initialize v, theta and latent messages
        self.v = {'0': 1.+tf.zeros_like(self.buses['baseKV'])}
        self.theta = {'0': tf.zeros_like(self.buses['baseKV'])}
        self.latent_message = {'0' : tf.zeros_like(self.dummy_message)}

        # Control the voltage for generators
        self.v['0'] = v_check_control(self.v['0'], self.gens['Vg'], self.indices_gens)
        ########################


        # Initialize latent message at zero
        #self.message_dim = self.latent_dim
        #temp1 = tf.reshape(self.n_samples,[1])
        #temp2 = tf.reshape(self.n_nodes,[1])
        #temp3 = tf.reshape(self.message_dim,[1])
        #temp_dim = tf.concat([temp1, temp2, temp3], axis=0)
        #temp_input = tf.reshape(0.,[1,1,1])
        #self.latent_message = {'0' : tf.tile(temp_input, temp_dim)}
        #self.message_neighbors = {'0' : tf.tile(temp_input, temp_dim)} 

        

        

        # First : perform a compensation by neglecting the losses
        self.sum_p_gen_target_per_sample = tf.reduce_sum(self.gens['Pg']/ self.gens['mbase'] , axis=1, keepdims=True)
        self.sum_p_gen_max_per_sample = tf.reduce_sum(self.gens['Pmax']/ self.gens['mbase'], axis=1, keepdims=True)
        self.sum_p_gen_min_per_sample = tf.reduce_sum(self.gens['Pmin']/ self.gens['mbase'], axis=1, keepdims=True)

        # Ensure that the tension at nodes that have a production are at the right level
        

        # Iterate the neural network local updates
        for update in range(self.n_updates+1):

            # Send nodes V to lines origins and extremities
            self.v_from[str(update)] = tf.gather(self.v[str(update)], self.lines['from'], batch_dims=1)
            self.v_to[str(update)] = tf.gather(self.v[str(update)], self.lines['to'], batch_dims=1)

            # Send nodes theta to lines origins and extremities
            self.theta_from[str(update)] = tf.gather(self.theta[str(update)], self.lines['from'], batch_dims=1)
            self.theta_to[str(update)] = tf.gather(self.theta[str(update)], self.lines['to'], batch_dims=1)

            # At each line, get the active power leaving the node at the "from" side
            self.p_from_to[str(update)] = self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.sin(self.theta_from[str(update)] - self.theta_to[str(update)] - self.delta_ij - self.lines['angle']) +\
                self.v_from[str(update)]**2 / self.lines['ratio']**2 * self.y_ij * tf.math.sin(self.delta_ij)

            # At each line, get the active power leaving the node at the "to" side
            self.p_to_from[str(update)] = self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.sin(self.theta_to[str(update)] - self.theta_from[str(update)] - self.delta_ij + self.lines['angle']) +\
                self.v_to[str(update)]**2 * self.y_ij * tf.math.sin(self.delta_ij)
            
            # At each line, get the reactive power leaving the node at the "from" side
            self.q_from_to[str(update)] = - self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.cos(self.theta_from[str(update)] - self.theta_to[str(update)] - self.delta_ij - self.lines['angle']) +\
                self.v_from[str(update)]**2 / self.lines['ratio']**2 * (self.y_ij * tf.math.cos(self.delta_ij) - self.lines['b']/2)

            # At each line, get the reactive power leaving the node at the "to" side
            self.q_to_from[str(update)] = - self.v_from[str(update)]*self.v_to[str(update)]*self.y_ij/self.lines['ratio']*\
                tf.math.cos(self.theta_to[str(update)] - self.theta_from[str(update)] - self.delta_ij + self.lines['angle']) +\
                self.v_to[str(update)]**2 * (self.y_ij * tf.math.cos(self.delta_ij) - self.lines['b']/2)
            
            # Compute the active imbalance at each node
            self.delta_p[str(update)] = - scatter_nd_add_diff(self.dummy, self.indices_from, 
                                                             tf.reshape(self.p_from_to[str(update)], [-1])) \
                                        - scatter_nd_add_diff(self.dummy, self.indices_to, 
                                                             tf.reshape(self.p_to_from[str(update)], [-1])) \
                                        - self.buses['Pd'] / self.baseMVA \
                                        - self.buses['Gs'] * self.v[str(update)]**2 / self.baseMVA

            # Compute the reactive imbalance at each node
            self.delta_q[str(update)] = - scatter_nd_add_diff(self.dummy, self.indices_from, 
                                                             tf.reshape(self.q_from_to[str(update)], [-1])) \
                                        - scatter_nd_add_diff(self.dummy, self.indices_to, 
                                                             tf.reshape(self.q_to_from[str(update)], [-1])) \
                                        - self.buses['Qd'] / self.baseMVA \
                                        + self.buses['Bs'] * self.v[str(update)]**2 / self.baseMVA

            


            ### GLOBAL ACTIVE COMPENSATION ###
            self.p_joule[str(update)] = tf.abs(tf.abs(self.p_from_to[str(update)])-tf.abs(self.p_to_from[str(update)])) 

            self.apparent_consumption_per_node[str(update)] = self.buses['Pd'] / self.baseMVA \
                                                              + self.buses['Gs'] * self.v[str(update)]**2 / self.baseMVA

            self.apparent_consumption_per_sample[str(update)] = tf.reduce_sum(self.apparent_consumption_per_node[str(update)], axis=1, keepdims=True)
            self.apparent_consumption_per_sample[str(update)] += tf.reduce_sum(self.p_joule[str(update)], axis=1, keepdims=True)


            self.is_above[str(update)] = tf.math.sigmoid((self.apparent_consumption_per_sample[str(update)] - self.sum_p_gen_target_per_sample)/ \
                (1e-6 * (self.sum_p_gen_max_per_sample - self.sum_p_gen_min_per_sample)))
            self.is_below[str(update)] = 1. - self.is_above[str(update)]

            self.p_gen[str(update)] = self.is_below[str(update)] * ( (self.gens['Pg'] - self.gens['Pmin'])/self.gens['mbase'] \
                    * (self.apparent_consumption_per_sample[str(update)] - self.sum_p_gen_min_per_sample) \
                    / (self.sum_p_gen_target_per_sample - self.sum_p_gen_min_per_sample) \
                    +self.gens['Pmin']/self.gens['mbase']) \
                    + self.is_above[str(update)] * ((self.gens['Pmax'] - self.gens['Pg'])/self.gens['mbase'] \
                    * (self.apparent_consumption_per_sample[str(update)] + self.sum_p_gen_max_per_sample - 2*self.sum_p_gen_target_per_sample) \
                    / (self.sum_p_gen_max_per_sample - self.sum_p_gen_target_per_sample) \
                    +(-self.gens['Pmax'] +2*self.gens['Pg'])/self.gens['mbase'])

            

            self.delta_p[str(update)] += scatter_nd_add_diff(self.dummy, self.indices_gens, 
                                                              tf.reshape(self.p_gen[str(update)], [-1]))

            # Set the reactive generation to locally compensate
            self.q_gen[str(update)] = tf.gather(-self.delta_q[str(update)], self.gens['bus'], batch_dims=1)

            self.delta_q[str(update)] += scatter_nd_add_diff(self.dummy, self.indices_gens, tf.reshape(self.q_gen[str(update)], [-1]))

            #################################

            if update < self.n_updates : 

                ### NEURAL NETWORK UPDATE ###

                # Building message sum
                self.indices_from_messages = tf.ones([1,self.latent_dim, 1], dtype=tf.int32) * tf.expand_dims(self.lines['from'], 1)
                self.message_from[str(update)] = tf.gather(tf.transpose(self.latent_message[str(update)], [0,2,1]), 
                                                        self.indices_from_messages, 
                                                        batch_dims=2)
                self.message_from[str(update)] = tf.transpose(self.message_from[str(update)], [0,2,1])
                self.phi_message_from[str(update)] = self.phi(self.message_from[str(update)],
                                                            self.y_ij,
                                                            self.delta_ij,
                                                            self.lines['b'],
                                                            self.lines['ratio'],
                                                            self.lines['angle'])

                self.indices_to_messages = tf.ones([1,self.latent_dim, 1], dtype=tf.int32) * tf.expand_dims(self.lines['to'], 1)
                self.message_to[str(update)] = tf.gather(tf.transpose(self.latent_message[str(update)], [0,2,1]), 
                                                        self.indices_to_messages, 
                                                        batch_dims=2)
                self.message_to[str(update)] = tf.transpose(self.message_to[str(update)], [0,2,1])
                self.phi_message_to[str(update)] = self.phi(self.message_to[str(update)],
                                                          self.y_ij,
                                                          self.delta_ij,
                                                          self.lines['b'],
                                                          self.lines['ratio'],
                                                          self.lines['angle'])

                self.message_neighbors[str(update)] = scatter_nd_add_diff(self.dummy_message, self.indices_from, 
                                                                  tf.reshape(self.phi_message_to[str(update)], [-1, self.latent_dim])) \
                                                    + scatter_nd_add_diff(self.dummy_message, self.indices_to, 
                                                                  tf.reshape(self.phi_message_from[str(update)], [-1, self.latent_dim]))

                

                self.v[str(update+1)] = self.v[str(update)] + self.L_k(self.v[str(update)],
                                                                      self.theta[str(update)],
                                                                      self.delta_p[str(update)],
                                                                      self.delta_q[str(update)],
                                                                      self.latent_message[str(update)],
                                                                      self.message_neighbors[str(update)],
                                                                      type='v',
                                                                      update=update)

                self.theta[str(update+1)] = self.theta[str(update)] + self.L_k(self.v[str(update)],
                                                                              self.theta[str(update)],
                                                                              self.delta_p[str(update)],
                                                                              self.delta_q[str(update)],
                                                                              self.latent_message[str(update)],
                                                                              self.message_neighbors[str(update)],
                                                                              type='theta',
                                                                              update=update)

                self.latent_message[str(update+1)] = self.latent_message[str(update)] + self.L_k(self.v[str(update)],
                                                                                                self.theta[str(update)],
                                                                                                self.delta_p[str(update)],
                                                                                                self.delta_q[str(update)],
                                                                                                self.latent_message[str(update)],
                                                                                                self.message_neighbors[str(update)],
                                                                                                type='latent_message',
                                                                                                update=update)

                # Control V for the generators, so that it stays at the desired value
                self.v[str(update+1)] = v_check_control(self.v[str(update+1)], self.gens['Vg'], self.indices_gens)            

            # Define local loss at each node and on each sample
            #self.local_loss[str(update+1)] = tf.reduce_mean(tf.reduce_sum(self.delta_p[str(update)]**2 + self.delta_q[str(update)]**2, axis=1))
            self.local_loss[str(update+1)] = tf.reduce_mean(self.delta_p[str(update)]**2 + self.delta_q[str(update)]**2)
            #self.local_loss[str(update+1)] /= self.local_loss[str(1)]



            if self.total_loss is None:
                self.total_loss =  self.discount**((self.n_updates-update-1)/5)*self.local_loss[str(update+1)]#/self.local_loss[str(1)]
            else:
                self.total_loss +=  self.discount**((self.n_updates-update-1)/5)*self.local_loss[str(update+1)]#/self.local_loss[str(1)]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.gradients = self.optimizer.compute_gradients(self.total_loss)
        #def ClipIfNotNone(grad):
        #    if grad is None:
        #        return grad
        #    return tf.clip_by_value(grad, -self.max_grad, self.max_grad)
        #self.clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in self.gradients]
        #self.opt_op = self.optimizer.apply_gradients(self.clipped_gradients)

        self.opt_op = self.optimizer.minimize(self.total_loss)

        self.saver = tf.train.Saver()


    def _summary(self):
        """
        Build summaries
        """
        for update in range(self.n_updates+1):
            tf.summary.scalar("loss_at_update_{}".format(update), self.local_loss[str(update+1)]/self.local_loss[str(1)])
            
        tf.summary.scalar("total_loss", self.total_loss)

        self.merged_summary_op = tf.summary.merge_all()


    def _build_dicts(self):
        """
        Defines the dictionnaries that contain all the variables that will be updates throughout the model
        """

        self.delta_p = {}       # local active power mismatch
        self.delta_q = {}       # local reactive power mismatch

        self.v_from = {}        # v gathered at the "from" side of each line
        self.v_to = {}          # v gathered at the "to" side of each line

        self.theta_from = {}    # theta gathered at the "from" side of each line
        self.theta_to = {}      # theta gathered at the "to" side of each line
        
        self.p_from_to = {}     # active power going from the "from" side into the line
        self.p_to_from = {}     # active power going from the "to" side into the line

        self.q_from_to = {}     # reactive power going from the "from" side into the line
        self.q_to_from = {}     # reactive power going from the "to" side into the line

        self.message_from = {}    # messages carried by nodes on the "from" side of every line
        self.message_to = {}    # messages carried by nodes on the "to" side of every line
        self.message_neighbors = {}     # sum of neighboring messages for each node
        self.phi_message_from = {}
        self.phi_message_to = {}
        self.sum_message = {}
        self.global_message = {}

        self.p_joule = {}       # active power lost by Joule's effect
        self.p_gen = {}         # active power produced by each generator
        self.q_gen = {}         # reactive power produced by each generator 

        self.apparent_consumption_per_node = {}     # Consumption at each node, taking all effect into account
        self.apparent_consumption_per_sample = {}   # Total consumption + loss on each sample

        self.is_above = {}      # factor that is 1 if the load is above the generation setpoint
        self.is_below = {}      # factor that is 0 if the load is above the generation setpoint

        self.local_loss = {}    # Kirchhoff's law violation at each node
        self.total_loss = None  # total loss 



    def _params_to_variables(self):
        """
        This method declares some parameters as tf variables, so that they can later be modified.
        For instance, it may be useful to learn on small networks and then test on large ones.
        Such variables can then be modified by using:
        sess.run(model.n.assign(NEW_VALUE))
        This also means that after a call to this method there should be no other reference to
        the params dictionaries 
        """

        # Training parameters
        self.n_samples = tf.Variable(self.train_params['num_samples'], trainable=False)
        self.n_updates = self.model_params['num_updates']
        self.learning_rate = tf.Variable(self.train_params['learning_rate'], trainable=False)

        # Neural network params
        self.latent_layers = self.model_params['latent_layers']
        self.latent_dim = self.model_params['latent_dim']

        

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def L_k(self, v, theta, delta_p, delta_q, latent_message, message_neighbors, type='v', update=0):
        """
        """

        h = tf.stack([tf.reshape(v, [-1]),
                     tf.reshape(theta, [-1]),
                     tf.reshape(delta_p, [-1]),
                     tf.reshape(delta_q, [-1])],
                     axis=1)

        h = tf.concat([h, 
                      tf.reshape(latent_message, [-1, self.latent_dim]), 
                      tf.reshape(message_neighbors, [-1, self.latent_dim])],
                      axis=1)

        h = tf.layers.batch_normalization(h, training=True, name='batch_norm_'+type+'_{}'.format(update)) #, momentum=0.1

        for layer in range(self.latent_layers):

            left_dim = self.latent_dim
            right_dim = self.latent_dim
            if layer==0:
                left_dim = 4 + 2 * self.latent_dim
            if layer==self.latent_layers-1:
                right_dim = 1
                if type == 'latent_message':
                    right_dim = self.latent_dim

            W = tf.get_variable(name='W_'+type+'_{}_{}'.format(update, layer),
                shape=[left_dim, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)

            b = tf.get_variable(name='b_'+type+'_{}_{}'.format(update, layer),
                shape=[1, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)

            if layer==self.latent_layers-1:
                h = tf.matmul(h, W) + b
            else:
                h = self.non_lin(tf.matmul(h, W)+ b)

        if type == 'latent_message':
            h_out = tf.reshape(h, [self.n_samples, -1, self.latent_dim])
        else:
            h_out = tf.reshape(h, [self.n_samples, -1])

        return h_out * self.neural_net_output_coeff

    def phi(self, message, y, delta, b, ratio, angle):

        h = tf.stack([tf.reshape(y, [-1]),
                     tf.reshape(delta, [-1]),
                     tf.reshape(b, [-1]),
                     tf.reshape(ratio, [-1]),
                     tf.reshape(angle, [-1])],
                     axis=1)

        h = tf.concat([h, 
                      tf.reshape(message, [-1, self.latent_dim])],
                      axis=1)
        h = tf.layers.batch_normalization(h, training=True, name='batch_norm_phi')

        for layer in range(self.latent_layers):

            left_dim = self.latent_dim
            right_dim = self.latent_dim
            if layer==0:
                left_dim = 5 + self.latent_dim
            if layer==self.latent_layers-1:
                right_dim = self.latent_dim

            W = tf.get_variable(name='W_phi_{}'.format(layer),
                shape=[left_dim, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)

            b = tf.get_variable(name='b_phi_{}'.format(layer),
                shape=[1, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)

            if layer==self.latent_layers-1:
                h = tf.matmul(h, W) + b
            else:
                h = self.non_lin(tf.matmul(h, W)+ b)

        h_out = tf.reshape(h, [self.n_samples, -1, self.latent_dim])

        return h_out * self.neural_net_output_coeff

   
    def store_summary(self, learn_iter):
        self.summary = self.sess.run(self.merged_summary_op)
        self.summary_writer.add_summary(self.summary, learn_iter)
        
    def normalization_params(self):

        # outputs
        self.alpha_line_out_theta = 1e-3
        self.alpha_line_out_V = 1e-3

    def import_power_grid(self, case='case14'):
        """
        Import files from pypower
        """
        from pypower.case9 import case9        
        from pypower.case14 import case14
        from pypower.case30 import case30
        from pypower.case57 import case57
        from pypower.case118 import case118
        from pypower.case300 import case300

        if case=='case9':
            self.input_data = case9()
        elif case=='case14':
            self.input_data = case14()
        elif case=='case30':
            self.input_data = case30()
        elif case=='case57':
            self.input_data = case57()
        elif case=='case118':
            self.input_data = case118()
        elif case=='case300':
            self.input_data = case300()
        else:
            print("This case is not available!")



        bus_i = np.reshape(self.input_data['bus'][:,0], [1, -1])
        fbus = np.reshape(self.input_data['branch'][:,0], [-1, 1])
        tbus = np.reshape(self.input_data['branch'][:,1], [-1, 1])
        gbus = np.reshape(self.input_data['gen'][:,0], [-1, 1])

        fbus = np.where(bus_i-fbus==0)[1]
        tbus = np.where(bus_i-tbus==0)[1]
        gbus = np.where(bus_i-gbus==0)[1]


        ratios = self.input_data['branch'][:,8] +1.*(self.input_data['branch'][:,8]==0.)



        self.n_nodes = tf.Variable(self.input_data['bus'].shape[0], trainable=False, dtype=tf.int32)
        self.n_gens = tf.Variable(self.input_data['gen'].shape[0], trainable=False, dtype=tf.int32)
        self.n_lines = tf.Variable(self.input_data['branch'].shape[0], trainable=False, dtype=tf.int32)

        self.baseMVA_default = tf.Variable(self.input_data['baseMVA']*np.ones([1,1]), trainable=False, dtype=tf.float32)

        self.buses_default = {}
        self.buses_default['baseKV'] = tf.Variable(np.reshape(self.input_data['bus'][:,9], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Pd'] = tf.Variable(np.reshape(self.input_data['bus'][:,2], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Qd'] = tf.Variable(np.reshape(self.input_data['bus'][:,3], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Gs'] = tf.Variable(np.reshape(self.input_data['bus'][:,4], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Bs'] = tf.Variable(np.reshape(self.input_data['bus'][:,5], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)

        self.lines_default = {}
        self.lines_default['from'] = tf.Variable(np.reshape(fbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['to'] = tf.Variable(np.reshape(tbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['r'] = tf.Variable(np.reshape(self.input_data['branch'][:,2], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['x'] = tf.Variable(np.reshape(self.input_data['branch'][:,3], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['b'] = tf.Variable(np.reshape(self.input_data['branch'][:,4], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['ratio'] = tf.Variable(np.reshape(ratios, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['angle'] = tf.Variable(np.reshape(self.input_data['branch'][:,9], [1, -1])*np.pi/180, trainable=False, dtype=tf.float32)#, validate_shape=False)

        self.gens_default = {}
        self.gens_default['bus'] = tf.Variable(np.reshape(gbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.gens_default['Vg'] = tf.Variable(np.reshape(self.input_data['gen'][:,5], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pg'] = tf.Variable(np.reshape(self.input_data['gen'][:,1], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmin'] = tf.Variable(np.reshape(self.input_data['gen'][:,9], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmax'] = tf.Variable(np.reshape(self.input_data['gen'][:,8], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['mbase'] = tf.Variable(np.reshape(self.input_data['gen'][:,6], [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)


        self.sess.run(tf.initializers.variables(self.buses_default.values()))
        self.sess.run(tf.initializers.variables(self.lines_default.values()))
        self.sess.run(tf.initializers.variables(self.gens_default.values()))
        self.sess.run(tf.initializers.variables([self.n_nodes,
                                                 self.n_gens,
                                                 self.n_lines,
                                                 self.baseMVA_default]))

    """
    def import_power_grid(self, case='case14'):

        # Import case
        data = parse_file('data/'+case+'.m')

        # Adjust for the fact that the ratio is set to 0.0 for transmission lines
        data['lines'].ratio = 1.*(data['lines'].ratio==0.0) + data['lines'].ratio

        # Convert indices to integers and subtract 1 so that indexing starts at 0
        data['lines'] = data['lines'].astype({"fbus": int, "tbus": int})
        #data['lines'][['fbus', 'tbus']] -= 1
        data['buses'] = data['buses'].astype({"bus_i": int})
        #data['buses']['bus_i'] -= 1
        data['gens'] = data['gens'].astype({"bus": int})
        #data['gens']['bus'] -= 1

        bus_i = np.reshape(data['buses'].bus_i.values, [1, -1])
        fbus = np.reshape(data['lines'].fbus.values, [-1, 1])
        tbus = np.reshape(data['lines'].tbus.values, [-1, 1])
        gbus = np.reshape(data['gens'].bus.values, [-1, 1])

        fbus = np.where(bus_i-fbus==0)[1]
        tbus = np.where(bus_i-tbus==0)[1]
        gbus = np.where(bus_i-gbus==0)[1]

        # n_samples est déjà récupéré dans _params_to_variables
        #self.n_samples = tf.Variable(, trainable=False)


        self.n_nodes = tf.Variable(len(data['buses']), trainable=False, dtype=tf.int32)
        self.n_gens = tf.Variable(len(data['gens']), trainable=False, dtype=tf.int32)
        self.n_lines = tf.Variable(len(data['lines']), trainable=False, dtype=tf.int32)

        self.baseMVA_default = tf.Variable(data['baseMVA']*np.ones([1,1]), trainable=False, dtype=tf.float32)

        self.buses_default = {}
        self.buses_default['baseKV'] = tf.Variable(np.reshape(data['buses'].baseKV.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Pd'] = tf.Variable(np.reshape(data['buses'].Pd.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Qd'] = tf.Variable(np.reshape(data['buses'].Qd.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Gs'] = tf.Variable(np.reshape(data['buses'].Gs.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.buses_default['Bs'] = tf.Variable(np.reshape(data['buses'].Bs.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)

        self.lines_default = {}
        self.lines_default['from'] = tf.Variable(np.reshape(fbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['to'] = tf.Variable(np.reshape(tbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.lines_default['r'] = tf.Variable(np.reshape(data['lines'].r.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['x'] = tf.Variable(np.reshape(data['lines'].x.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['b'] = tf.Variable(np.reshape(data['lines'].b.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['ratio'] = tf.Variable(np.reshape(data['lines'].ratio.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.lines_default['angle'] = tf.Variable(np.pi/180*np.reshape(data['lines'].angle.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)

        self.gens_default = {}
        self.gens_default['bus'] = tf.Variable(np.reshape(gbus, [1, -1]), trainable=False, dtype=tf.int32)#, validate_shape=False)
        self.gens_default['Vg'] = tf.Variable(np.reshape(data['gens'].Vg.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pg'] = tf.Variable(np.reshape(data['gens'].Pg.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmin'] = tf.Variable(np.reshape(data['gens'].Pmin.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['Pmax'] = tf.Variable(np.reshape(data['gens'].Pmax.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)
        self.gens_default['mbase'] = tf.Variable(np.reshape(data['gens'].mBase.values, [1, -1]), trainable=False, dtype=tf.float32)#, validate_shape=False)


        self.sess.run(tf.initializers.variables(self.buses_default.values()))
        self.sess.run(tf.initializers.variables(self.lines_default.values()))
        self.sess.run(tf.initializers.variables(self.gens_default.values()))
        self.sess.run(tf.initializers.variables([self.n_nodes,
                                                 self.n_gens,
                                                 self.n_lines,
                                                 self.baseMVA_default]))
    """

    def perturbate_power_grid(self, noise_params):

        # Duplicate along the sample dimension
        temp_1 = tf.reshape(self.n_samples,[1])
        temp_2 = tf.ones([1], tf.int32)
        temp_dim = tf.concat([temp_1, temp_2], axis=0)
        temp_duplicate_int = tf.ones(temp_dim, tf.int32)
        temp_duplicate_float = tf.ones(temp_dim, tf.float32)

        self.buses = {}
        self.lines = {}
        self.gens = {}

        self.baseMVA = self.baseMVA_default*temp_duplicate_float

        self.buses['baseKV'] = self.buses_default['baseKV']*temp_duplicate_float
        self.buses['Pd'] = self.buses_default['Pd']*temp_duplicate_float
        self.buses['Qd'] = self.buses_default['Qd']*temp_duplicate_float
        self.buses['Gs'] = self.buses_default['Gs']*temp_duplicate_float
        self.buses['Bs'] = self.buses_default['Bs']*temp_duplicate_float

        self.lines['from'] = self.lines_default['from']*temp_duplicate_int
        self.lines['to'] = self.lines_default['to']*temp_duplicate_int
        self.lines['r'] = self.lines_default['r']*temp_duplicate_float
        self.lines['x'] = self.lines_default['x']*temp_duplicate_float
        self.lines['b'] = self.lines_default['b']*temp_duplicate_float
        self.lines['ratio'] = self.lines_default['ratio']*temp_duplicate_float
        self.lines['angle'] = self.lines_default['angle']*temp_duplicate_float

        self.gens['bus'] = self.gens_default['bus']*temp_duplicate_int
        self.gens['Vg'] = self.gens_default['Vg']*temp_duplicate_float
        self.gens['Pg'] = self.gens_default['Pg']*temp_duplicate_float
        self.gens['Pmin'] = self.gens_default['Pmin']*temp_duplicate_float
        self.gens['Pmax'] = self.gens_default['Pmax']*temp_duplicate_float
        self.gens['mbase'] = self.gens_default['mbase']*temp_duplicate_float

        # Get all the noise parameters
        self.is_noise_active = tf.Variable(noise_params['is_noise_active'], trainable=False)

        self.noise = {}
        self.noise['r_min_coeff'] = tf.Variable(noise_params['r_min_coeff'], trainable=False)
        self.noise['r_max_coeff'] = tf.Variable(noise_params['r_max_coeff'], trainable=False)
        self.noise['x_min_coeff'] = tf.Variable(noise_params['x_min_coeff'], trainable=False)
        self.noise['x_max_coeff'] = tf.Variable(noise_params['x_max_coeff'], trainable=False)
        self.noise['b_min_coeff'] = tf.Variable(noise_params['b_min_coeff'], trainable=False)
        self.noise['b_max_coeff'] = tf.Variable(noise_params['b_max_coeff'], trainable=False)
        self.noise['ratio_min_coeff'] = tf.Variable(noise_params['ratio_min_coeff'], trainable=False)
        self.noise['ratio_max_coeff'] = tf.Variable(noise_params['ratio_max_coeff'], trainable=False)
        self.noise['angle_min_offset'] = tf.Variable(noise_params['angle_min_offset'], trainable=False)
        self.noise['angle_max_offset'] = tf.Variable(noise_params['angle_max_offset'], trainable=False)
        self.noise['Vg_min'] = tf.Variable(noise_params['Vg_min'], trainable=False)
        self.noise['Vg_max'] = tf.Variable(noise_params['Vg_max'], trainable=False)
        self.noise['Pd_min_coeff'] = tf.Variable(noise_params['Pd_min_coeff'], trainable=False)
        self.noise['Pd_max_coeff'] = tf.Variable(noise_params['Pd_max_coeff'], trainable=False)
        self.noise['Qd_min_coeff'] = tf.Variable(noise_params['Qd_min_coeff'], trainable=False)
        self.noise['Qd_max_coeff'] = tf.Variable(noise_params['Qd_max_coeff'], trainable=False)
        self.noise['Bs_min_coeff'] = tf.Variable(noise_params['Bs_min_coeff'], trainable=False)
        self.noise['Bs_max_coeff'] = tf.Variable(noise_params['Bs_max_coeff'], trainable=False)
        self.noise['Gs_min_coeff'] = tf.Variable(noise_params['Gs_min_coeff'], trainable=False)
        self.noise['Gs_max_coeff'] = tf.Variable(noise_params['Gs_max_coeff'], trainable=False)

        self.sess.run(tf.initializers.variables([self.is_noise_active]))
        self.sess.run(tf.initializers.variables(self.noise.values()))

        # Perturbate line characteristics ???
        self.lines['r'] = self.lines['r'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_lines], self.noise['r_min_coeff'], self.noise['r_max_coeff']))
        self.lines['x'] = self.lines['x'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_lines], self.noise['x_min_coeff'], self.noise['x_max_coeff']))
        self.lines['b'] = self.lines['b'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_lines], self.noise['b_min_coeff'], self.noise['b_max_coeff']))
        self.lines['ratio'] = self.lines['ratio'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_lines], self.noise['ratio_min_coeff'], self.noise['ratio_max_coeff']))
        self.lines['angle'] = self.lines['angle'] + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_lines], self.noise['angle_min_offset'], self.noise['angle_max_offset'])


        self.gens['Pg'] = (1.-self.is_noise_active) * self.gens['Pg'] +\
            self.is_noise_active*( (0.75*self.gens['Pmin'] + 0.25*self.gens['Pmax']) + (0.5*self.gens['Pmax']-0.5*self.gens['Pmin']) *\
             tf.random.uniform([self.n_samples, self.n_gens], 0., 1., dtype=tf.float32))
        self.gens['Vg'] = (1.-self.is_noise_active) * self.gens['Vg'] +\
            self.is_noise_active*( self.noise['Vg_min'] + (self.noise['Vg_max']-self.noise['Vg_min']) * tf.random.uniform([self.n_samples, self.n_gens], 0., 1., dtype=tf.float32))

        self.P_tot = tf.reduce_sum(self.gens['Pg'], axis=1, keepdims=True)

        self.buses['Pd'] = self.buses['Pd'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_nodes], self.noise['Pd_min_coeff'], self.noise['Pd_max_coeff']))
        self.buses['Pd'] = (1.-self.is_noise_active)*self.buses['Pd'] + self.is_noise_active * self.buses['Pd'] * self.P_tot / tf.reduce_sum(self.buses['Pd'], axis=1, keepdims=True)

        self.buses['Qd'] = self.buses['Qd'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_nodes], self.noise['Qd_min_coeff'], self.noise['Qd_max_coeff']))
        self.buses['Gs'] = self.buses['Gs'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_nodes], self.noise['Gs_min_coeff'], self.noise['Gs_max_coeff']))
        self.buses['Bs'] = self.buses['Bs'] * (1.0 + self.is_noise_active * \
            tf.random.uniform([self.n_samples, self.n_nodes], self.noise['Bs_min_coeff'], self.noise['Bs_max_coeff']))



    def change_default_power_grid(self, case='case14'):

        from pypower.case9 import case9        
        from pypower.case14 import case14
        from pypower.case30 import case30
        from pypower.case57 import case57
        from pypower.case118 import case118
        from pypower.case300 import case300

        if case=='case9':
            self.input_data = case9()
        elif case=='case14':
            self.input_data = case14()
        elif case=='case30':
            self.input_data = case30()
        elif case=='case57':
            self.input_data = case57()
        elif case=='case118':
            self.input_data = case118()
        elif case=='case300':
            self.input_data = case300()
        else:
            print("This case is not available!")



        bus_i = np.reshape(self.input_data['bus'][:,0], [1, -1])
        fbus = np.reshape(self.input_data['branch'][:,0], [-1, 1])
        tbus = np.reshape(self.input_data['branch'][:,1], [-1, 1])
        gbus = np.reshape(self.input_data['gen'][:,0], [-1, 1])

        fbus = np.where(bus_i-fbus==0)[1]
        tbus = np.where(bus_i-tbus==0)[1]
        gbus = np.where(bus_i-gbus==0)[1]


        ratios = self.input_data['branch'][:,8] +1.*(self.input_data['branch'][:,8]==0.)

        self.sess.run(self.n_nodes.assign(self.input_data['bus'].shape[0]))
        self.sess.run(self.n_gens.assign(self.input_data['gen'].shape[0]))
        self.sess.run(self.n_lines.assign(self.input_data['branch'].shape[0]))
        self.sess.run(self.baseMVA_default.assign(self.input_data['baseMVA']*np.ones([1,1])))

        self.sess.run(tf.assign(self.buses_default['baseKV'], np.reshape(self.input_data['bus'][:,9], [1, -1]) , validate_shape=False))
        self.sess.run(tf.assign(self.buses_default['Pd'], np.reshape(self.input_data['bus'][:,2], [1, -1]), validate_shape=False))
        self.sess.run(tf.assign(self.buses_default['Qd'], np.reshape(self.input_data['bus'][:,3], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.buses_default['Gs'], np.reshape(self.input_data['bus'][:,4], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.buses_default['Bs'], np.reshape(self.input_data['bus'][:,5], [1, -1]), validate_shape=False ))

        self.sess.run(tf.assign(self.lines_default['from'], np.reshape(fbus, [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['to'], np.reshape(tbus, [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['r'], np.reshape(self.input_data['branch'][:,2], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['x'], np.reshape(self.input_data['branch'][:,3], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['b'], np.reshape(self.input_data['branch'][:,4], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['ratio'], np.reshape(ratios, [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.lines_default['angle'], np.reshape(self.input_data['branch'][:,9], [1, -1])*np.pi/180, validate_shape=False ))

        self.sess.run(tf.assign(self.gens_default['bus'], np.reshape(gbus, [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.gens_default['Pg'], np.reshape(self.input_data['gen'][:,1], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.gens_default['Vg'], np.reshape(self.input_data['gen'][:,5], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.gens_default['Pmin'], np.reshape(self.input_data['gen'][:,9], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.gens_default['Pmax'], np.reshape(self.input_data['gen'][:,8], [1, -1]), validate_shape=False ))
        self.sess.run(tf.assign(self.gens_default['mbase'], np.reshape(self.input_data['gen'][:,6], [1, -1]), validate_shape=False ))

    """
    def change_default_power_grid(self, case='case14'):

        # Import case
        data = parse_file('data/'+case+'.m')

        # Adjust for the fact that the ratio is set to 0.0 for transmission lines
        data['lines'].ratio = 1.*(data['lines'].ratio==0.0) + data['lines'].ratio

        # Convert indices to integers and subtract 1 so that indexing starts at 0
        data['lines'] = data['lines'].astype({"fbus": int, "tbus": int})
        #data['lines'][['fbus', 'tbus']] -= 1
        data['buses'] = data['buses'].astype({"bus_i": int})
        #data['buses']['bus_i'] -= 1
        data['gens'] = data['gens'].astype({"bus": int})
        #data['gens']['bus'] -= 1

        bus_i = np.reshape(data['buses'].bus_i.values, [1, -1])
        fbus = np.reshape(data['lines'].fbus.values, [-1, 1])
        tbus = np.reshape(data['lines'].tbus.values, [-1, 1])
        gbus = np.reshape(data['gens'].bus.values, [-1, 1])

        fbus = np.where(bus_i-fbus==0)[1]
        tbus = np.where(bus_i-tbus==0)[1]
        gbus = np.where(bus_i-gbus==0)[1]

        # n_samples est déjà récupéré dans _params_to_variables
        #self.n_samples = tf.Variable(, trainable=False)


        self.sess.run([self.n_nodes.assign(len(data['buses'])),
                       self.n_gens.assign(len(data['gens'])),
                       self.n_lines.assign(len(data['lines'])),
                       self.baseMVA_default.assign(data['baseMVA']*np.ones([1,1]))])

        self.sess.run([tf.assign(self.buses_default['baseKV'], np.reshape(data['buses'].baseKV.values, [1, -1]), validate_shape=False),
                       tf.assign(self.buses_default['Pd'], np.reshape(data['buses'].Pd.values, [1, -1]), validate_shape=False),
                       tf.assign(self.buses_default['Qd'], np.reshape(data['buses'].Qd.values, [1, -1]), validate_shape=False),
                       tf.assign(self.buses_default['Gs'], np.reshape(data['buses'].Gs.values, [1, -1]), validate_shape=False),
                       tf.assign(self.buses_default['Bs'], np.reshape(data['buses'].Bs.values, [1, -1]), validate_shape=False)])

        self.sess.run([tf.assign(self.lines_default['from'], np.reshape(fbus, [1, -1]), validate_shape=False),       
                    tf.assign(self.lines_default['to'], np.reshape(tbus, [1, -1]), validate_shape=False),        
                    tf.assign(self.lines_default['r'], np.reshape(data['lines'].r.values, [1, -1]), validate_shape=False),
                    tf.assign(self.lines_default['x'], np.reshape(data['lines'].x.values, [1, -1]), validate_shape=False),
                    tf.assign(self.lines_default['b'], np.reshape(data['lines'].b.values, [1, -1]), validate_shape=False),
                    tf.assign(self.lines_default['ratio'], np.reshape(data['lines'].ratio.values, [1, -1]), validate_shape=False),
                    tf.assign(self.lines_default['angle'], np.pi/180*np.reshape(data['lines'].angle.values, [1, -1]), validate_shape=False)])

        self.sess.run([tf.assign(self.gens_default['bus'], np.reshape(gbus, [1, -1]), validate_shape=False),           
                    tf.assign(self.gens_default['Vg'], np.reshape(data['gens'].Vg.values, [1, -1]), validate_shape=False),
                    tf.assign(self.gens_default['Pg'], np.reshape(data['gens'].Pg.values, [1, -1]), validate_shape=False),
                    tf.assign(self.gens_default['Pmin'], np.reshape(data['gens'].Pmin.values, [1, -1]), validate_shape=False),
                    tf.assign(self.gens_default['Pmax'], np.reshape(data['gens'].Pmax.values, [1, -1]), validate_shape=False),
                    tf.assign(self.gens_default['mbase'], np.reshape(data['gens'].mBase.values, [1, -1]), validate_shape=False)])
    """
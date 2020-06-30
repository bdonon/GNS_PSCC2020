"""import tensorflow as tf

non_lin = tf.nn.leaky_relu
#non_lin = tf.tanh

alpha_out_V = 400./10000000
alpha_out_theta = 0.000001
#alpha_in = 1e-6
#init = 1e-4

alpha_in_V = 0.1/400.
alpha_in_theta = 0.1/(0.00001)
alpha_in_P = 0.1/100.
alpha_in_Q = 0.1/100.
alpha_in_r = 0.1e-3
alpha_in_x = 0.1e-1

def update_through_lines(self, V, theta, lines, line_charac, update):

    delta_V = tf.batch_gather(V, lines[:,:,0]) \
              - tf.batch_gather(V, lines[:,:,1])
    delta_V = tf.expand_dims(delta_V, -1)

    delta_theta = tf.batch_gather(theta, lines[:,:,0]) \
                  - tf.batch_gather(theta, lines[:,:,1])
    delta_theta = tf.expand_dims(delta_theta, -1)

    h_in = tf.concat([alpha_in_V*delta_V,
                      alpha_in_theta*delta_theta, 
                      alpha_in_r*line_charac['r'], 
                      alpha_in_x*line_charac['x']], axis=-1)
    h_in = tf.reshape(h_in, [-1,4])

    for layer in range(self.latent_layers):

        left_dim = self.latent_dim
        right_dim = self.latent_dim
        if layer==0:
            left_dim = 4 # TODO : Exprimer ça de façon générique
        if layer==self.latent_layers-1:
            right_dim = 2 # TODO : Exprimer ça de façon générique

        W = tf.get_variable(name='W_{}_{}'.format(update, layer),
            shape=[left_dim, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        b = tf.get_variable(name='b_{}_{}'.format(update, layer),
            shape=[1, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        if layer==self.latent_layers-1:
            h_in = non_lin(tf.matmul(h_in, W)+ b)
        else:
            h_in = tf.matmul(h_in, W) + b




    return alpha_out_V*(h_in[:,0]), alpha_out_theta*(h_in[:,1])



def update_through_load(self, V, theta, P, Q, connection, update):


    V_load = tf.batch_gather(V, connection[:,:,1])
    theta_load = tf.batch_gather(theta, connection[:,:,1])

    h_load_in = tf.stack([alpha_in_V*V_load,
                          alpha_in_theta*theta_load,
                          alpha_in_P*P, 
                          alpha_in_Q*Q], 
                          axis=-1)
    h_load_in = tf.reshape(h_load_in, [-1,4])

    for layer in range(self.latent_layers):

        left_dim = self.latent_dim
        right_dim = self.latent_dim
        if layer==0:
            left_dim = 4
        if layer==self.latent_dim-1:
            right_dim = 2

        W = tf.get_variable(name='W_load_{}_{}'.format(update, layer),
            shape=[left_dim, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        b = tf.get_variable(name='b_load_{}_{}'.format(update, layer),
            shape=[1, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        if layer==self.latent_dim-1:
            h_load_in = non_lin(tf.matmul(h_load_in, W)+ b)
        else:
            h_load_in = tf.matmul(h_load_in, W) + b

    return alpha_out_V*(h_load_in[:,0]), alpha_out_theta*(h_load_in[:,1])

def update_through_gen(self, V, theta, P, V_consign, connection, update):


    V_load = tf.batch_gather(V, connection[:,:,1])
    theta_load = tf.batch_gather(theta, connection[:,:,1])

    h_load_in = tf.stack([alpha_in_V*V_load,
                          alpha_in_theta*theta_load,
                          alpha_in_P*P, 
                          alpha_in_V*V_consign], 
                          axis=-1)
    h_load_in = tf.reshape(h_load_in, [-1,4])

    for layer in range(self.latent_layers):

        left_dim = self.latent_dim
        right_dim = self.latent_dim
        if layer==0:
            left_dim = 4
        if layer==self.latent_dim-1:
            right_dim = 1

        W = tf.get_variable(name='W_gen_{}_{}'.format(update, layer),
            shape=[left_dim, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        b = tf.get_variable(name='b_gen_{}_{}'.format(update, layer),
            shape=[1, right_dim],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
            trainable=True,
            dtype=tf.float32)

        if layer==self.latent_dim-1:
            h_load_in = non_lin(tf.matmul(h_load_in, W)+ b)
        else:
            h_load_in = tf.matmul(h_load_in, W) + b



    return alpha_out_V*(h_load_in[:,0])


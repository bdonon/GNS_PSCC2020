
import tensorflow as tf


def network_sampler(network_params, train_params, sess):
    """
    Samples multiple different power grid topologies, as well as line characteristics.
    lines_connection is a set of couples (node_or, node_ex) and lines_charac is a set
    of couples (r, x)

    Inputs : 
        injection_params : dict
        train_params = dict

    Outputs : 
        line_connection : tf.Tensor
        line_charac : dict of tf.Tensor
    """


    ### TOPOLOGY SAMPLING ###

    # The top part of injection_params is actually the same for every sampled power grid
    # Since the ordering has no importance
    line_connection_up_left = tf.range(0, self.n_nodes-1, 1)
    line_connection_up_right = tf.range(1, self.n_nodes, 1)
    line_connection_up = tf.stack([line_connection_up_left, line_connection_up_right], 1)
    line_connection_up = tf.expand_dims(line_connection_up, 0)
    line_connection_up = line_connection_up * tf.ones([self.n_samples,1,1], tf.int32)

    # The bottom part is the one that is sampled randomly
    line_connection_down = tf.random_uniform([self.n_samples, self.n_lines-self.n_nodes+1, 2], 
                                             minval=0, 
                                             maxval=self.n_nodes-1, 
                                             dtype=tf.int32)

    # Concatenate the top and bottom parts of lines_connection
    line_connection = tf.concat([line_connection_up, line_connection_down], 1)
    #line_connection = tf.Variable(line_connection)
    #sess.run(tf.global_variables_initializer())

    ### CHARACTERISTICS SAMPLING ###

    # Initialize output
    line_charac = {}

    # Sample line characteristics
    line_charac['r'] = tf.random_uniform([ns, m, 1],
                          minval=self.r_min,
                          maxval=self.r_max)
    line_charac['x'] = tf.random_uniform([ns, m, 1],
                          minval=self.x_min,
                          maxval=self.x_max)

    line_connection = tf.Variable(line_connection, trainable=False)
    line_charac['x'] = tf.Variable(line_charac['x'], trainable=False)
    line_charac['r'] = tf.Variable(line_charac['r'], trainable=False)

    line_init = tf.initializers.variables(
        [line_connection, line_charac['x'], line_charac['r']],
        name='line_init'
    )
        
    return line_connection, line_charac, line_init





def injection_sampler(injection_params, network_params, train_params, sess):
    """
    Samples injection locations and characteristics

    Inputs : 
        injection_params : dict
        network_params : dict
        train_params : dict

    Outputs : 
        gen : dict of tf.Tensor
        load : dict of tf.Tensor
    """

    # Get constants from param dicts
    n_nodes = network_params['n']               # Number of nodes
    n_lines = network_params['m']               # Number of lines
    n_samples = train_params['NUM_SAMPLES']     # Number of samples
    n_loads = injection_params['n_loads']        # Number of loads
    n_gens = injection_params['n_gens']          # Number of generators

    # Initialize outputs
    gen = {}
    load = {}


    ### CONSUMPTION LEVEL ###

    # One needs to compute, for each sample the total amount of consumption
    P_load_tot = tf.random_uniform([n_samples, 1],
                                   injection_params['P_load_tot_min'],
                                   injection_params['P_load_tot_max'])
    P_load_tot = tf.Variable(P_load_tot, trainable=False)
    sess.run(tf.global_variables_initializer())




    ### LOAD SAMPLING ###

    # Build left part of the connections
    load_connection_left = tf.range(0, n_loads, 1)
    load_connection_left = tf.expand_dims(load_connection_left, 0)
    load_connection_left = load_connection_left * tf.ones([n_samples,1], tf.int32)

    # Sample right part of the connections
    load_connection_right = tf.random_uniform([n_samples, n_nodes])
    load_connection_right = tf.nn.top_k(load_connection_right, n_loads)[1]

    # Stack the left and right parts of the connections
    load['connection'] = tf.stack([load_connection_left, load_connection_right], -1)
    load['connection'] = tf.Variable(load['connection'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sample potentials to dispatch consumption
    E_load = tf.random_uniform([n_samples, n_loads],
                               injection_params['E_load_min'],
                               injection_params['E_load_max'])
    P_load_coeff = tf.math.sigmoid(-E_load)
    P_load_coeff = P_load_coeff / tf.reduce_sum(P_load_coeff, axis=1, keep_dims=True)
    load['P'] = P_load_tot * P_load_coeff
    load['P'] = tf.Variable(load['P'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sample reactive load
    Q_load = injection_params['A'] * load['P'] + injection_params['B'] 
    load['Q'] = Q_load + tf.random_uniform([n_samples, n_loads],
                                              -injection_params['Q_load_amplitude']/2,
                                              injection_params['Q_load_amplitude']/2)
    load['Q'] = tf.Variable(load['Q'], trainable=False)
    sess.run(tf.global_variables_initializer())


    ### GEN SAMPLING ###

    # Build left part of the connections
    gen_connection_left = tf.range(0, n_gens, 1)
    gen_connection_left = tf.expand_dims(gen_connection_left, 0)
    gen_connection_left = gen_connection_left * tf.ones([n_samples,1], tf.int32)

    # Sample right part of the connections
    gen_connection_right = tf.random_uniform([n_samples, n_nodes])
    gen_connection_right = tf.nn.top_k(gen_connection_right, n_gens)

    # Stack the left and right parts of the connections
    gen['connection'] = tf.stack([gen_connection_left, gen_connection_right[1]], -1)
    gen['connection'] = tf.Variable(gen['connection'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sample active generation
    E_gen = tf.random_uniform([n_samples, n_gens],
                                  injection_params['E_gen_min'],
                                  injection_params['E_gen_max'])
    P_load_coeff = tf.math.sigmoid(-E_gen)
    P_load_coeff = P_load_coeff / tf.reduce_sum(P_load_coeff, axis=1, keep_dims=True)
    gen['P_target'] = P_load_tot * P_load_coeff
    gen['P_target'] = tf.Variable(gen['P_target'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sample tension consign
    gen['V'] = tf.random_uniform([n_samples, n_gens],
                          injection_params['V_gen_min'],
                          injection_params['V_gen_max'])
    gen['V'] = tf.Variable(gen['V'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sampling P_max
    gen['P_max'] = injection_params['P_gen_max'] * tf.ones([n_samples, n_gens])
    gen['P_max'] = tf.Variable(gen['P_max'], trainable=False)
    sess.run(tf.global_variables_initializer())
    gen['P_min'] = injection_params['P_gen_min'] * tf.ones([n_samples, n_gens])
    gen['P_min'] = tf.Variable(gen['P_min'], trainable=False)
    sess.run(tf.global_variables_initializer())

    # Sampling Q_max
    gen['Q_max'] = injection_params['Q_gen_max'] * tf.ones([n_samples, n_gens])
    gen['Q_max'] = tf.Variable(gen['Q_max'], trainable=False)
    sess.run(tf.global_variables_initializer())
    gen['Q_min'] = injection_params['Q_gen_min'] * tf.ones([n_samples, n_gens])
    gen['Q_min'] = tf.Variable(gen['Q_min'], trainable=False)
    sess.run(tf.global_variables_initializer())

    inj_init = tf.initializers.variables(
        [P_load_tot,
        load['connection'],
        load['P'],
        load['Q'],
        gen['connection'],
        gen['P_target'],
        gen['V'],
        gen['P_max'],
        gen['P_min'],
        gen['Q_max'],
        gen['Q_min']
        ],
        name='inj_init'
    )


    return gen, load, inj_init


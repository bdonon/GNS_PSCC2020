# This script aims at training and testing GNS

import os
import tqdm
import time
import argparse
import json
import shutil

import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

from pypower.runpf import runpf
from pypower.rundcpf import rundcpf

from module.model import GNS


def main(train_case, storing_dir):

    # Define the parameters of the training. 
    # More informations about the distributions used during training can be found 
    # in the notebook Sampling random power networks.
    train_params = {
        'num_samples' : 1000,
        'learning_rate' : 3e-3,
    }

    # Define the parameters of the model.
    model_params = {
        'num_updates' : 30,#10,
        'latent_dim' : 10,
        'latent_layers' : 2#3
    }

    noise_params = {
        'is_noise_active' : 1.,

        'r_min_coeff' : -0.1,
        'r_max_coeff' : +0.1,
        'x_min_coeff' : -0.1,
        'x_max_coeff' : +0.1,
        'b_min_coeff' : -0.1,
        'b_max_coeff' : +0.1,
        'ratio_min_coeff' : -0.2,
        'ratio_max_coeff' : +0.2,
        'angle_min_offset' : -0.2,
        'angle_max_offset' : +0.2,

        'Vg_min' : 0.95,
        'Vg_max' : 1.05,

        'Pd_min_coeff' : -0.5,
        'Pd_max_coeff' : +0.5,
        'Qd_min_coeff' : -0.5,
        'Qd_max_coeff' : +0.5,
        'Gs_min_coeff' : -0.,
        'Gs_max_coeff' : +0.,
        'Bs_min_coeff' : -0.,
        'Bs_max_coeff' : +0.,
    }

    TEST_SIZE = 100
    LEARN_ITER = 10

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    config.log_device_placement=False

    case_list = ['case9', 'case14', 'case30', 'case118']

    error_NR_list = {}
    error_DC_list = {}
    error_GNS_list = {}
    time_NR_list = {}
    time_DC_list = {}
    time_GNS_list = {}
    
    timestr = time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    expe_dir = 'expe_' + train_case + '_' + timestr
    try:
        os.stat(os.path.join(path, expe_dir))
    except:
        os.mkdir(os.path.join(path, expe_dir))
    
    print("Building model that will be trained on "+train_case)
    tf.reset_default_graph()
    sess = tf.Session(config=config)
    expe_path = os.path.join(path, expe_dir)
    with tf.variable_scope('GNS', reuse=tf.AUTO_REUSE):
        model = GNS(train_case, train_params, model_params, noise_params, expe_path, sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print("    Learning on "+train_case+"...")
    for learn_iter in tqdm.tqdm(range(LEARN_ITER)):
        # Apply a gradient descent
        sess.run(model.opt_op)
        # Store the loss
        model.store_summary(learn_iter)
    print("    The model trained on "+train_case+" is ready to be tested!")

    all_p_nr = {}
    all_p_dc = {}
    all_p_gns = {}
    time_NR_list = {}
    time_DC_list = {}
    time_GNS_list = {}
    for test_case in case_list:
        print("    Testing on "+test_case+"...")
        model.change_default_power_grid(case=test_case)

        # Sample
        all_p_nr[test_case] = None
        all_p_dc[test_case] = None
        all_p_gns[test_case] = None
        
        time_NR_list[test_case] = []
        time_DC_list[test_case] = []
        time_GNS_list[test_case] = []
        
        i=0
        while i < TEST_SIZE:
            print(i)
            
            # Sample a power grid
            gens, buses, lines, p_gen, pft, ptf, qft, qtf = sess.run([model.gens, model.buses, model.lines, model.p_gen,
                model.p_from_to[str(model_params['num_updates'])], 
                model.p_to_from[str(model_params['num_updates'])],
                model.q_from_to[str(model_params['num_updates'])], 
                model.q_to_from[str(model_params['num_updates'])],
            ])
            sample_id=0
            
            
            
            model.input_data['gen'][:,1] = p_gen[str(model_params['num_updates'])][sample_id]*100
            model.input_data['gen'][:,5] = gens['Vg'][sample_id]
            model.input_data['bus'][:,2] = buses['Pd'][sample_id]
            model.input_data['bus'][:,3] = buses['Qd'][sample_id]
            model.input_data['bus'][:,4] = buses['Gs'][sample_id]
            model.input_data['bus'][:,5] = buses['Bs'][sample_id]
            model.input_data['branch'][:,2] = lines['r'][sample_id]
            model.input_data['branch'][:,3] = lines['x'][sample_id]
            model.input_data['branch'][:,4] = lines['b'][sample_id]
            model.input_data['branch'][:,8] = lines['ratio'][sample_id] 
            model.input_data['branch'][:,9] = lines['angle'][sample_id] * 180/np.pi
            
            i += 1
            try:
                start = time.time()
                a_nr = runpf(model.input_data)
                time_NR_list[test_case].append(time.time()-start)
                
            except:
                print('did not converge')
                i -= 1
                continue
            if a_nr[0]['success'] == 0:
                print('did not converge')
                i -= 1
                continue
                
            # Now try with DC approx
            start = time.time()
            a_dc = rundcpf(model.input_data)
            time_DC_list[test_case].append(time.time()-start)
                
            # Get flows from GNS
            p_from_gns = pft[sample_id]*100
            p_to_gns = ptf[sample_id]*100
            q_from_gns = qft[sample_id]*100
            q_to_gns = qtf[sample_id]*100
            
            if all_p_gns[test_case] is None:
                all_p_gns[test_case] = np.r_[p_from_gns, p_to_gns]
            else:
                all_p_gns[test_case] = np.c_[all_p_gns[test_case], np.r_[p_from_gns, p_to_gns]]
            
            # Get flows from Newton-Raphson
            p_from_nr = a_nr[0]['branch'][:,13]
            q_from_nr = a_nr[0]['branch'][:,14]
            p_to_nr = a_nr[0]['branch'][:,15]
            q_to_nr = a_nr[0]['branch'][:,16]
            
            if all_p_nr[test_case] is None:
                all_p_nr[test_case] = np.r_[p_from_nr, p_to_nr]
            else:
                all_p_nr[test_case] = np.c_[all_p_nr[test_case], np.r_[p_from_nr, p_to_nr]]
            
            # Get flows from DC approx
            p_from_dc = a_dc[0]['branch'][:,13]
            q_from_dc = a_dc[0]['branch'][:,14]
            p_to_dc = a_dc[0]['branch'][:,15]
            q_to_dc = a_dc[0]['branch'][:,16]
            
            if all_p_dc[test_case] is None:
                all_p_dc[test_case] = np.r_[p_from_dc, p_to_dc]
            else:
                all_p_dc[test_case] = np.c_[all_p_dc[test_case], np.r_[p_from_dc, p_to_dc]]
    
            # Now try with GNS
            v_time, theta_time = sess.run([model.v, model.theta])
            start = time.time()
            v_time, theta_time = sess.run([model.v, model.theta])
            time_GNS_list[test_case].append((time.time()-start)/train_params['num_samples'])
            

    
    np.save( os.path.join(storing_dir,expe_dir+"_all_p_nr.npy"), all_p_nr )
    np.save( os.path.join(storing_dir,expe_dir+"_all_p_gns.npy"), all_p_gns )
    np.save( os.path.join(storing_dir,expe_dir+"_all_p_dc.npy"), all_p_dc )
    np.save( os.path.join(storing_dir,expe_dir+"_time_nr.npy"), time_NR_list )
    np.save( os.path.join(storing_dir,expe_dir+"_time_gns.npy"), time_GNS_list )
    np.save( os.path.join(storing_dir,expe_dir+"_time_dc.npy"), time_DC_list )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='Id of the GPU you want to use')
    parser.add_argument('--case', help='IEEE case on which you want to train')
    args = parser.parse_args()
    
    # Activating only the desired GPU
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    # Build a directory for the experiments if one doesn't exist already
    path = 'experiments/'
    try:
        os.stat(path)
    except:
        os.mkdir(path)
        
    storing_dir = 'experiments/'+args.case+'_time'
    if os.path.exists(storing_dir):
        shutil.rmtree(storing_dir)
    os.makedirs(storing_dir)
    
    NUM_MODELS = 1
    
    for i in range(NUM_MODELS):
        main(args.case, storing_dir)

import argparse
import os
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
import time

from configs import ParseParams

from shared.decode_step import RNNDecodeStep
from model.attention_agent import RLAgent

def load_task_specific_components():
    from VRP.vrp_utils import DataGenerator,Env,reward_func
    from VRP.vrp_attention import AttentionVRPActor,AttentionVRPCritic
    AttentionActor = AttentionVRPActor
    AttentionCritic = AttentionVRPCritic
    return DataGenerator, Env, reward_func, AttentionActor, AttentionCritic

def main(args, prt):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    DataGenerator, Env, reward_func, AttentionActor, AttentionCritic = load_task_specific_components()

    dataGen = DataGenerator(args)
    dataGen.reset()
    env = Env(args)

    agent = RLAgent(args, prt, env, dataGen, reward_func, AttentionActor, AttentionCritic, is_train=args['is_train'])
    agent.Initialize(sess)

    # agent.inference(args['infer_type'])

    prt.print_out('Training started ...')

    #
    for step in range(args['n_train']):
        summary = agent.run_train_step()
        print summary
if __name__ == "__main__":
    args, prt = ParseParams()
    # Random
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        prt.print_out("# Set random seed to %d" % random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    tf.reset_default_graph()

    main(args, prt)

import numpy as np
import random
import pickle
from HO_initialize import Compute
from globalval import GlobalVal as gv
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from Qlearning_model import QLearningTable
QL = QLearningTable(actions=gv.ACTION1)
move_points = 30000
if __name__ == '__main__':
    '''训练网络'''
    for episode in range(100):
        # J = 1+episode
        print('episode= ',episode)
        v,dir,HOM,TTT=Compute.reset()# Initialize user and HCPs
        n=0
        while n < move_points:
            if n % 50 == 0: # Choose direction every
                # random.seed(J)#Set fixed paths for easy comparison
                # J += 5
                dir= random.choice(gv.directions)
            next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
            gv.prior_UE_x.append(next_x),gv.prior_UE_y.append(next_y)
            #Status information
            old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
            # Determine if handovering is taking place
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v), gv.DIR.append(dir)
                old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR, prior_ALL_distance[tar_BS_ID], v, dir]
                action = QL.choose_action(old_state)#Selection of actions according to the Q table
                NEW_HOM = gv.ACTION1[action]
                # Calculate within TTT whether sinr is satisfied with A3
                TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP ,ALL_RSRP =\
                    Compute.satisfyTTT(TTT,gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,tar_BS_ID)
                # Determining the type of HO
                Compute.label(TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP,NEW_HOM,tar_BS_ID)
                # Statistical handover rate
                PPHOR,TLHOR,TEHOR = Compute.radio(gv.INDEX)
                #Simulate the next state
                next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
                #Calculating the reward function
                reward =Compute.compute_reward(throughput,TLHOR,TEHOR,PPHOR)
                #Update State
                new_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR, prior_ALL_distance[tar_BS_ID], v, dir]
                new_state = np.array(new_state)
                #Updating the Q-value
                QL.learn(old_state, action, reward, new_state)
                old_state = new_state
                n=n+1
            else:
                n=n+1
                continue
        #Save Q-Table
        with open("../q_learning.pickle", "wb") as f:
            pickle.dump(dict(
                QL.q_table), f)
            print("model saved")

    '''Qlearning算法优化'''
    for episode in range(100):
        # J = 1+episode
        print('episode= ',episode)
        v,dir,HOM,TTT=Compute.reset()# Initialize user and HCPs
        n=0
        while n < move_points:
            if n % 50 == 0: # Choose direction every 5 seconds
                # random.seed(J)#Set fixed paths for easy comparison
                # J += 5
                dir= random.choice(gv.directions)
            next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
            gv.prior_UE_x.append(next_x),gv.prior_UE_y.append(next_y)
            #State information
            old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR, prior_ALL_distance[tar_BS_ID], v, dir]
            # Determine if handovering is taking place
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v), gv.DIR.append(dir)
                old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR, prior_ALL_distance[tar_BS_ID], v, dir]
                action = QL.choose_action(old_state)#Selection of actions according to the Q table
                NEW_HOM = gv.ACTION1[action]
                # Calculate within TTT whether sinr is satisfied with A3
                TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP ,ALL_RSRP =\
                    Compute.satisfyTTT(TTT,gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,tar_BS_ID)
                # Determining the type of HO
                Compute.label(TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP,NEW_HOM,tar_BS_ID)
                # Statistical handover rate
                PPHOR,TLHOR,TEHOR = Compute.radio(gv.INDEX)
                #Simulate the next state
                next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
                #Calculating the reward function
                reward =Compute.compute_reward(throughput,TLHOR,TEHOR,PPHOR)
                #Update State
                new_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR, prior_ALL_distance[tar_BS_ID], v, dir]
                new_state = np.array(new_state)
                #Updating the Q-value
                QL.learn(old_state, action, reward, new_state)
                old_state = new_state
                n=n+1
            else:
                n=n+1
                continue
        del gv.connect_BS[-1]
        Compute.save_csv(gv.connect_BS, gv.target_BS, gv.S_RSRP, gv.T_RSRP, gv.S_SINR, gv.T_SINR, gv.HO_start_distance,
                         gv.V, gv.DIR, gv.INDEX, episode)  # Save feature data
    Compute.merge_all_csv()  # merge data
        
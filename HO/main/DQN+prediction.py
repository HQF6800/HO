import numpy as np
import random
import matplotlib.pyplot as plt
from HO_initialize import *
from globalval import GlobalVal as gv
import tensorflow as tf
from xgboost.sklearn import XGBClassifier
tf.compat.v1.disable_eager_execution()
from DQN_model import DeepQNetwork
import pickle
# RL = DeepQNetwork(len(gv.ACTION1), 6)
RL = DeepQNetwork(len(gv.ACTION1), 9)
move_points = 30000

if __name__ == '__main__':
    '''训练网络'''
    for episode in range(200):
        J = 1+episode
        print('episode= ', episode)
        v,dir,HOM,TTT=Compute.reset()# Initialize user and HCPs
        n=0
        REWARD=[]
        while n < move_points:
            if n % 60 == 0:  # Choose direction
                random.seed(J)#Set fixed paths for easy comparison
                J += 5
                # v = np.random.randint(2, 8)  # Random selection of speed
                dir = random.choice(gv.directions)
            next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
            gv.prior_UE_x.append(next_x),gv.prior_UE_y.append(next_y)
            # State information
            old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
            # Determine if handovering is taking place
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR),gv.Throughput.append(throughput)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v), gv.DIR.append(dir)
                old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
                # old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                #              prior_SSINR, prior_TSINR]
                old_state = np.array(old_state)
                #Selection of actions according to State
                action = RL.choose_action(old_state)
                NEW_HOM = gv.ACTION1[action]
                '''handover_prediction'''
                # Loading XGBClassifier Models
                '''XGB'''
                clf_XGB = XGBClassifier()
                clf_XGB.load_model("../model/model.xgb")
                # old_state is processed as a 2-dimensional array
                data_reshaped  = old_state.reshape(-1, 1).T
                prediction_type = clf_XGB.predict(data_reshaped)
                '''MLP'''
                # with open("../model/MLPClassifier.pkl", "rb") as f:
                    # MLP = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = MLP.predict(data_reshaped)
                '''DecisionTreeClassifier'''
                # with open("../model/DecisionTreeClassifier.pkl", "rb") as f:
                #     DT = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = DT.predict(data_reshaped)
                '''LogisticRegression'''
                # with open("../model/LogisticRegression.pkl", "rb") as f:
                #     LR = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = LR.predict(data_reshaped)
                '''AdaBoostClassifier'''
                # with open("../model/AdaBoostClassifier.pkl", "rb") as f:
                #     AdaBoost = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = AdaBoost.predict(data_reshaped)
                '''KNeighborsClassifier'''
                # with open("../model/KNeighborsClassifier.pkl", "rb") as f:
                #     KNN = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = KNN.predict(data_reshaped)
                '''RandomForestClassifier'''
                # with open("../model/RandomForestClassifier.pkl", "rb") as f:
                #     RF = pickle.load(f)
                # data_reshaped  = old_state.reshape(-1, 1).T
                # prediction_type = RF.predict(data_reshaped)

                # Selection of HCPs based on handover type
                NEW_HOM,NWE_TTT=Compute.HCP_choose(prediction_type, HOM, NEW_HOM,TTT)
                # Calculate within TTT whether sinr is satisfied with A3
                TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP ,ALL_RSRP =\
                    Compute.satisfyTTT(NWE_TTT,gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,tar_BS_ID)
                # Determining the type of HO
                Compute.label(TTT_SSINR ,TTT_TSINR
                              ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP,NEW_HOM,tar_BS_ID)
                # Statistical handover rate
                PPHOR,TLHOR,TEHOR = Compute.radio(gv.INDEX)
                # Simulate the next state
                next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
                # Calculating the reward function
                reward =Compute.compute_reward(throughput,TLHOR,TEHOR,PPHOR)
                # REWARD.append(reward)
                # Update State
                new_state =[gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
                # new_state =[gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                #              prior_SSINR, prior_TSINR]
                new_state = np.array(new_state)
                # Experience playback storage
                RL.store_transition(old_state,NEW_HOM,reward,new_state)
                old_state = new_state
                n=n+1
            else:
                n=n+1
                continue
        RL.learn()#Updating the Q-value
        RL.save_net()#Saving Networks
        # average_reward = sum(REWARD) / len(REWARD) if REWARD else 0
        # gv.Average_reward.append(average_reward)
        del gv.connect_BS[-1]
        Compute.save_csv(gv.connect_BS, gv.target_BS, gv.S_RSRP, gv.T_RSRP, gv.S_SINR, gv.T_SINR, gv.HO_start_distance,
                         gv.V, gv.DIR, gv.INDEX, episode)  # Save feature data
    Compute.merge_all_csv()  # merge data
    print('PPHOR,TLHOR,TEHOR',PPHOR,TLHOR,TEHOR)
    # RL.plot_cost(episode)#plot the loss function
    # '''plot reward'''
    # Compute.save_reward()
    # plt.plot(np.arange(len(gv.Average_reward)), gv.Average_reward, label='reward')
    # plt.ylabel('Reward')
    # plt.xlabel('Train episodes')
    # plt.legend(loc='upper right', frameon=True, prop={'size': 8})
    # plt.show()
    '''DQN算法优化'''
    # for episode in range(200):
    #     J = 1+episode
    #     print('episode= ', episode)
    #     v, dir, HOM, TTT = Compute.reset()  # Initialize user and HCPs
    #     n = 0
    #     REWARD = []
    #     while n < move_points:
    #         if n % 60 == 0:  # Choose direction every 5 seconds
    #             random.seed(J)#Set fixed paths for easy comparison
    #             J += 5
    #             dir = random.choice(gv.directions)
    #         next_x, next_y, prior_ALL_distance, prior_ALL_RSRP, tar_BS_ID, prior_SSINR, prior_TSINR, throughput = \
    #             Compute.step(gv.prior_UE_x[-1], gv.prior_UE_y[-1], v, dir, gv.connect_BS, HOM)
    #         gv.prior_UE_x.append(next_x), gv.prior_UE_y.append(next_y)
    #         # State information
    #         old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
    #                          prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
    #         # Determine if handovering is taking place
    #         if tar_BS_ID != gv.connect_BS[-1]:
    #             gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
    #             gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR), gv.Throughput.append(throughput)
    #             gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
    #             gv.V.append(v), gv.DIR.append(dir)
    #             old_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
    #                          prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
    #             old_state = np.array(old_state)
    #             # Selection of actions according to State
    #             action = RL.choose_action(old_state)
    #             NEW_HOM = gv.ACTION1[action]
    #             '''handover_prediction'''
    #             # Loading XGBClassifier Models
    #             clf_XGB = XGBClassifier()
    #             clf_XGB.load_model("../model/model.xgb")
    #             # old_state is processed as a 2-dimensional array
    #             data_reshaped = old_state.reshape(-1, 1).T
    #             prediction_type = clf_XGB.predict(data_reshaped)
    #             # Selection of HCPs based on handover type
    #             NEW_HOM, NEW_TTT = Compute.HCP_choose(prediction_type, HOM, NEW_HOM, TTT)
    #             # Calculate within TTT whether sinr is satisfied with A3
    #             TTT_SSINR, TTT_TSINR, TTT_TRSRP, TTT_SRSRP, ALL_RSRP = Compute.satisfyTTT(NEW_TTT, gv.prior_UE_x[-1], gv.prior_UE_y[-1],v, dir, gv.connect_BS,tar_BS_ID)
    #             # Determining the type of HO
    #             Compute.label(TTT_SSINR, TTT_TSINR
    #                           , TTT_TRSRP, TTT_SRSRP, ALL_RSRP, NEW_HOM, tar_BS_ID)
    #             # Statistical handover rate
    #             PPHOR, TLHOR, TEHOR = Compute.radio(gv.INDEX)
    #             # Simulate the next state
    #             next_x, next_y, prior_ALL_distance, prior_ALL_RSRP, tar_BS_ID, prior_SSINR, prior_TSINR, throughput = \
    #                 Compute.step(gv.prior_UE_x[-1], gv.prior_UE_y[-1], v,dir, gv.connect_BS, HOM)
    #             # Calculating the reward function
    #             reward = Compute.compute_reward(throughput, TLHOR, TEHOR, PPHOR)
    #             REWARD.append(reward)
    #             # Update State
    #             new_state = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
    #                          prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
    #             new_state = np.array(new_state)
    #             # Experience playback storage
    #             RL.store_transition(old_state, NEW_HOM, reward, new_state)
    #             old_state = new_state
    #             n = n + 1
    #         else:
    #             n = n + 1
    #             continue
    #     average_reward = sum(REWARD) / len(REWARD) if REWARD else 0
    #     gv.Average_reward.append(average_reward)
    #     del gv.connect_BS[-1]
    #     Compute.save_csv(gv.connect_BS, gv.target_BS, gv.S_RSRP, gv.T_RSRP, gv.S_SINR, gv.T_SINR, gv.HO_start_distance,
    #                      gv.V, gv.DIR, gv.INDEX, episode)  # Save feature data
    # Compute.merge_all_csv()  # merge data
    # print('PPHOR,TLHOR,TEHOR',PPHOR,TLHOR,TEHOR)
    # RL.plot_cost(episode)
    # '''plot reward'''
    # Compute.save_reward()
    # plt.plot(np.arange(len(gv.Average_reward)), gv.Average_reward, label='reward')
    # plt.ylabel('Reward')
    # plt.xlabel('Train episodes')
    # plt.legend(loc='upper right', frameon=True, prop={'size': 8})
    # plt.show()

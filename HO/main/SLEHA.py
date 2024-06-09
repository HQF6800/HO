import numpy as np
import random
from HO_initialize import Compute
from globalval import GlobalVal as gv
import tensorflow as tf
from xgboost.sklearn import XGBClassifier
tf.compat.v1.disable_eager_execution()

move_points = 50000

if __name__ == '__main__':
    '''训练网络'''
    for episode in range(100):
        # J = 1 + episode
        print('episode= ', episode)
        v, dir, HOM, TTT = Compute.reset()# Initialize user and HCP parameters
        n = 0
        while n < move_points:
            if n % 50 == 0: # Choose direction
                # random.seed(J)#Set fixed paths for easy comparison
                # J += 5
                dir = random.choice(gv.directions)
            next_x, next_y, prior_ALL_distance, prior_ALL_RSRP, tar_BS_ID, prior_SSINR, prior_TSINR, throughput = \
                Compute.step(gv.prior_UE_x[-1], gv.prior_UE_y[-1], v, dir, gv.connect_BS, HOM)
            gv.prior_UE_x.append(next_x), gv.prior_UE_y.append(next_y)
            old_state = [tar_BS_ID, gv.connect_BS[-1], prior_SSINR, prior_TSINR
                , prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID], prior_ALL_distance[tar_BS_ID]]
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR), gv.Throughput.append(throughput)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v), gv.DIR.append(dir)
                '''handover_prediction'''
                # Load the XGBClassifier model
                clf_XGB = XGBClassifier()
                clf_XGB.load_model("./model/model.xgb")
                # feature processed as a 2-dimensional array
                feature = [gv.connect_BS[-1], tar_BS_ID, prior_ALL_RSRP[gv.connect_BS[-1]], prior_ALL_RSRP[tar_BS_ID],
                             prior_SSINR, prior_TSINR,prior_ALL_distance[tar_BS_ID], v, dir]
                feature=np.array(feature)
                data_reshaped = feature.reshape(-1, 1).T
                prediction_type = clf_XGB.predict(data_reshaped)
                NEW_HOM,NEW_TTT=Compute.SLEHA_HCPchooose(prediction_type)
                # print(NEW_HOM)
                # Calculate within TTT whether sinr is satisfied with A33
                TTT_SSINR, TTT_TSINR, TTT_TRSRP, TTT_SRSRP, ALL_RSRP = \
                    Compute.satisfyTTT(NEW_TTT, gv.prior_UE_x[-1], gv.prior_UE_y[-1], v, dir, gv.connect_BS,
                                       tar_BS_ID)
                # Determine the type of HO
                Compute.label(TTT_SSINR, TTT_TSINR
                              , TTT_TRSRP, TTT_SRSRP, ALL_RSRP, NEW_HOM, tar_BS_ID)
                PPHOR, TLHOR, TEHOR = Compute.radio(gv.INDEX)#Statistical handover rate
                n = n + 1
            else:
                n = n + 1
                continue
        Compute.save_csv(gv.connect_BS, gv.target_BS, gv.S_RSRP, gv.T_RSRP, gv.S_SINR, gv.T_SINR,
                             gv.HO_start_distance, gv.V, gv.DIR, gv.INDEX, episode)  # Save feature data
    Compute.merge_all_csv()  # merge data
    # print('PPHOR,TLHOR,TEHOR',PPHOR,TLHOR,TEHOR)
    # print(gv.Throughput)


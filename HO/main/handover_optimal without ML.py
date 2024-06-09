from random import choice
import numpy as np
import random
import math
from HO_initialize import Compute
from globalval import GlobalVal as gv

move_points = 30000
if __name__ == '__main__':
    for episode in range(100):
        # J = 1+episode
        print('episode= ', episode)
        v,dir,HOM,TTT=Compute.reset()# Initialize user and HCPs
        n=0
        while n < move_points:
            if n % 60 == 0:  # Choose direction
                # random.seed(J)#Set fixed paths for easy comparison
                # J += 5
                dir = random.choice(gv.directions)
            next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput  =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)
            gv.prior_UE_x.append(next_x),gv.prior_UE_y.append(next_y)
            # Determine if handovering is taking place
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR),gv.Throughput.append(throughput)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v), gv.DIR.append(dir)
                # Calculate within TTT whether SINR is satisfied with A3
                TTT_SSINR_late ,TTT_TSINR_late ,TTT_TRSRP_late ,TTT_SRSRP_late ,ALL_RSRP_late =\
                    Compute.satisfyTTT(TTT,gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,tar_BS_ID)
                # Determine the type of HO
                handover_type=Compute.label(TTT_SSINR_late ,TTT_TSINR_late ,TTT_TRSRP_late ,TTT_SRSRP_late,ALL_RSRP_late,HOM,tar_BS_ID)
                HOM, TTT = Compute.HCP_optimal(handover_type, HOM, TTT)
                # Statistical handover rate
                PPHOR, TLHOR, TEHOR = Compute.radio(gv.INDEX)
                n=n+1
            else:
                n=n+1
                continue
        del gv.connect_BS[-1]
        # Compute.save_csv(gv.connect_BS, gv.target_BS, gv.S_RSRP, gv.T_RSRP, gv.S_SINR, gv.T_SINR, gv.HO_start_distance,gv.V, gv.DIR, gv.INDEX, episode)  # Save feature data
    # Compute.merge_all_csv()  # merge data
    print('PPHOR,TLHOR,TEHOR', PPHOR, TLHOR, TEHOR)
    # print(gv.Throughput)


from random import choice
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from HO_initialize import *
from globalval import GlobalVal as gv
plt.rc('text', usetex=True)
move_points = 30000
if __name__ == '__main__':
    for episode in range(200):
        J = 1+episode
        print('episode= ', episode)
        v,dir,HOM,TTT=Compute.reset()# Initialize user and HCPs
        n=0
        while n < move_points:
            if n % 60 == 0:  # Choose direction every 6 seconds
                random.seed(J)#Set fixed paths for easy comparison
                J += 5
                # v = np.random.randint(2, 8)  # Random selection of speed
                dir= random.choice(gv.directions)
            next_x,next_y,prior_ALL_distance,prior_ALL_RSRP,tar_BS_ID,prior_SSINR,prior_TSINR,throughput =\
                    Compute.step(gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,HOM)#Statistics for each step of the user's update
            gv.prior_UE_x.append(next_x),gv.prior_UE_y.append(next_y)
            # Determine if handovering is taking place
            if tar_BS_ID != gv.connect_BS[-1]:
                gv.target_BS.append(tar_BS_ID), gv.HO_start_distance.append(prior_ALL_distance[tar_BS_ID])
                gv.T_SINR.append(prior_TSINR), gv.S_SINR.append(prior_SSINR),gv.Throughput.append(throughput)
                gv.S_RSRP.append(prior_ALL_RSRP[gv.connect_BS[-1]]), gv.T_RSRP.append(prior_ALL_RSRP[tar_BS_ID])
                gv.V.append(v),gv.DIR.append(dir)
                # Calculate within TTT whether sinr is satisfied with A3
                TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP ,ALL_RSRP =\
                    Compute.satisfyTTT(TTT,gv.prior_UE_x[-1],gv.prior_UE_y[-1],v,dir,gv.connect_BS,tar_BS_ID)
                # Determine the type of HO
                Compute.label(TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP,HOM,tar_BS_ID)
                # Statistical handover rate
                PPHOR, TLHOR, TEHOR = Compute.radio(gv.INDEX)
                n=n+1
            else:
                n=n+1
                continue
        del gv.connect_BS[-1]
        '''Plot user walk path'''
        # x, y = zip(*BS_position)
        # fig1, axs = plt.subplots(1, figsize=(8, 8))
        # axs.scatter(x, y, marker='^', s=10, facecolors='r', label='macro_BS')
        # axs.scatter(gv.prior_UE_x[0], gv.prior_UE_y[0], facecolors='#FFA510', label='start')
        # axs.scatter(gv.prior_UE_x[-1], gv.prior_UE_y[-1], facecolors='red', label='end')
        # for i, txt in enumerate(BS_position):
        #     plt.text(txt[0], txt[1], f'({i + 1})', fontsize=8, color='b', ha='left', va='bottom')
        # axs.set_xlim(0, 2000)
        # axs.set_ylim(0, 2000)
        # axs.plot(gv.prior_UE_x, gv.prior_UE_y, '#00FFD8', linestyle='-', label='user path')
        # axs.grid(color='k', linestyle='-', linewidth=0.08)
        # axs.set_xlabel(r' x{m}', fontsize=10)
        # axs.set_ylabel(r' y{m}', fontsize=10)
        # axs.legend(loc='upper left', frameon=True, prop={'size': 8})
        # # plt.savefig('..\\figures\\Throughput_path.png', format='png', dpi=300)
        # dic = {'x':gv.prior_UE_x,'y':gv.prior_UE_y}
        # df=pd.DataFrame(dic)
        # df.to_csv('..\\data\\' +'user walk path.csv',index = False)
        # axs.set_rasterized(True)
        # plt.grid()
        # plt.show()
        Compute.save_csv(gv.connect_BS,gv.target_BS,gv.S_RSRP,gv.T_RSRP,gv.S_SINR,gv.T_SINR,gv.HO_start_distance,gv.V,gv.DIR,gv.INDEX,episode)#Save feature data
    Compute.merge_all_csv()#merge data
    print('PPHOR,TLHOR,TEHOR',PPHOR,TLHOR,TEHOR)
    # print(gv.Throughput)

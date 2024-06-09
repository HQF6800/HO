import numpy as np
import random
import math
import os
import glob
from globalval import GlobalVal as gv
import pandas as pd

# BS deployments
rect_x1, rect_y1 = 0, 0#Define the coordinates of the lower left corner of the area
rect_x2, rect_y2 = 2000, 2000#Define the coordinates of the upper right corner of the area
BS_position = [(x, y) for x in range(100,rect_x2,400) for y in range(100,rect_y2,200)]# 50 BS deployed at equal spacing
# BS_position = [(x, y) for x in range(100,rect_x2,200) for y in range(100,rect_y2,200)]# 100 BS deployed at equal spacing
directions = [0,45,90,135,180,225,270,315]#Define the direction of motion

class Compute(object):
    def reset():
        # initialization all parameters
        gv.clr(gv.prior_UE_x,gv.prior_UE_y,gv.connect_BS,gv.S_SINR,gv.T_SINR,gv.S_RSRP,
               gv.T_RSRP,gv.target_BS,gv.HO_start_distance,gv.V,gv.DIR,gv.INDEX)
        HOM = gv.ACTION1[6]
        TTT = gv.ACTION2[2]
        # v = 1# v=5m/s,6m/s,v=8m/s
        v = 2 # v=10m/s，v=20m/s,v=30m/s
        dir= random.choice(directions)
        # initial position
        # x, y = random.randint(rect_x1, rect_x2), random.randint(rect_y1, rect_y2)
        x, y = 1000, 1000#Fixed user starting position
        gv.prior_UE_x.append(x)
        gv.prior_UE_y.append(y)
        ALL_distance = Compute.compute_distance(gv.prior_UE_x[-1],gv.prior_UE_y[-1])
        ALL_RSRP = Compute.compute_RSRP(ALL_distance)
        BS_ID = Compute.choose_BS_use_max_RSRP(ALL_RSRP)
        gv.connect_BS.append(BS_ID)
        # SINR = Compute.computer_SINR(ALL_distance,ALL_RSRP,BS_ID)
        return v,dir,HOM,TTT


    def compute_distance(location_x,location_y):
        '''Calculate the distance of the UE from all BSs'''
        distance_of_BS_AND_UE= []
        for i in range(len(BS_position)):
            dis = math.sqrt(pow(location_x-BS_position[i][0],2) + pow(location_y-BS_position[i][1],2))
        # print('The distance from BS'+str(i)+'is:',distance_S)
            distance_of_BS_AND_UE.append(dis)
        return distance_of_BS_AND_UE
    
    def compute_RSRP(ALL_distance):
        ''' Calculate the RSRP between the UE and all base stations.'''
        R =[]
        sigma_sf = 2
        chi_sigma = np.random.normal(0,sigma_sf) #Shadow fading
        for i in range(len(BS_position)):       
            path_loss = 36.7 * np.log10(ALL_distance[i]) + 22.7 + 26 * np.log10(3.5)#path_loss model
            P1 = 30 -path_loss-chi_sigma # BS transmit power 30dBm
            R.append(P1)
        return R
    
    def choose_BS_use_max_RSRP(ALL_RSRP): # First selection of BS
        '''RSRP maximum value to select the connected BS'''
        BS_ID = ALL_RSRP.index(max(ALL_RSRP))
        # print('The BS for the UE connection is: '+str(BS_ID)+'No. BS')
        return BS_ID
        
    def random_move(location_x,location_y,v,dir):
        while True:
            # Calculate next position
            next_x = location_x +  v*math.cos(math.radians(dir))
            next_y = location_y +  v*math.sin(math.radians(dir))
            # Determine if you have reached the boundary and need to reselect the direction of movement
            if next_x < rect_x1 or next_x > rect_x2 or next_y < rect_y1 or next_y > rect_y2:
                dir= random.choice(directions)
                continue           
            break
        return next_x, next_y
    
    def choose_BS_use_A3(ALL_RSRP,connect_BS,HOM):
        #Handovering based on A3 trigger event
        candidate_BS = []
        for i in range(len(BS_position)):
            if ALL_RSRP[connect_BS[-1]] <=ALL_RSRP[i] - HOM:
                candidate_BS.append(i)
        if len(candidate_BS)==0:
            candidate_BS.append(connect_BS[-1])
        k=[]
        #Selection of candidate base stations
        for bs in candidate_BS:
            k.append(ALL_RSRP[bs])
        max_value = max(k)
        max_indices = [i for i, v in enumerate(k) if v == max_value]
        #There are two candidate base stations for maximum RSRP
        if len(max_indices) > 1:
            random_index = random.choice(max_indices)
        else:
            random_index = max_indices[0]
        BS_ID = candidate_BS[random_index]
        # print(candidate_BS)
        return BS_ID
    
    def computer_one_RSRP(one_dis):
        #Calculate RSRP for individual base stations
        sigma_sf = 2
        chi_sigma = np.random.normal(0,sigma_sf) #Shadow fading
        path_loss = 36.7 * np.log10(one_dis) + 22.7 + 26 * np.log10(3.5)-chi_sigma
        P = 30 - path_loss
        return P
    
    def computer_SINR(ALL_distance,RSRP,BS_ID):
        #Calculate the SINR of the user and the serving base station
        dis= ALL_distance[:]
        del dis[BS_ID]
        ganrao = []# Interference is RSRP between the other cell and the UE
        ganraodBm = []
        for d in dis:
            if d<=700:
                a = Compute.computer_one_RSRP(d) # dBm
                ganraodBm.append(a)
                interference = pow(10,a/10) # dB converted to a numeric value
                ganrao.append(interference)
        I = sum(ganrao)
        N = 3.98e-11 # mw -174dBm/Hz = 3.98e-18(mw/Hz) B =10MHz 
        S = pow(10,RSRP[BS_ID]/10) # 
        sinr = 10*np.log10((S/(I+N))) # dB
        return sinr
    def computer_throughput(ALL_distance,RSRP,BS_ID):
        #Calculate the throughput of the user and the serving base station
        dis= ALL_distance[:]
        del dis[BS_ID]
        ganrao = []
        ganraodBm = []
        for d in dis:
            if d<=700:
                a = Compute.computer_one_RSRP(d) # dBm
                ganraodBm.append(a)
                interference = pow(10,a/10) # dB converted to a numeric value
                ganrao.append(interference)
        I = sum(ganrao)
        N = 3.98e-11 # mw -174dBm/Hz = 3.98e-18(mw/Hz) B =10MHz
        S = pow(10,RSRP[BS_ID]/10) #
        throughput=10*np.log2(1+(S/(I+N)))
        return throughput

    def step(x,y,v,dir,connect_BS,HOM):
        #Statistics for each step of the user's information
        next_x,next_y = Compute.random_move(x,y,v,dir)# Calculate the user's next position
        ALL_distance = Compute.compute_distance(next_x,next_y)# Calculate user-base station distance
        ALL_RSRP = Compute.compute_RSRP(ALL_distance)#Calculate all RSRPs between user and base station
        target_BS_ID = Compute.choose_BS_use_A3(ALL_RSRP,connect_BS,HOM)# Calculate A3 event selection base station IDs
        SSINR = Compute.computer_SINR(ALL_distance,ALL_RSRP,connect_BS[-1])# Calculate current service base station SINR
        TSINR = Compute.computer_SINR(ALL_distance,ALL_RSRP,target_BS_ID)# Calculate target base station SINR
        Throughput = Compute.computer_throughput(ALL_distance, ALL_RSRP, connect_BS[-1])  # Computing current services base station throughput
        return next_x,next_y,ALL_distance,ALL_RSRP,target_BS_ID,SSINR,TSINR,Throughput
    
    def one_distance(BS_position,UE_position_x,UE_position_y):  # Output is m
        dis = math.sqrt(pow(UE_position_x - BS_position[0],2) + pow(UE_position_y - BS_position[1],2) + pow(10-1.5,2))
        return dis
    
    def TTT_step(x,y,v,dir,connect_BS,tar_BS_ID):
        # Statistics for each step of the user's information
        next_x,next_y = Compute.random_move(x,y,v,dir)
        gv.prior_UE_x.append(next_x) ,gv.prior_UE_y.append(next_y)
        ALL_distance = Compute.compute_distance(next_x,next_y)
        ALL_RSRP = Compute.compute_RSRP(ALL_distance)
        TBS_RSRP = ALL_RSRP[tar_BS_ID]#RSRP of the target base station
        SBS_RSRP = ALL_RSRP[connect_BS[-1]]#RSRP of the service base station
        SBS_SINR = Compute.computer_SINR(ALL_distance,ALL_RSRP,connect_BS[-1])#SINR of the service base station
        TBS_SINR = Compute.computer_SINR(ALL_distance,ALL_RSRP,tar_BS_ID)#SINR of the target base station
        return TBS_RSRP,SBS_RSRP,SBS_SINR,TBS_SINR,ALL_RSRP
    

    def satisfyTTT(TTT,x,y,v,dir,connect_BS,tar_BS_ID):
        #Determine whether a user satisfies the TTT
        TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP= [],[],[],[]
        for i in range(TTT):
            TBS_RSRP,SBS_RSRP,SBS_SINR,TBS_SINR,ALL_RSRP =\
                    Compute.TTT_step(x,y,v,dir,connect_BS,tar_BS_ID)
            TTT_SSINR.append(SBS_SINR) ,TTT_TSINR.append(TBS_SINR)
            TTT_TRSRP.append(TBS_RSRP) ,TTT_SRSRP.append(SBS_RSRP)
        return TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP

    def HCP_optimal(Index,HOM,TTT):
        #Optimization HCPs techniques without ML
        if Index == 0:
            pass
        elif Index == 1:
            HOM+=0.5
            TTT+=1
        elif Index == 2:
            HOM-=0.5
            TTT-=2
        elif Index == 3:
            HOM+=0.5
            TTT+=1
        return HOM,TTT

    '''0:SHO, 1:PPHO 2:TLHO 3:TEHO 4:Anomaly handover'''
    def label(TTT_SSINR ,TTT_TSINR ,TTT_TRSRP ,TTT_SRSRP,ALL_RSRP,HOM,tar_BS_ID):
        #Determining the type of handovering
        ave_TTT_TRSRP=np.mean(TTT_TRSRP)
        ave_TTT_SRSRP=np.mean(TTT_SRSRP)
        if TTT_SSINR[-1] > gv.SINR_threshhold:
            if ave_TTT_TRSRP > ave_TTT_SRSRP + HOM: #Determine whether A3 is satisfied
                if TTT_TSINR[-1] < gv.SINR_threshhold:
                    # Immediately after the successful HO, RLF occurs with the target BS and connects to the source BS, Too Early HO.
                    BS_ID2 = Compute.choose_BS_use_max_RSRP(ALL_RSRP)
                    if BS_ID2 == gv.connect_BS[-1]:
                        gv.connect_BS.append(BS_ID2)
                        gv.INDEX.append(3)
                    else:
                        gv.connect_BS.append(BS_ID2)
                        gv.INDEX.append(2)
                else: # Judging SHOs and PPHOs
                    gv.connect_BS.append(tar_BS_ID)
                    if len(gv.connect_BS) >= 3:
                        # After handovering to another base station and then handovering back to the original base station, PPHO
                        if gv.connect_BS[-1] == gv.connect_BS[-3] and gv.connect_BS[-1] != gv.connect_BS[-2]:
                            gv.INDEX.append(1) #PPHO
                        else:
                            gv.INDEX.append(0) # SHO
                    else:
                        gv.INDEX.append(0)
            else:
                BS_ID3 = Compute.choose_BS_use_max_RSRP(ALL_RSRP)
                gv.connect_BS.append(BS_ID3)
                gv.INDEX.append(3)

        else: # TLHO
            BS_ID1 = Compute.choose_BS_use_max_RSRP(ALL_RSRP)
            gv.connect_BS.append(BS_ID1)
            if BS_ID1 == tar_BS_ID: 
                gv.INDEX.append(2)
            else:
                gv.INDEX.append(4) 
    def HCP_choose(prediction_type,HOM,NEW_HOM,TTT):
        #Selection of HCPs based on handover type
        if prediction_type[0] == 0:
            return HOM,TTT
        elif prediction_type[0] == 1 or prediction_type[0] == 3:
            # NEW_TTT = 7
            if NEW_HOM >= 7:
                NEW_HOM = 2.5
                # return NEW_HOM,NEW_TTT
                return NEW_HOM, TTT
            else:
                # return NEW_HOM,NEW_TTT
                return NEW_HOM, TTT
        elif prediction_type[0] == 2:
            NEW_TTT = 3
            if NEW_HOM <= 3:
                NEW_HOM = 7.5
                return NEW_HOM,NEW_TTT
            else:
                return NEW_HOM,NEW_TTT
    def SLEHA_HCPchooose(prediction_type):
        #Supervised Learning Based Adjustment for Extreme HCPs
        if  prediction_type[0] == 0:
            return 0,5
        if prediction_type[0] == 1 or prediction_type[0] == 3:
                return 0,5
        if prediction_type[0] == 2:
                return 10,9

    def radio(lst):
        #Statistical PPHOR,TLHOR,TEHOR rate
        NHO = len(lst)
        SHO = lst.count(0)
        PPHO = lst.count(1)
        TLHO = lst.count(2)
        TEHO = lst.count(3)
        PPHOR = PPHO/ NHO
        TLHOR = TLHO/ NHO
        TEHOR = TEHO/ NHO
        return PPHOR,TLHOR,TEHOR

    def compute_reward(throughput,TLHOR,TEHOR,PPHOR):
        #reward function
        w1,w2,w3,w4=1,1,1,1#weighting factor
        # reward=w1*((throughput-1)/195)-w2*TLHOR-w3*TEHOR-w4*PPHOR #consideration of throughput
        reward =- w2 * TLHOR - w3 * TEHOR - w4 * PPHOR #No consideration of throughput
        return reward

    def save_csv(connect_BS,target_BS,S_RSRP,T_RSRP,S_SINR,T_SINR,HO_start_distance,v,dir,INDEX,episode):
        #Feature Saving
        dic = {'connect_BS':connect_BS,'target_BS':target_BS,'S_RSRP':S_RSRP,'T_RSRP':T_RSRP,'S_SINR':S_SINR,'T_SINR':T_SINR,
               'HO_start_distance':HO_start_distance,'V':v,'dir':dir,'INDEX':INDEX}
        df=pd.DataFrame(dic)
        df.to_csv('..\\data\\hebing\\data' + str(episode) + '.csv',index = False)
    
    def merge_all_csv():
        #Merge data
        os.chdir(r'../data/hebing') #Feature Saving Address
        csv_list = glob.glob('data*.csv')
        print('Total %s of CSV files'%len(csv_list))
        print('Starting the merger......')
        merge_path = 'merge.csv'
        for i in range(len(csv_list)):  # Loop over csv files in the same folder
            path = 'data' + str(i) + '.csv'
            with open(path, 'rb') as fr:
                data = fr.read()
                with open(merge_path, 'ab') as f:  # Save the result as merge.csv
                    f.write(data)
            print(i)
        qc = pd.read_csv(merge_path, header=None)
        datalist = qc.drop_duplicates()
        datalist.to_csv(merge_path, index=False, header=False)
        print('The merger is complete!')

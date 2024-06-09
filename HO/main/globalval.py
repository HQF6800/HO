class GlobalVal:
    prior_UE_x=[] #  The x-coordinate of the UE
    prior_UE_y=[] # The y-coordinate of the UE
    S_SINR = [] # SINR with Service BS
    T_SINR = [] # SINR with target BS
    S_RSRP = [] #RSRP with Service BS
    T_RSRP = [] #RSRP with target BS
    connect_BS = []#Service BS
    target_BS = [] # Target BS
    HO_start_distance = []#Distance between user and target base station
    Throughput=[]
    SINR_threshhold=-12
    INDEX=[] #Handover Type Label
    cost_DQN = []
    cost_Qlearning = []
    V=[]
    DIR=[]
    directions= [0,45,90,135,180,225,270,315] # Eight directions
    ACTION1 = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10] # HOM
    ACTION2 = [1,3,5,7,9] # TTT
    def clr(a,b,c,d,e,f,g,h,i,j,k,l):
        del a,b,c,d,e,f,g,h,i,j,k,l

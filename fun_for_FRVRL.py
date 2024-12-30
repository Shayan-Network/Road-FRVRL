from fuzzy_funs import fuzzy_direction
from fuzzy_funs import fuzzy_light
from fuzzy_funs import fuzzy_noe
from fuzzy_funs import fuzzy_vel
from fuzzy_funs import fuzzy_non
from fuzzy_funs import fuzzy_evod
from fuzzy_funs import fuzzy_turn

from fire_action import fire_action_direction
from fire_action import fire_action_light
from fire_action import fire_action_noe
from fire_action import fire_action_vel
from fire_action import fire_action_non
from fire_action import fire_action_evod
from fire_action import fire_action_turn
import numpy as np

def action_cal(MAT,R,s,x,out,nedge,control):
    max_X={}

    ########## for direction
    print("1")
    if len(s['direct'])>0:
        for i in range(len(MAT['diection_f3'])):
            x['direct'][i] = fire_action_direction(R['direct'], s['direct'][i])
            out['direct'][i]=fuzzy_direction(s['direct'][i], R['direct'])
        
        domydirec=[np.zeros(len(MAT['diection_f3'])) for _ in range(25)]
        for i in range(len(MAT['diection_f3'])):
            for j in range(25):
                domydirec[j][i]=x['direct'][i][j]
        domydirec=np.array(domydirec)
        max_X['direct']=np.zeros(25)
        for i in range(25):
            if len(MAT['diection_f3'])!=0:
                max_X['direct'][i]=np.max(domydirec[i])

        action2=out['direct'] # this action is for direction change
    else:
        max_X['direct']=np.zeros(25)
        action2=[]
            
    
    #################for light
    print("2")
    if len(s['light'])>0:
        for i in range(len(MAT['greenside'])):
            x['light'][i] = fire_action_light(R['light'], s['light'][i])
            out['light'][i]=fuzzy_light(s['light'][i], R['light'])
    
        domylight=[np.zeros(len(MAT['greenside'])) for _ in range(25)]
        for i in range(len(MAT['greenside'])):
            for j in range(25):
                domylight[j][i]=x['light'][i][j]
        domylight=np.array(domylight)
        max_X['light']=np.zeros(25)
        for i in range(25):
            if len(domylight)!=0:
                max_X['light'][i]=np.max(domylight[i])

        action1=out['light'] # this action is for direction change
    else:
        max_X['light']=np.zeros(25)
        action1=[]
################noe###############
    print("3")
    if len(s['noe'])>0:
        for i in range(len(MAT['noef1'])):
            x['noe'][i] = fire_action_noe(R['noe'], s['noe'][i])
            out['noe'][i]=fuzzy_noe(s['noe'][i], R['noe'])
        
        domynoe=[np.zeros(len(MAT['noef1'])) for _ in range(75)]
        for i in range(len(MAT['noef1'])):
            for j in range(75):
                domynoe[j][i]=x['noe'][i][j]
        domynoe=np.array(domynoe)
        max_X['noe']=np.zeros(75)
        for i in range(75):
            if len(MAT['noef1'])!=0:
                max_X['noe'][i]=np.max(domynoe[i])

        action4=out['noe'] # this action is for direction change
    else:
        max_X['noe']=np.zeros(75)
        action4=[]
    ################for vel################################################
    print("4")
    action3=50*np.ones(nedge)
    for i in  range(len(control)):
        x['vel'][i] = fire_action_vel(R['vel'], s['vel'][i])
        out['vel'][i]=fuzzy_vel(s['vel'][i], R['vel'])
        action3[control[i]]=out['vel'][i] # this action is for direction change
            
        
    
    domyvel=[np.zeros(len(control)) for _ in range(25)]
    for i in range(len(control)):
            for j in range(25):
                domyvel[j][i]=x['vel'][i][j]
    domyvel=np.array(domyvel)
    max_X['vel']=np.zeros(25)
    for i in range(25):
        max_X['vel'][i]=np.max(domyvel[i])

   #############non######################################################
    print("5") 
    if len(s['nostop'])>0:
        for i in range(len(MAT['nostop'])):
            if isinstance(s['nostop'][i][0], (list, np.ndarray)):
                s['nostop'][i]=np.zeros(3)
                s['nostop'][i][0]=0
                s['nostop'][i][1]=100
                s['nostop'][i][2]=22
 
            x['nostop'][i] = fire_action_non(R['nostop'], s['nostop'][i])
            out['nostop'][i]=fuzzy_non(s['nostop'][i], R['nostop'])
        
        domynon=[np.zeros(len(MAT['nostop'])) for _ in range(75)]
        for i in range(len(MAT['nostop'])):
            for j in range(75):
                domynon[j][i]=x['nostop'][i][j]
        domynon=np.array(domynon)
        max_X['nostop']=np.zeros(75)
        
        for i in range(75):
            if len(MAT['nostop'])!=0:
                max_X['nostop'][i]=np.max(domynon[i])

        action5=out['nostop'] # this action is for direction change
    else:
        max_X['nostop']=np.zeros(75)
        action5=[]
        
    ##########Time_limit###############################################################
    print("6")
    if len(s['limit'])>0:
        for i in range(len(MAT['limit'])):
            x['limit'][i] = fire_action_evod(R['limit'], s['limit'][i])
            out['limit'][i]=fuzzy_evod(s['limit'][i], R['limit'])
        
        domyevod=[np.zeros(len(MAT['limit'])) for _ in range(10)]
        for i in range(len(MAT['limit'])):
            for j in range(10):
                domyevod[j][i]=x['limit'][i][j]
        domyevod=np.array(domyevod)
        max_X['limit']=np.zeros(10)
        for i in range(10):
            if len(MAT['limit'])!=0:
                max_X['limit'][i]=np.max(domyevod[i])

        action7=out['limit'] # this action is for direction change
    else:
        max_X['limit']=np.zeros(10)
        action7=[]
        
    #################noturn###############################################################
    print("7")   
    if len(s['noturn'])>0:
        for i in range(len(MAT['noturn'])):
            x['noturn'][i] = fire_action_turn(R['noturn'], s['noturn'][i])
            out['noturn'][i]=fuzzy_turn(s['noturn'][i], R['noturn'])
            
        domyturn=[np.zeros(len(MAT['noturn'])) for _ in range(125)]
        for i in range(len(MAT['noturn'])):
            for j in range(125):
                domyturn[j][i]=x['noturn'][i][j]
        domyturn=np.array(domyturn)
        max_X['noturn']=np.zeros(125)
        for i in range(125):
            if len(MAT['noturn'])!=0:
                max_X['noturn'][i]=np.max(domyturn[i])

        action6=out['noturn'] # this action is for direction change
    else:
        max_X['noturn']=np.zeros(125)
        action6=[]    
    
    return action2,action3,max_X,action4,action1,action5,action7,action6

import numpy as np

def H(input):
    h = []
    
    if len(input) == 2:
        for i in range(input[0] ):
            for j in range(input[1]):
                h.append([i, j])
    
    elif len(input) == 3:
        for i in range(input[0]):
            for j in range(input[1]):
                for k in range(input[2]):
                    h.append([i, j, k])
    
    return np.array(h)  # Convert the list to a numpy array for better handling

import numpy as np
## disterbuting the reward 
def r_disterb(x, rr):
    # Initialize an empty list for direction
    rd_direction = np.zeros(len(x['direct']))  # Using a NumPy array for performance
    rd_light= np.zeros(len(x['light']))
    rd_noe=np.zeros(len(x['noe']))
    rd_vel=np.zeros(len(x['vel']))
    rd_non=np.zeros(len(x['nostop']))
    rd_evod=np.zeros(len(x['limit']))
    rd_noturn=np.zeros(len(x['noturn']))


    # Calculate rd.direction
    for i in range(len(x['direct'])):
        x2 = x['direct'][i]
        rd_direction[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
        
    for i in range(len(x['light'])):
        x2 = x['light'][i]
        rd_light[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
    for i in range(len(x['nostop'])):
        x2 = x['noe'][i]
        rd_noe[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))

    for i in range(len(x['vel'])):
        x2 = x['vel'][i]
        rd_vel[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
    
    for i in range(len(x['nostop'])):
        x2 = x['nostop'][i]
        rd_non[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
   


    for i in range(len(x['limit'])):
        x2 = x['limit'][i]
        rd_evod[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
   
    for i in range(len(x['noturn'])):
        x2 = x['noturn'][i]
        rd_noturn[i] = (x2 * rr) / (np.sum(x['light'])+ np.sum(x['direct'])+np.sum(x['noe'])+np.sum(x['vel'])+np.sum(x['nostop'])+np.sum(x['limit'])+np.sum(x['noturn']))
         
    return rd_direction,rd_light,rd_noe,rd_vel,rd_non,rd_evod,rd_noturn

import numpy as np

def rule_making(Q,epsilon2):
    exp_exp = abs(np.random.rand())
    ep=epsilon2
    direction=np.zeros(25)
    light=np.zeros(25)
    noe=np.zeros(75)
    vel=np.zeros(25)
    non=np.zeros(75)
    evod=np.zeros(10)
    noturn=np.zeros(125)

    direction = [int(x) for x in direction]
    light = [int(x) for x in light]
    noe = [int(x) for x in noe]
    vel = [int(x) for x in vel]
    non = [int(x) for x in non]
    evod = [int(x) for x in evod]
    noturn = [int(x) for x in noturn]

    

    

    for i in range(25):
        if exp_exp<=ep:
            light[i] = np.random.randint(0, 4)
            direction[i] = np.random.randint(0, 2)
            vel[i] = np.random.randint(0, 5)
        else:
            light[i]= np.argmax(Q['light'][i])  # Get the index of the maximum value
            direction[i]= np.argmax(Q['direct'][i])  # Get the index of the maximum value
            vel[i]= np.argmax(Q['vel'][i])  # Get the index of the maximum value

    for i in range(75):
        if exp_exp<=ep:
            noe[i] = np.random.randint(0, 2)
            non[i] = np.random.randint(0, 2)

        else:
            noe[i]= np.argmax(Q['noe'][i])  # Get the index of the maximum value
            non[i] =np.argmax(Q['nostop'][i])

    for i in range(10):
        if exp_exp<=ep:
            evod[i] = np.random.randint(0, 3)

        else:
            evod[i]= np.argmax(Q['limit'][i])  # Get the index of the maximum value

    for i in range(125):
        if exp_exp<=ep:
            noturn[i] = np.random.randint(0, 6)

        else:
            noturn[i]= np.argmax(Q['noturn'][i])  # Get the index of the maximum value



    R1d=H([5, 5])
    R1noe=H([5,3,5])
    R1evod=H([2,5])
    R1noturn=H([5,5,5])

    R2d=[[] for _ in range(25)]
    R2l=[[] for _ in range(25)]
    R2noe=[[] for _ in range(75)]
    R2vel=[[] for _ in range(25)]
    R2non=[[] for _ in range(75)]
    R2evod=[[] for _ in range(10)]
    R2noturn=[[] for _ in range(125)]


    R={}

    for i in range(25):
       R2d[i].append(R1d[i][0])
       R2d[i].append(R1d[i][1])
       R2d[i].append(direction[i])
       R2l[i].append(R1d[i][0])
       R2l[i].append(R1d[i][1])
       R2l[i].append(light[i])
       R2vel[i].append(R1d[i][0])
       R2vel[i].append(R1d[i][1])
       R2vel[i].append(vel[i])
    for i in range(75):
       R2noe[i].append(R1noe[i][0])
       R2noe[i].append(R1noe[i][1])
       R2noe[i].append(R1noe[i][2])
       R2noe[i].append(noe[i])
       R2non[i].append(R1noe[i][0])
       R2non[i].append(R1noe[i][1])
       R2non[i].append(R1noe[i][2])
       R2non[i].append(non[i])
    for i in range(10):
       R2evod[i].append(R1evod[i][0])
       R2evod[i].append(R1evod[i][1])
       R2evod[i].append(evod[i])

    for i in range(125):
       R2noturn[i].append(R1noturn[i][0])
       R2noturn[i].append(R1noturn[i][1])
       R2noturn[i].append(R1noturn[i][2])
       R2noturn[i].append(noturn[i])


    R['direct']=np.array(R2d)
    R['light']=np.array(R2l)
    R['noe']=np.array(R2noe)
    R['vel']=np.array(R2vel)
    R['nostop']=np.array(R2non)
    R['limit']=np.array(R2evod)
    R['noturn']=[]
    R['noturn']=np.array(R2noturn)

    return R


import numpy as np
def states(MAT,edge_dencity,edge_v,G,edge_car_con,trafic_light_matrix1,dis,more_way,control):
    from fun_for_graphmining import find_edge_index
    

    state={}
    edge_v=np.array(edge_v)
    edge_dencity=np.array(edge_dencity)

    ############FOR DIRECTION
    di=np.zeros(len(MAT['diection_f3']))
    di = [int(x) for x in di]
    if len(MAT['diection_f3'])>0:
        for i in range(len(MAT['diection_f3'])):
            A= find_edge_index(G, MAT['diection_f3'][i][0], MAT['direction_f4'][i][0])
            di[i]=A
            

        AS=[np.zeros(2) for _ in range(len(MAT['diection_f3']))]
        for i in range (len(MAT['diection_f3'])):
            A=0
            A=edge_dencity[di[i]]
            if isinstance(edge_dencity[di[i]], np.ndarray) or isinstance(edge_v[di[i]], np.ndarray):
                AS[i]=[min(edge_dencity[di[i]]),min(edge_v[di[i]])]
                print("A")
            else:
                AS[i]=[edge_dencity[di[i]],edge_v[di[i]]]
    else:
        AS=[]
    ################# FOR light
    
    LS=[[0,0] for _ in range(len(trafic_light_matrix1[1]))]
    if len(MAT['greenside'])>0:
        for i in range (len(trafic_light_matrix1[1])):
            s=0
            for j in range(len(trafic_light_matrix1[1][i])):      
                s=s+edge_car_con[trafic_light_matrix1[1][i][j]]
                LS[i][0]=s
            
        s=0
        for i in range (len(trafic_light_matrix1[1])):
            s=0
            for j in range(len(trafic_light_matrix1[0][i])):      
                s=s+edge_car_con[trafic_light_matrix1[0][i][j]]
                LS[i][1]=s
            
    else:
        LS=[]
   #################noe############################### 
    dnoe=np.zeros(len(MAT['noef1']))
    dnoe = [int(x) for x in dnoe]
    if len(MAT['noef1'])>0:
        for i in range(len(MAT['noef1'])):
            A= find_edge_index(G, MAT['noef1'][i][0], MAT['noef2'][i][0])
            dnoe[i]=A
            

        noS=[np.zeros(3) for _ in range(len(MAT['noef1']))]
        for i in range (len(MAT['noef1'])):
            if isinstance(edge_dencity[dnoe[i]], np.ndarray) or isinstance(edge_v[dnoe[i]], np.ndarray) or isinstance(dis[dnoe[i]], np.ndarray):
                noS[i]=[min(edge_dencity[dnoe[i]]),min(dis[dnoe[i]]),min(edge_v[dnoe[i]])]
            else:
                noS[i]=[(edge_dencity[dnoe[i]]),(dis[dnoe[i]]),(edge_v[dnoe[i]])]
                
    else:
        noS=[]
    
    ##############vel################################
    dvel=np.zeros(len(control))
    dvel = [int(x) for x in dvel]

    vels=[np.zeros(2) for _ in range(len(control))]
    for i in range (len(control)):
        vels[i]=[edge_dencity[i],edge_v[i]]  
 
    #####################non###########################################
    dnon=np.zeros(len(MAT['nostop']))
    dnon = [int(x) for x in dnon]
    if len(MAT['nostop'])>0: 
        for i in range(len(MAT['nostop'])):
            A=0
            A= find_edge_index(G, MAT['nostop'][i][0], MAT['nostop'][i][1])
            dnon[i]=A
            

        nonS=[np.zeros(3) for _ in range(len(MAT['nostop']))]
        for i in range (len(MAT['nostop'])):
            if isinstance(edge_dencity[dnon[i]], np.ndarray) or isinstance(edge_v[dnon[i]], np.ndarray) or isinstance(dis[dnon[i]], np.ndarray):
                nonS[i]=[min(edge_dencity[dnon[i]]),min(dis[dnon[i]]),min(edge_v[dnon[i]])]
            else:
                nonS[i]=[(edge_dencity[dnon[i]]),(dis[dnon[i]]),(edge_v[dnon[i]])]
                
            

    else:
        nonS=[]
    #####################limit##################################################
    devod=np.zeros(len(MAT['limit']))
    devod = [int(x) for x in devod]
    if len(MAT['limit'])>0:
        for i in range(len(MAT['limit'])):
            A= find_edge_index(G, MAT['limit'][i][0], MAT['limit'][i][1])
            if A!=None:
                devod[i]=A

        nevod=[np.zeros(2) for _ in range(len(MAT['limit']))]
        for i in range (len(MAT['limit'])):
            if isinstance(edge_dencity[devod[i]], np.ndarray) or isinstance(more_way[devod[i]], np.ndarray) :
                nevod[i]=[min(more_way[devod[i]]),min(edge_dencity[devod[i]])]
            else:
                nevod[i]=[(more_way[devod[i]]),(edge_dencity[devod[i]])]
                
    else:
        nevod=[]
###################no_turn#################################        
    denotur=np.zeros(len(MAT['noturn']))
    denotur = [int(x) for x in denotur]
    if len(MAT['noturn'])>0:
        for i in range(len(MAT['noturn'])):
            A=0
            B=0
            C=0
            A= find_edge_index(G, MAT['noturn'][i][0][0], MAT['noturn'][i][0][1])
            B= find_edge_index(G, MAT['noturn'][i][1][0], MAT['noturn'][i][1][1])
            C= find_edge_index(G, MAT['noturn'][i][2][0], MAT['noturn'][i][2][1])
            
            denotur[i]=[A,B,C]

        nturn=[np.zeros(3) for _ in range(len(MAT['noturn']))]
        for i in range (len(MAT['noturn'])):
            if isinstance(edge_dencity[denotur[i][0]], np.ndarray) or isinstance(edge_dencity[denotur[i][1]], np.ndarray) or isinstance(edge_dencity[denotur[i][2]], np.ndarray):
                nturn[i]=[min(edge_dencity[denotur[i][0]]),min(edge_dencity[denotur[i][1]]),min(edge_dencity[denotur[i][2]])]
            else:
                nturn[i]=[(edge_dencity[denotur[i][0]]),(edge_dencity[denotur[i][1]]),(edge_dencity[denotur[i][2]])]

    else:
        nturn=[]
    return AS,LS,noS,vels,nonS,nevod,nturn

import numpy as np
from fun_for_graphmining import find_edge_index
import numpy as np
from scipy.stats import zscore

def W_r( arivel_time,edge_car_con, error_turn, edge_v,MAT,G,trafic_light_matrix1,edge_dencity):
    # Sort edge_car_con in descending order and get the indices
    conjunction=[]
    conjunction=edge_car_con/(edge_v+1)
    z_scores = zscore(edge_dencity)
    threshold = 1
    outlier_indices_of_dencity= [i for i in range(len(edge_dencity)) if z_scores[i] > threshold and edge_dencity[i]>np.mean(edge_dencity) ]
    print(len(outlier_indices_of_dencity))
    A=[]
    A = np.sort(edge_car_con)[::-1]  # Sort edge_car_con in descending order
    W = np.sum(edge_car_con[outlier_indices_of_dencity]) /len(edge_car_con[outlier_indices_of_dencity])
    
    # Calculate V as the average of the corresponding top 15 elements from edge_v
    V = np.sum(edge_v[outlier_indices_of_dencity]) /len(edge_v[outlier_indices_of_dencity])
    con=np.sum(conjunction[outlier_indices_of_dencity]) /len(conjunction[outlier_indices_of_dencity])
    dennn=np.sum(edge_dencity[outlier_indices_of_dencity]) /len(edge_dencity[outlier_indices_of_dencity])
    # Calculate r using the same formula
    arivel=(arivel_time/(1000*len(edge_v)))
    reward_global = -(5*con) - (.001*np.sum(error_turn))-500*dennn-50*(arivel_time/(1000*len(edge_v)))
   
#############local rewards############################################
############FOR DIRECTION##############################
    alfa=100
    di=[0 for _ in range(len(MAT['diection_f3']))]
    di = [int(x) for x in di]

    for i in range(len(MAT['diection_f3'])):
        A= find_edge_index(G, MAT['diection_f3'][i][0], MAT['direction_f4'][i][0])
        if A!=None:
            di[i]=A
        
    direction_reward=0
    if len(di)>0:
        direction_reward=-alfa*(np.sum(edge_car_con[di])/(np.sum(edge_v[di])+1))
        direction_reward=direction_reward
    else:
        direction_reward=0
##############light#################################################
    light_reward=0
    if len(trafic_light_matrix1[1])>0:
        for i in range (len(trafic_light_matrix1[1])):
            for j in range(len(trafic_light_matrix1[1][i])):
                light_reward=light_reward+edge_car_con[trafic_light_matrix1[1][i][j]]/(edge_v[trafic_light_matrix1[1][i][j]]+1)
            
        for i in range (len(trafic_light_matrix1[0])):
            for j in range(len(trafic_light_matrix1[0][i])):
                light_reward=light_reward+edge_car_con[trafic_light_matrix1[0][i][j]]/(edge_v[trafic_light_matrix1[0][i][j]]+1)
        light_reward=(-alfa*light_reward)
    else:
        light_reward=0
       #################noe############################### 
    dnoe=[]

    dnoe=[0 for _ in range(len(MAT['noef1']))]
    dnoe = [int(x) for x in dnoe]
    for i in range(len(MAT['noef1'])):
        A=[]        
        A= find_edge_index(G, MAT['noef1'][i][0], MAT['noef2'][i][0])
        if A!=None:
            dnoe[i]=A
    if len(dnoe)>0:  
        noe_reward=-alfa*(np.sum(edge_car_con[dnoe])/(np.sum(edge_v[dnoe]))+1)
        noe_reward=noe_reward
        if np.isnan(noe_reward)==1 or np.isinf(noe_reward)==1:
            noe_reward=0    
    else:
        noe_reward=0
    ##############vel################################
    dvel=np.zeros(len(edge_v))
    dvel = [int(x) for x in dvel]
    vel_reward=-alfa*(np.sum(edge_car_con)/(np.sum(edge_v))+1)
    vel_reward=vel_reward

 
    #####################non###########################################
    dnon=[]
    dnon=[0 for _ in range(len(MAT['nostop']))]
    dnon = [int(x) for x in dnon]

    for i in range(len(MAT['nostop'])):
        A=[]
        A= find_edge_index(G, MAT['nostop'][i][0], MAT['nostop'][i][1])
        if A!=None:
            dnon[i]=A
    if len(dnon)>0:
        non_reward=-alfa*np.sum(edge_car_con[dnon])/(np.sum(edge_v[dnon])+1)
        non_reward=non_reward
    else:
        non_reward=0
    
    #####################limit##################################################
    devod=[]
    devod=[0 for _ in range(len(MAT['limit']))]
    devod = [int(x) for x in devod]

    for i in range(len(MAT['limit'])):
        A=[]

        A= find_edge_index(G, MAT['limit'][i][0], MAT['limit'][i][1])
        if A!=None:
            devod[i]=A
    if len(devod)>0:
        evod_reward=-alfa*np.sum(edge_car_con[devod])/(np.sum(edge_v[devod])+1)
        evod_reward=evod_reward
    else:
        evod_reward=0
    ###############################noturn##############################################
    denotur=np.zeros(len(MAT['noturn']))
    denotur = [int(x) for x in denotur]
    noturn_reward=0
    for i in range(len(MAT['noturn'])):
        A=0
        B=0
        C=0
        A= find_edge_index(G, MAT['noturn'][i][0][0], MAT['noturn'][i][0][1])
        B= find_edge_index(G, MAT['noturn'][i][1][0], MAT['noturn'][i][1][1])
        C= find_edge_index(G, MAT['noturn'][i][2][0], MAT['noturn'][i][2][1])
        
        denotur[i]=[A,B,C]

        if len(denotur[i])>0:
            noturn_reward=noturn_reward+np.sum(edge_car_con[denotur[i]])/(np.sum(edge_v[denotur[i]])+1)
        noturn_reward=(-alfa*noturn_reward)
    I=1
    return arivel,dennn,con, reward_global, V, I,evod_reward,non_reward,vel_reward,noe_reward,light_reward,direction_reward,noturn_reward,W 



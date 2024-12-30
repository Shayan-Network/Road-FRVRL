####importing needed libraries 
import networkx as nx
import numpy as np
import random as rann
import matplotlib.pyplot as plt
from functions_for_prep import path2
from functions_for_prep import simulate_traffic
from env import env
from fun_for_FRVRL import rule_making
from fun_for_FRVRL import action_cal
from fun_for_FRVRL import states
from fun_for_FRVRL import W_r
from fun_for_FRVRL import r_disterb
from fun_for_graphmining import MAT_determin
from graph_model import graph_model
from fun_for_graphmining import initialize_q_table 
from fun_for_graphmining import choose_action
from fun_for_graphmining import action_handeling_mat
from fun_for_graphmining import reward_handeling_mat
from fun_for_graphmining import find_edge_index

### geting the city model
edgematrix,nedge,G,distances=graph_model()
### data for input determining of city
car_ratio=50
dt=1
simulation_time=50
arrival_times = simulate_traffic(car_ratio,.01, simulation_time)
num_cars=len(arrival_times)
inputnodes=list(G.nodes)
outputnodes=list(G.nodes)
safepath=[1,2]
car_ege_path=path2(G,num_cars,safepath,inputnodes,outputnodes,edgematrix,nedge) # it gives pathes for each car in edge mode
## defining empty matrixes
MAT = {}
MAT['diection_f3'] = []
MAT['direction_f4'] = []
MAT['noef1'] =[]
MAT['noef2'] = []
MAT['nostop'] = []
MAT['limit'] = []
MAT['redside']= []
MAT['greenside'] =[]
MAT['noturn'] =[]
action1=[]
action2=[]
action4=[]
action7=[]
action6 = []
action5=[]
action3=50*np.ones(len(edgematrix))
### env data for input determining of city
if len(MAT['greenside'])>0 :
    redside=[[] for _ in range(len(MAT['redside']))]
    for i in range(len(MAT['redside'])):
        for j in range(len(MAT['redside'][i])):
            redindex=0
            redindex=find_edge_index(G, MAT['redside'][i][j][0], MAT['redside'][i][j][1])
            redside[i].append(redindex[0])

    greenside=[[] for _ in range(len(MAT['greenside']))]
    for i in range(len(MAT['greenside'])):
        for j in range(len(MAT['greenside'][i])):
            greenindex=0
            greenindex=find_edge_index(G, MAT['greenside'][i][j][0], MAT['greenside'][i][j][1])
            greenside[i].append(greenindex[0])

    trafic_light_matrix1=[]
    trafic_light_matrix1 = [redside, greenside]  # edge of traffic light on green, col are edges
else:
    trafic_light_matrix1=[[],[]]
pos2,edge_dencity, edge_car_con, edge_v,error_turn,trafic_light_matrix1,more_way,arivel_time= env(distances,inputnodes,outputnodes,safepath,action1, action2, action3, action4, action5, action6, action7, G, num_cars, simulation_time, car_ege_path, arrival_times, dt,MAT,edgematrix,2,trafic_light_matrix1)
inter_p=.0 
control=[]
sorted=np.argsort(edge_dencity)
control=sorted
con_prob=(max(edge_dencity)-min(edge_dencity))/2 ## for conjunction
###############################in put determinig phase: first RL model##########################################################
## initial data for first RL model
PP,MAT=MAT_determin(car_ratio,dt,simulation_time,con_prob,inter_p,arrival_times,G, edge_car_con, edge_v,MAT,control,edge_dencity)
if len(MAT['greenside'])>0 :
    redside=[[] for _ in range(len(MAT['redside']))]
    for i in range(len(MAT['redside'])):
        for j in range(len(MAT['redside'][i])):
            redindex=0
            redindex=find_edge_index(G, MAT['redside'][i][j][0], MAT['redside'][i][j][1])
            redside[i].append(redindex)

    greenside=[[] for _ in range(len(MAT['greenside']))]
    for i in range(len(MAT['greenside'])):
        for j in range(len(MAT['greenside'][i])):
            greenindex=0
            greenindex=find_edge_index(G, MAT['greenside'][i][j][0], MAT['greenside'][i][j][1])
            greenside[i].append(greenindex)

    trafic_light_matrix1=[]
    trafic_light_matrix1 = [redside, greenside]  # edge of traffic light on green, col are edges
else:
    trafic_light_matrix1=[[],[]]
    
#MAT_determin(car_ratio,dt,simulation_time,con_prob,inter_p,arrival_times,G, edge_car_con, edge_v,MAT)

q_table = initialize_q_table(PP)
epi2max=500
epi2=0
alpha = 0.5
gamma = 0.8
epsilon = 0.2
## empty matrixes
rewards1 = {state: 1*rann.uniform(-1, 1) for state in PP}
cost=[]
rew=[]
errorcount=[]
## initial data for second RL model
gamma2=.8
alpha2=.5
pathcheck=20
maxepi=1
car_ratio=50
dt=1
simulation_time=50
arrival_times = simulate_traffic(car_ratio,.01, simulation_time)
num_cars=len(arrival_times)
inputnodes=list(G.nodes)
outputnodes=list(G.nodes)
safepath=[1,2]
car_ege_path=path2(G,num_cars,safepath,inputnodes,outputnodes,edgematrix,nedge) # it gives pathes for each car in edge mode
s={}
Q={}
x={}
Q['direct']=np.random.rand(25,2)
Q['light']=np.random.rand(25,4)
Q['noe']=np.random.rand(75,2)
Q['vel']=np.random.rand(25,5)
Q['nostop']=np.random.rand(75,2)
Q['limit']=np.random.rand(10,3)
Q['noturn']=np.random.rand(125,6)

secepi=0
innnn=-np.inf
## start of first RL
while epi2<epi2max:
    print("start graph mining RL")
    MAT['diection_f3'] = []
    MAT['direction_f4'] = []
    MAT['noef1'] =[]
    MAT['noef2'] = []
    MAT['nostop'] = []
    MAT['limit'] = []
    epi2=epi2+1
    action=[]
    for i in range(len(PP)):
        action.append(int(choose_action(PP[i], q_table,.1)))
    MAT=action_handeling_mat(PP,edgematrix,action,MAT)
 ## start of second RL
    for epi in range(maxepi):
        print("start FRVRL")

        ## inital states
        if epi==0 :
            x['direct']=[[] for _ in range(len(MAT['diection_f3']))]
            x['light']=[[] for _ in range(len(MAT['greenside']))]
            x['noe']=[[] for _ in range(len(MAT['noef1']))]
            nedge = G.number_of_edges()
            x['vel']=[[] for _ in range(len(control))]
            x['nostop']=[[] for _ in range(len(MAT['nostop']))]
            x['limit']=[[] for _ in range(len(MAT['limit']))]
            x['noturn']=[[] for _ in range(len(MAT['noturn']))]
            s['noe'] = []
            s['light'] =[]
            s['direct'] = []
            s['vel']=[]
            s['nostop']=[]
            s['limit']=[]
            s['noturn']=[]
            ds,LS,noes,vels,nonS,nevod,nturn=states(MAT,edge_dencity,edge_v,G,edge_car_con,trafic_light_matrix1,distances,more_way,control)
            s['noe'] = noes
            s['light'] = LS
            s['direct'] = ds
            s['vel']=vels
            s['nostop']=nonS
            s['limit']=nevod
            s['noturn']=nturn
#
        print("epiRL2=",epi,"epiRL1=",epi2)
        print("making fuzzy rules")
        secepi=secepi+1
        R=rule_making(Q,.1)
        max_X={}
        out={}
        out['direct']=np.zeros(len(MAT['diection_f3']))
        out['light']=np.zeros(len(MAT['greenside']))
        out['noe']=np.zeros(len(MAT['noef1']))
        out['vel']=np.zeros(nedge)
        out['nostop']=np.zeros(len(MAT['nostop']))
        out['limit']=np.zeros(len(MAT['limit']))
        out['noturn']=np.zeros(len(MAT['noturn']))

        ##geting the actions and stese for RL2
        print("calculating env action")
        action2,action3,max_X,action4,action1,action5,action7,action6=action_cal(MAT,R,s,x,out,nedge,control)   
        print("start env")
        pos2,edge_dencity, edge_car_con, edge_v,error_turn,trafic_light_matrix1,more_way,arivel_time= env(distances,inputnodes,outputnodes,safepath,action1, action2, action3, action4, action5, action6, action7, G, num_cars, simulation_time, car_ege_path, arrival_times, dt,MAT,edgematrix,pathcheck,trafic_light_matrix1)                         
        ds,LS,noes,vels,nonS,nevod,nturn=[],[],[],[],[],[],[]
        ds,LS,noes,vels,nonS,nevod,nturn=states(MAT,edge_dencity,edge_v,G,edge_car_con,trafic_light_matrix1,distances,more_way,control)
        s['noe'] = noes
        s['light'] = LS
        s['direct'] = ds
        s['vel']=vels
        s['nostop']=nonS
        s['limit']=nevod
        s['noturn']=nturn
        ##geting the local and global rawards in RL2
        arivel,dennn,W, r, V, I,evod_reward,non_reward,vel_reward,noe_reward,light_reward,direction_reward,noturn_reward,ww=W_r(arivel_time, edge_car_con, error_turn, edge_v,MAT,G,trafic_light_matrix1,edge_dencity)
        rew.append(r)
        errorcount.append(np.sum(error_turn))
        ## disterbuting global reward in RL2
        rd,rl,rnoe,rvel,rnon,revod,rnotr=r_disterb(max_X, r)
        ##Q table
        for i in range(25):  # Adjust for zero-based indexing in Python
            for j in range(2):
                # Update the Q.direction values
                Q['direct'][i, j] += alpha2 * ((rd[i]+(direction_reward*max_X['direct'][i]/(np.sum(max_X['direct'])+.00001))) + gamma2 * np.max(Q['direct'][i, :]) - Q['direct'][i, j])
                if np.isnan(Q['direct'][i, j] )==1:
                    Q['direct'][i, j] =100*abs(rann.uniform(-1, 1))                
        for i in range(25):  # Adjust for zero-based indexing in Python
            for j in range(2):
                # Update the Q.direction values
                Q['noturn'][i, j] += alpha2 * ((rnotr[i]+(noturn_reward*max_X['noturn'][i]/(np.sum(max_X['noturn'])+.00001))) + gamma2 * np.max(Q['noturn'][i, :]) - Q['noturn'][i, j])        
                if np.isnan(Q['noturn'][i, j] )==1:
                    Q['noturn'][i, j] =100*abs(rann.uniform(-1, 1))     
        for i in range(25):  # Adjust for zero-based indexing in Python
            for j in range(4):
                # Update the Q.direction values
                Q['light'][i, j] += alpha2 * ((rl[i]+(light_reward*max_X['light'][i]/(np.sum(max_X['light'])+.00001))) + gamma2 * np.max(Q['light'][i, :]) - Q['light'][i, j])
                if np.isnan(Q['light'][i, j] )==1:
                    Q['light'][i, j] =100*abs(rann.uniform(-1, 1)) 
        for i in range(75):  # Adjust for zero-based indexing in Python
            for j in range(2):
                # Update the Q.direction values
                Q['noe'][i, j] += alpha2 * ((rnoe[i]+(noe_reward*max_X['noe'][i]/(np.sum(max_X['noe'])+.00001))) + gamma2 * np.max(Q['noe'][i, :]) - Q['noe'][i, j])
                if np.isnan(Q['noe'][i, j])==1:
                    Q['noe'][i, j]=100*abs(rann.uniform(-1, 1))
        for i in range(25):  # Adjust for zero-based indexing in Python
            for j in range(5):
                # Update the Q.direction values
                Q['vel'][i, j] += alpha2 * ((rvel[i]+(vel_reward*max_X['vel'][i]/(np.sum(max_X['vel'])+.00001))) + gamma2 * np.max(Q['vel'][i, :]) - Q['vel'][i, j])
                if np.isnan(Q['vel'][i, j] )==1:
                    Q['vel'][i, j] =100*abs(rann.uniform(-1, 1)) 
        for i in range(75):  # Adjust for zero-based indexing in Python
            for j in range(2):
                # Update the Q.direction values
                Q['nostop'][i, j] += alpha2 * ((rnon[i]+(non_reward*max_X['nostop'][i]/(np.sum(max_X['nostop'])+.00001))) + gamma2 * np.max(Q['nostop'][i, :]) - Q['nostop'][i, j])
                if np.isnan(Q['nostop'][i, j] )==1:
                    Q['nostop'][i, j] =100*abs(rann.uniform(-1, 1)) 
        for i in range(10):  # Adjust for zero-based indexing in Python
            for j in range(3):
                # Update the Q.direction values
                Q['limit'][i, j] += alpha2 * ((revod[i]+(evod_reward*max_X['limit'][i]/(np.sum(max_X['limit'])+.00001))) + gamma2 * np.max(Q['limit'][i, :]) - Q['limit'][i, j])
                if np.isnan(Q['limit'][i, j] )==1:
                    Q['limit'][i, j] =100*abs(rann.uniform(-1, 1)) 
    ##ststes update
        s['noe'] = []
        s['light'] =[]
        s['direct'] = []
        s['vel']=[]
        s['nostop']=[]
        s['limit']=[]
        s['noturn']=[]


   ###############end of RL2######################################
    rewards1=reward_handeling_mat(PP,edgematrix,action,MAT,evod_reward,non_reward,vel_reward,noe_reward,light_reward,direction_reward,r)
    for i in range(len(PP)):
        for j in range(5):
         # Update Q-value using the Q-learning formula
         q_table[PP[i]][j] =  alpha * ((rewards1[i][j]) + gamma * np.max(q_table[PP[i]]) - q_table[PP[i]][j])
         if np.isnan(q_table[PP[i]][j] )==1:
             q_table[PP[i]][j]=0 
    ###############end of RL1######################################


plt.figure(8)
plt.plot(rew)
plt.title("global reward")
plt.xlabel('episode')
plt.ylabel('amount')
plt.show()

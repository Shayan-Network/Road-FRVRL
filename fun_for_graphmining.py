## actions for graph mining RL
def action_handeling_mat(PP,edgematrix,action,MAT):
    for state in range(len(PP)):
        if action[state]==0:
            MAT['nostop'].append([int(edgematrix[PP[state]][0]),int(edgematrix[PP[state]][1])])
        elif action[state]==1:
            MAT['diection_f3'].append([int(edgematrix[PP[state]][0])])
            MAT['direction_f4'].append([int(edgematrix[PP[state]][1])])
        elif action[state]==2:
            MAT['noef1'].append([int(edgematrix[PP[state]][0])])
            MAT['noef2'].append([int(edgematrix[PP[state]][1])])
        elif action[state]==3:
            MAT['limit'].append([int(edgematrix[PP[state]][0]),int(edgematrix[PP[state]][1])])
    return MAT

## finding the edge indext
import networkx as nx
def find_edge_index(G, u, v):
    try:
        return G.edges[u, v]['index']
    except KeyError:
        return None
import numpy as np
import random

# Initialize Q-learning parameters
def initialize_q_table(pp):
    # Initialize Q-table with states based on PP and 5 possible actions (0 to 4)
    q_table = {}
    for entry in pp:
        q_table[entry] = [0] * 5  # 5 possible actions
    return q_table

# Choose action using epsilon-greedy strategy
def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        # Exploration: choose random action
        return random.randint(0, 5)
    else:
        # Exploitation: choose the best action based on Q-values
        return np.argmax(q_table[state])

#################################################################################
import networkx as nx

def noturn_mat(inte,G):
    A_ll=[]
    A=[]
    B=[]
    A=list(G.predecessors(inte))
    B=list(G.successors(inte))
    A1=[[int(A[0]),inte],[inte,int(B[1])],[inte,int(B[3])]]
    A2=[[int(A[1]),inte],[inte,int(B[2])],[inte,int(B[0])]]
    A3=[[int(A[2]),inte],[inte,int(B[3])],[inte,int(B[1])]]
    A4=[[int(A[3]),inte],[inte,int(B[2])],[inte,int(B[0])]]
        
    return A1,A2,A3,A4    


##candidate ditermination
import networkx as nx
import numpy as np
def MAT_determin(car_ratio,dt,simulation_time,con_prob,inter_p,arrival_times,G, edge_car_con, edge_v,MAT,control,edge_dencity):


    num_steps = simulation_time / dt

    #print("arrival_times=",arrival_times)

    ###############geting the probebity matrix####################################################
    edgematrix = np.array(G.edges)
    conjunction=edge_car_con/(edge_v+1)
    for i in range(len(conjunction)):
        if np.isnan(conjunction[i])==1:
            conjunction[i]=0
        


    ##########determining the intersections###############################
    inter_p=5
    intersections=[]
    for node in G.nodes():
        in_degree = G.in_degree(node)   # Number of incoming edges
        out_degree = G.out_degree(node)  # Number of outgoing edges
        if in_degree == 4 and out_degree ==4:
            intersections.append(node)
    intersections = [int(x) for x in intersections]
    inter_node_input=[[] for _ in range(len(intersections))]
    inter_node_output=[[] for _ in range(len(intersections))]
    
    for i in range(len(intersections)):
        inputs=[]    
        inputs = list(G.predecessors(intersections[i]))  # Nodes with edges going into the intersection
        inputs = [int(x) for x in inputs]
        for j in range(len(inputs)):
            inter_node_input[i].append([inputs[j],intersections[i]])
            
        output=[]    
        output = list(G.successors(intersections[i]))  # Nodes with edges going into the intersection
        output = [int(x) for x in output]
        for j in range(len(output)):
            inter_node_output[i].append([intersections[i],output[j]]) 
            
            
    inter_edge_input=[[] for _ in range(len(intersections))]
    inter_edge_output=[[] for _ in range(len(intersections))] 
    ## converting  to edge number       
    for i in range(len(intersections)):
        for j in range(4):
            inter_edge_input[i].append(find_edge_index(G, inter_node_input[i][j][0], inter_node_input[i][j][1]))
            inter_edge_output[i].append(find_edge_index(G, inter_node_output[i][j][0], inter_node_output[i][j][1]))

    

    inter_edge_con_in=[[] for _ in range(len(intersections))]
    inter_edge_con_out=[[] for _ in range(len(intersections))]

    for i in range(len(intersections)):
        inter_edge_con_in[i]=conjunction[inter_edge_input[i]]
        inter_edge_con_out[i]=conjunction[inter_edge_output[i]]
    
    intersections2=[]
    indexintersection=[]
    for i in range(len(intersections)):
        AA=0
        BB=0
        AA=np.sum(inter_edge_con_in[i])
        BB=np.sum(inter_edge_con_out[i])
        CC=0
        CC=AA+BB
        if CC>(max(conjunction)-min(conjunction))/4:
            intersections2.append(intersections[i])
            indexintersection.append(i)
    
    MAT['noturn'] =[]
    
    for i in range(len(intersections2)):
        A1,A2,A3,A4=[],[],[],[]  
        A1,A2,A3,A4=noturn_mat(intersections2[i],G)
        MAT['noturn'].append(A1)
        MAT['noturn'].append(A2)
        MAT['noturn'].append(A3)
        MAT['noturn'].append(A4)

        
        
    
                

    inputs2=[]
    outputs2=[]
    lightmat=[]
    MAT['redside']=[[] for _ in range(len(intersections2))]
    MAT['greenside']=[[] for _ in range(len(intersections2))]
    for i in range(len(intersections2)):
        inputs = list(G.predecessors(intersections2[i]))  # Nodes with edges going into the intersection
        inputs = [int(x) for x in inputs]
        inputs2.append(inputs)
        for j in range(len(inputs)):
            if j%2==1:
                MAT['redside'][i].append([inputs[j],intersections2[i]])
                lightmat.append(find_edge_index(G, inputs[j], intersections2[i]))
            else:
                MAT['greenside'][i].append([inputs[j],intersections2[i]])
                lightmat.append(find_edge_index(G, inputs[j], intersections2[i]))
        outputs = list(G.successors(intersections2[i]))   # Nodes with edges going out of the intersection
        outputs = [int(x) for x in outputs]
        outputs2.append(inputs)
        for j in range(len(outputs)):
            if j%2==0:
                MAT['redside'][i].append([intersections2[i],outputs[j]])
                lightmat.append(find_edge_index(G, intersections2[i], outputs[j]))
            else:
                MAT['greenside'][i].append([intersections2[i],outputs[j]])
                lightmat.append(find_edge_index(G, intersections2[i], outputs[j]))
                
    PP=[]
    thresh=con_prob
    for i in control:
        if edge_dencity[i]>=thresh:
            if  (i in lightmat)==0:
                PP.append(int(i))
         
    return PP,MAT
########################################
import numpy as np
def reward_handeling_mat(PP,edgematrix,action,MAT,evod_reward,non_reward,vel_reward,noe_reward,light_reward,direction_reward,globalreward):
    rewards1=[np.zeros(5) for _ in range(len(PP))]
    for state in range(len(PP)):
        if action[state]==0:
            if np.isnan(non_reward)==1:
                non_reward=0
            rewards1[state][0]=(.5*non_reward+1*globalreward)
        elif action[state]==1:
            if np.isnan(direction_reward)==1:
                direction_reward=0
            rewards1[state][1]=(.5*direction_reward+1*globalreward)
        elif action[state]==2:
            if np.isnan(noe_reward)==1:
                noe_reward=0            
            rewards1[state][2]=(.5*noe_reward+1*globalreward)
        elif action[state]==3:
            if np.isnan(evod_reward)==1:
                evod_reward=0                   
            rewards1[state][3]=(.5*evod_reward+1*globalreward)
        elif action[state]==4:
            rewards1[state][4]=(globalreward)

    return rewards1
def direction_change(edgematrix, f3, f4):
    import networkx as nx
    import numpy as np
    import random as rann
    if not f3 and not f4:
        return edgematrix
    else:
        for i in range(len(f3)):
          if f3[i]!=[]:
            for j in range(len(edgematrix)):
              A=edgematrix[j]
              if A[0]==f3[i] and A[1]==f4[i]:
                edgematrix[j]=[f4[i],f3[i]]


        return edgematrix
    
###############################################################################    
def NtN_edgenumber1(number,edgematrix2):
  import networkx as nx
  import numpy as np
  import random as rann
        # Initialize an array of zeros
  NtN =[]

        # Iterate through each row in 'number'
            # Find the edge index using 'findedge' function
  for j1 in range(len(edgematrix2)):
    if(number[0] == edgematrix2[j1][0] and number[1] == edgematrix2[j1][1]):
      NtN.append(j1)

  return NtN
###################################################################################
def wrong_way(di,ent, evo, car_edge_path,notur_matrix):
  import networkx as nx
  import numpy as np
  import random as rann
  error_di=0
  error_ent=0
  error_evo=0
  error_noturn=0
  error=0
  


  a1=0
  a=0
  a11=0
  a2=0
  for i in range(len(di)):
    for j in range(len(car_edge_path)):
      if di[i] == car_edge_path[j]:
        a=a+1
  if a>=1:
    error_di=1
  else:
    error_di=0
  # for no entery
  for i in range(len(ent)):
    for j in range(len(car_edge_path)):
      if ent[i] == car_edge_path[j]:
        a11=a11+1
  if a11>=1:
    error_ent=1
  else:
    error_ent=0
# for no turn  
  for j in range(len(notur_matrix)):
    if len(notur_matrix[j])==2:
      n1=0
      n1=np.argwhere(notur_matrix[j][0]==car_edge_path)
      if len(n1)>0 and len(n1[0])>0:
        if car_edge_path[int(n1[0]+1)]==notur_matrix[j][1]:
          a1=1
        else:
          a1=0
    elif len(notur_matrix[j])==3:
      n1=0
      n1=np.argwhere(notur_matrix[j][0]==car_edge_path)
      if len(n1)>0 and len(n1[0])>0:
        if car_edge_path[int(n1[0]+1)]==notur_matrix[j][1] or car_edge_path[int(n1[0]+1)]==notur_matrix[j][2]:
          a1=1
        else:
          a1=0
  error_noturn=a1
  if error_evo==1 or error_di==1 or error_ent==1 or error_noturn==1:
      error=1
  else:
      error=0


  return error
######################################for only time limit#########################
def wrong_way2(evo, car_edge_path):
  import networkx as nx
  import numpy as np
  import random as rann

  error_evo=0
  error=0
  randommm = abs(np.random.rand())



  a11=0
  # for no entery
  for i in range(len(evo)):
    for j in range(len(car_edge_path)):
    
      if (evo[i][0]== car_edge_path[j]):
        a11=a11+1
  if a11>=1:
    error_evo=1
  else:
    error_evo=0
  if error_evo==1 and randommm>.2:
      error=1
  else:
      error=0
  return error
################################################################
import networkx as nx
import numpy as np
import random as rann
from functions_for_env import wrong_way
from functions_for_env import wrong_way2
from get_path import get_path
from fun_for_graphmining import find_edge_index

def pathfinder4(G, safepath, edgematrix, evo,ent, car_ege_path,di,qq,notur_matrix):

  C2=np.zeros(len(car_ege_path))
  C3=np.zeros(len(car_ege_path))
  for i in range(len(car_ege_path)):
    C2[i] = wrong_way(di,ent, evo, car_ege_path[i],notur_matrix)
    C3[i]=wrong_way2(evo, car_ege_path[i])
  C2 = [int(x) for x in C2]
  C3 = [int(x) for x in C3]

  d1=0
  d2=0
  error_index = np.zeros(len(car_ege_path))
  inedex = 0
  temp_list=[]
  temp_list2=[]
  for i in range(len(car_ege_path)):
    if i%5==0:
      print("path finder",i)
    a=0
    temp_list=[]
    temp_list2=[]
    if C2[i]==1 or C3[i]==1:
        lk = car_ege_path[i]
        if len(lk)<=2:
          path_list1=safepath
        else:          
          A = edgematrix[int(lk[0]), 0]
          B = edgematrix[int(lk[-1]), 1]
          paths =get_path(edgematrix,A,B,len(edgematrix))
          p=list(paths)
          d1=C2[i]
          d2=C3[i]

          while  d1==1 or d2==1 or a<3000:
              a=a+1
              paths =get_path(edgematrix,A,B,len(edgematrix))
              p=list(paths)
              d1 = wrong_way(di,ent, evo, p,notur_matrix)
              d2 =wrong_way2(evo, p)
              if a>11: 
                break
          if d1==1 or d2==1 :
                error_index[i]=1

                car_ege_path[i]=[1]
    
          elif d1==0 and d2==0:
                C2[i]=d1
          if len(p)==0:
            car_ege_path[i]=[1]
          else:
            for j in range(1,len(p)):
              temp_list.append([p[j-1],p[j]])

            for j in range(len(temp_list)):
              A=0
              A=find_edge_index(G, temp_list[j][0], temp_list[j][1])

              temp_list2.append(A)
                
        car_ege_path[i]=[]
        car_ege_path[i]=temp_list2


  path = car_ege_path
  return path,sum(error_index)
#######################################################################################
import numpy as np

def initial(G, edgematrix, num_cars, initial_pos, car_edge_speed, arivTime,distances):
    # Initialize the position array
    pos = np.zeros((6, num_cars))

    # Get the number of edges (rows in edgematrix)
    a = edgematrix.shape[0]

    # Initial positions
    for i in range(num_cars):
        for j in range(a):
            if np.array_equal(initial_pos[i, :2], edgematrix[j, :2]):
                initial_pos[i, 2] = distances[j]
                initial_pos[i, 3] = j

    # Car order in what edge, what car
    order1 = [[] for _ in range(a)]  # List of lists to store car orders per edge
    car_edge_order = [[] for _ in range(a)]  # List of lists to store the sorted car order
    car_edge_order_distance = [[] for _ in range(a)]  # List of lists to store sorted car distances

    for i in range(num_cars):
        for j in range(a):
            if initial_pos[i, 3] == j:
                order1[j].append(i)

    for i in range(a):
        A = initial_pos[order1[i], 2]
        B = np.sort(A)
        I = np.argsort(A)
        car_edge_order[i] = I
        car_edge_order_distance[i] = B

    for i in range(num_cars):
        for j in range(a):
            if i in order1[j]:
                A = order1[j]
                b1 = A.index(i)  # Find the index of car i in the edge list
                B = car_edge_order_distance[j]
                pos[0, i] = B[b1]  # Position on the edge
                pos[1, i] = j  # What edge
                pos[2, i] = distances[j]  # Total distance of the edge

                # Determine if it's the first car on the edge
                pos[3, i] = b1
                if b1 == 0:
                    pos[3, i] = b1 + 1  # First car
                else:
                    pos[3, i] = b1

                pos[4, i] = (i + 1) / 2 * car_edge_speed[int(pos[1, i])]

    # Set arrival time
    pos[5, :] = arivTime

    return pos
##############################################################################

def senarios(carstop_ratio, path, num_cars, nostop_path):
    import networkx as nx
    import numpy as np
    import random as rann
    num_stos = int(np.ceil(carstop_ratio * num_cars))
    #print('nostop_path',nostop_path)
    car_stop = np.random.randint(1, num_cars + 1, num_stos)
    car_stop_matrix = np.zeros((2, num_cars))
    car_stop_matrix[0, car_stop - 1] = 1  # Adjust for 0-based indexing in Python

    for i in range(num_cars):
        if i + 1 in car_stop:  # Adjust for 0-based indexing
            A = path[i][0]
            #print('A',A)
            for j in range(len(nostop_path)):
                if nostop_path[j] != A:
                    car_stop_matrix[1, i] = i + 1  # Adjust for 0-based indexing
                else:
                    car_stop_matrix[0, i] = 0

    return car_stop_matrix
#############################################################################################
import numpy as np

def dencity_cal(pos, edge_number, num_cars, distances, simulation_time, vmax, dt,TEMe):
    num_steps = int(simulation_time / dt)  # Number of simulation steps
    edge_dencity = np.zeros(edge_number)
    edge_car_con = np.zeros(edge_number)
    edge_car_v = np.zeros(edge_number)
    conter1_max=10
    conter1=0

    for t in range(num_steps):
        conter1=conter1+1
        for j in range(edge_number):
            for i in range(num_cars-1, num_cars):
                
                # Find cars on the current edge
                b3 = np.where(pos[t, 1, :] == j)[0]  # Find indices where cars are on edge 'j'
                FFF = pos[t, 0, b3]
                vv = pos[t, 4, b3]  # car velocities
                
                # Sort the cars based on their positions on the edge
                I1 = np.argsort(-FFF)  # Sort in descending order
                ko = b3[I1]  # Cars sorted by their positions on the edge
                
                # Count the number of cars on the current edge
                if t%conter1_max==0:
                  edge_car_con[j] += len(ko)
            
            # Calculate the sum of car velocities or add vmax if no cars
            la=len(vv)
            if la==0:
              la=1
            edge_car_v[j] += np.sum(vv)/la
            if np.isnan(np.sum(vv)) or np.sum(vv) < 0:
                edge_car_v[j] =50
    
    # Normalize car counts
    edge_car_con = edge_car_con / (num_steps*dt)
    
    # Calculate density
    edge_dencity = edge_car_con / distances[0:edge_number]
    
    # Calculate average velocity per edge
    for j in range(edge_number):
        if edge_car_con[j] == 0 or edge_car_v[j] == 0:
            edge_car_v[j] = 50
        # else:
        #     edge_car_v[j] = edge_car_v[j] / edge_car_con[j]
    
    # Normalize velocities
    edge_v = edge_car_v / (num_steps*dt)
    
    # Transpose results to match expected output format
    edge_dencity = edge_dencity.T
    edge_v = edge_v.T
    
    return edge_dencity, edge_car_con, edge_v
##################more way#####################################################
import networkx as nx
import numpy as np

def is_more_way(G):
    # Get the list of edges from the graph
    edgematrix = np.array(G.edges)
    a, b = edgematrix.shape
    more_way = np.zeros(a)
    for i in range(len(edgematrix)):
      out_degree1=0
      in_degree1=0
      out_degree2=0
      in_degree2=0      
      in_degree1 = G.in_degree(edgematrix[i][0])   # Number of incoming edges
      out_degree1 = G.out_degree(edgematrix[i][0])  # Number of outgoing edges
      in_degree2 = G.in_degree(edgematrix[i][1])   # Number of incoming edges
      out_degree2 = G.out_degree(edgematrix[i][1])  # Number of outgoing edges
      more_way[i]=in_degree1+in_degree2+out_degree1+out_degree2
    

    return more_way

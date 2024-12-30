import networkx as nx
import numpy as np
import random as rann
from get_path import get_path
from fun_for_graphmining import find_edge_index 
def path2(G,num_cars,safepath,inputnodes,outputnodes,edge_matrix,numbredges):

  A = 0
  B =0



  path_list1=[]
  paths=[]
# node to node mode for paths
  for i in range(num_cars):

    A=rann.choices(inputnodes, k=1)[0]
    B=rann.choices(outputnodes, k=1)[0]
    paths =get_path(edge_matrix,A,B,numbredges)

    while len(list(paths))==0 or A==B or len(list(paths))<6 :
      A=rann.choices(inputnodes, k=1)[0]
      B=rann.choices(outputnodes, k=1)[0]
      paths =get_path(edge_matrix,A,B,numbredges)
    
    p=list(paths)
    if len(p)==0:
      path_list1.append(safepath)
    else:

      path_list1.append((p))
      
  out_path=[]
  for i in range(num_cars):
    temp_list=[]
    temp_list2=[]

    index1=path_list1[i]
    len_index1=len(index1)

    for j in range(1,len_index1):
      temp_list.append([index1[j-1],index1[j]])

    for j in range(len(temp_list)):
      A=0
      A=find_edge_index(G, temp_list[j][0], temp_list[j][1])
      temp_list2.append(A)

    out_path.append(temp_list2)




  return out_path


def simulate_traffic(lam, time_interval, total_time):
    import networkx as nx
    import numpy as np
    import random as rann

    # Generate the number of cars that will arrive in the total time frame
    num_cars = np.random.poisson(lam)

    # Generate the arrival times for the cars based on Exponential distribution
    arrival_times = np.cumsum(np.random.exponential(time_interval, num_cars))

    # Filter out the cars that arrive after the simulation time frame
    arrival_times = arrival_times[arrival_times <= total_time]

    return arrival_times
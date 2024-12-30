import networkx as nx
import numpy as np
import random as rann

from functions_for_env import NtN_edgenumber1
from functions_for_env import pathfinder4
from functions_for_env import initial
from functions_for_env import senarios
from fun_for_idm import IDM7
from functions_for_env import dencity_cal
from functions_for_env import  is_more_way
from numba import njit
#@njit
def env(distances,inputnodes,outputnodes,safepath,action1, action2, action3, action4, action5, action6, action7, G, num_cars, simulation_time, car_ege_path, arivTime, dt,MAT,edgematrix,qq,trafic_light_matrix1):


 




  even_odd_matrix = []

 ###################direction change part 
  if len(MAT['diection_f3'])>0:
    f3=np.zeros(len(action2))
    f4=np.zeros(len(action2))
    di=[]
    for i in range(len(action2)):
      if action2[i] >= 0.5:
        #f3.append([])  # nodes first ends
        f3[i]=0
        f4[i]=0
        f3 = [int(x) for x in f3]
        f4 = [int(x) for x in f4]
      else:
        f3[i]=MAT['diection_f3'][i][0]
        f4[i]=MAT['direction_f4'][i][0]
        f3 = [int(x) for x in f3]
        f4 = [int(x) for x in f4]        
    for i in range(len(action2)):
      ind=[f3[i],f4[i]]
      A=NtN_edgenumber1(ind,edgematrix)
      di.append(A)
    di=[x for sublist in di for x in sublist]
    # edgematrix2= direction_change(edgematrix, f3, f4)
    # edgematrix=edgematrix2
    edgematrix2=edgematrix
  else:
    di=[]
    edgematrix2=edgematrix

    # No entry#######################################################

  if len(MAT['noef1'])>0:
    f1=np.zeros(len(action4))
    f2=np.zeros(len(action4))
    ent=[]
    for i in range(len(action4)):
      if action4[i] >= 0.5:

        f1[i]=0  # nodes
        f2[i]=0
        f1 = [int(x) for x in f1]
        f2 = [int(x) for x in f2]

      else:
        f1[i]=MAT['noef1'][i][0]  # nodes
        f2[i]=MAT['noef2'][i][0]
        f1 = [int(x) for x in f1]
        f2 = [int(x) for x in f2]
    for i in range(len(action4)):
      ine=[f1[i],f2[i]]
      A=NtN_edgenumber1(ine,edgematrix)
      ent.append(A)

    nedge = G.number_of_edges()
    ent=[x for sublist in ent for x in sublist]
  else:
    ent=[]


    #speed##############################################################
  car_edge_speed = action3 
  #  car_edge_speed = action3 * np.ones(nedge)


  #   # evon odd###########################################################
  if len(MAT['limit'])>0:
    evo=[]
    even_odd_matrix=[]
    for i in range(len(action7)):
      if action7[i] < .5: ## yes
        even_odd= NtN_edgenumber1(MAT['limit'][i], edgematrix2)
        evo.append(even_odd)  # even 0 or odd 1
  else:
    evo=[]
#############no turn###############################################
  if len(MAT['noturn'])>0:
    notur_matrix=[]    
    for i in range(len(action6)):
      if action6[i] < .6 and action6[i] >.4:
        A=0
        B=0
        A=NtN_edgenumber1(MAT['noturn'][i][0], edgematrix2)
        B=NtN_edgenumber1(MAT['noturn'][i][1], edgematrix2)
        notur_matrix.append([A,B])
      elif action6[i] < .8 and action6[i] >.6:
        A=0
        B=0
        A=NtN_edgenumber1(MAT['noturn'][i][0], edgematrix2)
        B=NtN_edgenumber1(MAT['noturn'][i][2], edgematrix2)
        notur_matrix.append([A,B])
      elif  action6[i] >.8:
        A=0
        B=0
        C=0
        A=NtN_edgenumber1(MAT['noturn'][i][0], edgematrix2)
        B=NtN_edgenumber1(MAT['noturn'][i][1], edgematrix2)
        C=NtN_edgenumber1(MAT['noturn'][i][2], edgematrix2)
        
        notur_matrix.append([A,B,C])
  else:
    notur_matrix=[]     
  #   ##path finder###########################
  if len(evo)>0 or len(ent)>0 or len(di)>0 or len(notur_matrix)>0:
    num_node=len(G.nodes)
    print("path cheking")
    (path,error_turn)=pathfinder4(G, safepath, edgematrix, evo,ent, car_ege_path,di,qq,notur_matrix)
    #path=car_ege_path
  else:
    path=car_ege_path
    error_turn=0
    
  initial_pos=np.zeros((len(car_ege_path),4))
  for i in range(num_cars):
    if len(path[i])==0:
      path[i]=[1]
      
    D = path[i][0]
    initial_pos[i,0] =edgematrix[[D][0]][0]
    initial_pos[i,1] =edgematrix[[D][0]][1]

  pos=initial(G, edgematrix, num_cars, initial_pos, car_edge_speed, arivTime,distances)
  pos[0,:]=0
  #   ##for yes_no stop 1###################################################
  if len(MAT['nostop'])>0:
    nostop_path =np.zeros(len(action5))
    for i in range(len(MAT['nostop'])):
      if action5[i] >= 0.5:
        nostop_path[i]=0
      else:
      # Assuming you have a graph G defined
        for j in range(len(edgematrix)):
          if MAT['nostop'][i][0]==edgematrix[j][0] and MAT['nostop'][i][1]==edgematrix[j][1]:
            f=j
            nostop_path[i]=f
  else:
    nostop_path=[]

  if len(nostop_path)>0:
    carstop_ratio = 0.8
    car_stop_matrix = senarios(carstop_ratio, path, num_cars, nostop_path)
  else:
    car_stop_matrix=[]
  
  numedge=len(edgematrix)

  #   #for trafic ligte 2########################################
  if len(trafic_light_matrix1[0])>0 :
    numTl1 = action1 + 5
  else:
    trafic_light_matrix1=[[],[]]
    numTl1=0
  #print("distances pos",distances)
  print("IDM") 
  pos2=[]
  pos2,TEMe=IDM7(pos, num_cars, path, car_edge_speed, distances, numedge, car_stop_matrix, trafic_light_matrix1, numTl1, simulation_time, dt)
  
  edge_dencity, edge_car_con, edge_v=dencity_cal(pos2, numedge, num_cars, distances, simulation_time, car_edge_speed, dt,TEMe)
  more_way=is_more_way(G)
  more_way = [int(x) for x in more_way]
  S=0
  for i in range (len(TEMe)):
    if TEMe[i]==0:
      S=S+1000
    else:
      A=0
      A=TEMe[i]*dt
      S=S+A
  arivel_time=S
  return pos2,edge_dencity, edge_car_con, edge_v,error_turn,trafic_light_matrix1,more_way,arivel_time
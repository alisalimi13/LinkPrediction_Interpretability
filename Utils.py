import numpy as np
import random
import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import linalg as LA
import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from collections import Counter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import timeit
import shutil
import subprocess
import matplotlib.pyplot as plt
import math
import pathlib
import os
from distutils.dir_util import copy_tree


def read_dataset(dataset):
  train = open('data/' + dataset+ '/train2id.txt').readlines()[1:]
  train = [i.split() for i  in train]
  test = open('data/' + dataset+ '/test2id.txt').readlines()[1:]
  test = [i.split() for i  in test]
  valid = open('data/' + dataset+ '/valid2id.txt').readlines()[1:]
  valid = [i.split() for i  in valid]
  train = [[int(i),int(k),int(j)] for i,j,k in train]
  test  = [[int(i),int(k),int(j)] for i,j,k in test ]
  valid = [[int(i),int(k),int(j)] for i,j,k in valid]
  WholeGraph = train+valid+test
  return train,valid,test,WholeGraph
  
def dataset_creator(train,test,valid,name = 'dataset'): 
  pathlib.Path('/content/dataset').mkdir(parents=True, exist_ok=True) 
  # create dataset in a format that suitable for openke
  shutil.copy('/content/OpenKE/benchmarks/FB13/n-n.py', '/content/'+name)

  graph = train + test + valid
  entities = [i[0] for i in graph] + [i[2] for i in graph] 
  relations= [i[1] for i in graph]
  entities = list(set(entities))
  relations = list(set(relations))
  ent_dict = {}
  for i in range(len(entities)):
    ent_dict[entities[i]] = i
  rel_dict = {}
  for i in range(len(relations)):
    rel_dict[relations[i]] = i    
    
  f1 = open('/content/'+name+'/entity2id.txt','w')
  f1.write(str(len(ent_dict))+'\n')
  for k,v in ent_dict.items():
    f1.write(str(k)+'\t'+str(v)+ '\n')
  f1.close()

  f1 = open('/content/'+name+'/relation2id.txt','w')
  f1.write(str(len(rel_dict))+'\n')
  for k,v in rel_dict.items():
    f1.write(str(k)+'\t'+str(v)+ '\n')
  f1.close()


  f1 = open('/content/'+name+'/train2id.txt','w')
  f1.write(str(len(train))+'\n')
  for row in train:
    f1.write(str(ent_dict[row[0]])+' '+str(ent_dict[row[2]])+' '+str(rel_dict[row[1]])+'\n')

  f1.close()

  f1 = open('/content/'+name+'/test2id.txt','w')
  f1.write(str(len(test))+'\n')
  for row in test:
    f1.write(str(ent_dict[row[0]])+' '+str(ent_dict[row[2]])+' '+str(rel_dict[row[1]])+'\n')
  f1.close() 

  f1 = open('/content/'+name+'/valid2id.txt','w')
  f1.write(str(len(valid))+'\n')
  for row in valid:
    f1.write(str(ent_dict[row[0]])+' '+str(ent_dict[row[2]])+' '+str(rel_dict[row[1]])+'\n')
  f1.close() 

  subprocess.call("/content/"+name+"/n-n.py.py", shell=True)

def n2n_py():
  lef = {}
  rig = {}
  rellef = {}
  relrig = {}

  triple = open("train2id.txt", "r")
  valid = open("valid2id.txt", "r")
  test = open("test2id.txt", "r")

  tot = (int)(triple.readline())
  for i in range(tot):
    content = triple.readline()
    h,t,r = content.strip().split()
    if not (h,r) in lef:
      lef[(h,r)] = []
    if not (r,t) in rig:
      rig[(r,t)] = []
    lef[(h,r)].append(t)
    rig[(r,t)].append(h)
    if not r in rellef:
      rellef[r] = {}
    if not r in relrig:
      relrig[r] = {}
    rellef[r][h] = 1
    relrig[r][t] = 1

  tot = (int)(valid.readline())
  for i in range(tot):
    content = valid.readline()
    h,t,r = content.strip().split()
    if not (h,r) in lef:
      lef[(h,r)] = []
    if not (r,t) in rig:
      rig[(r,t)] = []
    lef[(h,r)].append(t)
    rig[(r,t)].append(h)
    if not r in rellef:
      rellef[r] = {}
    if not r in relrig:
      relrig[r] = {}
    rellef[r][h] = 1
    relrig[r][t] = 1

  tot = (int)(test.readline())
  for i in range(tot):
    content = test.readline()
    h,t,r = content.strip().split()
    if not (h,r) in lef:
      lef[(h,r)] = []
    if not (r,t) in rig:
      rig[(r,t)] = []
    lef[(h,r)].append(t)
    rig[(r,t)].append(h)
    if not r in rellef:
      rellef[r] = {}
    if not r in relrig:
      relrig[r] = {}
    rellef[r][h] = 1
    relrig[r][t] = 1

  test.close()
  valid.close()
  triple.close()

  f = open("type_constrain.txt", "w")
  f.write("%d\n"%(len(rellef)))
  for i in rellef:
    f.write("%s\t%d"%(i,len(rellef[i])))
    for j in rellef[i]:
      f.write("\t%s"%(j))
    f.write("\n")
    f.write("%s\t%d"%(i,len(relrig[i])))
    for j in relrig[i]:
      f.write("\t%s"%(j))
    f.write("\n")
  f.close()

  rellef = {}
  totlef = {}
  relrig = {}
  totrig = {}
  # lef: (h, r)
  # rig: (r, t)
  for i in lef:
    if not i[1] in rellef:
      rellef[i[1]] = 0
      totlef[i[1]] = 0
    rellef[i[1]] += len(lef[i])
    totlef[i[1]] += 1.0

  for i in rig:
    if not i[0] in relrig:
      relrig[i[0]] = 0
      totrig[i[0]] = 0
    relrig[i[0]] += len(rig[i])
    totrig[i[0]] += 1.0

  s11=0
  s1n=0
  sn1=0
  snn=0
  f = open("test2id.txt", "r")
  tot = (int)(f.readline())
  for i in range(tot):
    content = f.readline()
    h,t,r = content.strip().split()
    rign = rellef[r] / totlef[r]
    lefn = relrig[r] / totrig[r]
    if (rign < 1.5 and lefn < 1.5):
      s11+=1
    if (rign >= 1.5 and lefn < 1.5):
      s1n+=1
    if (rign < 1.5 and lefn >= 1.5):
      sn1+=1
    if (rign >= 1.5 and lefn >= 1.5):
      snn+=1
  f.close()


  f = open("test2id.txt", "r")
  f11 = open("1-1.txt", "w")
  f1n = open("1-n.txt", "w")
  fn1 = open("n-1.txt", "w")
  fnn = open("n-n.txt", "w")
  fall = open("test2id_all.txt", "w")
  tot = (int)(f.readline())
  fall.write("%d\n"%(tot))
  f11.write("%d\n"%(s11))
  f1n.write("%d\n"%(s1n))
  fn1.write("%d\n"%(sn1))
  fnn.write("%d\n"%(snn))
  for i in range(tot):
    content = f.readline()
    h,t,r = content.strip().split()
    rign = rellef[r] / totlef[r]
    lefn = relrig[r] / totrig[r]
    if (rign < 1.5 and lefn < 1.5):
      f11.write(content)
      fall.write("0"+"\t"+content)
    if (rign >= 1.5 and lefn < 1.5):
      f1n.write(content)
      fall.write("1"+"\t"+content)
    if (rign < 1.5 and lefn >= 1.5):
      fn1.write(content)
      fall.write("2"+"\t"+content)
    if (rign >= 1.5 and lefn >= 1.5):
      fnn.write(content)
      fall.write("3"+"\t"+content)
  fall.close()
  f.close()
  f11.close()
  f1n.close()
  fn1.close()
  fnn.close()
  
def Cut_graph(dataset):
  population_size = 100
  generations     = 10
  temp            = 4000 # init temprature
  good_ppl_rate   = 0.6 #0.65
  train,valid,test,Graph = read_dataset(dataset)
  WholeGraph = Graph
  Entities = {}
  for i,j,k in WholeGraph:
    if i not in Entities:
      Entities[i] = 1
    if k not in Entities:
      Entities[k] = 1 
  def my_sigmoid(x):
    t = 1 / (1 + math.exp(-(x/3)))
    return  2*(1 -( t ))
  Entity_count = len(Entities.keys())
  matrix = {}
  for i in range(len(Graph)):
      head = Graph[i][0]
      tail = Graph[i][2]
      if head not in matrix:
          matrix[head] = []
          matrix[head].append(tail)
      elif tail not in matrix[head]:
          matrix[head].append(tail)
  for i in range(len(Graph)):
      tail = Graph[i][0]
      head = Graph[i][2]
      # tail, head =  head, tail
      if head not in matrix:
          matrix[head] = []
          matrix[head].append(tail)
      elif tail not in matrix[head]:
          matrix[head].append(tail)
  Components = []
  graph_bfs = []
  for k,v in matrix.items():
      graph_bfs.append([k]+v)
  Entities = {}
  for i,j,k in WholeGraph:
    if i not in Entities:
      Entities[i] = 1
    if k not in Entities:
      Entities[k] = 1 
  def get_vertices(WholeGraph):
    return list(range(Entity_count))
  def random_split(V,sample_size = 200, matrix = matrix): # mutation
    V1,V2 = V[:int(len(V)/2)],V[int(len(V)/2):]
    S1 = random.sample(range(1, len(V1)), sample_size)
    S2 = random.sample(range(1, len(V2)), sample_size)
    t1 = [V1[i] for i in S1] # changes (remove from set 1)
    t2 = [V2[i] for i in S2]
    t1 = {i:1 for i in t1}
    t2 = {i:1 for i in t2}
    V1_new = [i for i in V1 if i not in t1]
    V2_new = [i for i in V2 if i not in t2]
    V1_new += list(t2.keys())
    V2_new += list(t1.keys())
    V1_new = {i:1 for i in V1_new}
    V2_new = {i:1 for i in V2_new}
    v_cuts = 0
    for k,v in V1_new.items():
      if k  in matrix:
        for i in matrix[k]:
          if i in V2_new:
            v_cuts += 1
    return list(V1_new.keys()), list(V2_new.keys()),v_cuts
  city = []
  V = get_vertices(WholeGraph)
  for i in range(population_size):
    temp = int(temp*my_sigmoid(j))
    v1,v2,cut = random_split(V,temp)
    city.append([v1,v2,cut])
  for j in range(generations):
    city = sorted(city,key=lambda l:l[-1])
    city = city[:population_size]
    temp = int(temp*my_sigmoid(j))
    print([sorted([int(i[-1]) for i in city])[:10],sum([i[-1] for i in city])])
    death_cut = sorted([i[-1] for i in city])[int(len(city)*good_ppl_rate)-1]
    city = [i for i in city if i[-1] <= death_cut]
    new_ppl = []
    for i in range(len(city)):
      V1,V2 = city[i][0],city[i][1]
      v1,v2,cut = random_split(V1+V2,temp)
      new_ppl.append([v1,v2,cut])
    city += new_ppl
    immigs = []
    # for i in range(population_size-len(city)):
    for i in range(40):
      V11 = list(V)
      random.shuffle(V11)
      v1,v2,cut = random_split(V11,temp)
      immigs.append([v1,v2,cut])
    city += immigs
  return city[0]
# V1,V2,cut = Cut_graph(dataset)



def remove_coonections_of_graphs(v1,v2,train,test,valid,rate = 0.01):
  train_exp1 = [] # for two disjoint graph
  valid_exp1 = [] # test set
  test_exp1 = [] # for 2 disjoint graphs in test
  AliSalim_Test = [] # for connections in between
  V1 = {}
  for item in v1:
    V1[item] = 1
  V2 = {}
  for item in v2:
    V2[item] = 1  
  removed = []
  for row in train:
    h,r,t = row
    if h in V1 and t in V2:
      removed.append(row)
      continue
    if h in V2 and t in V1:
      removed.append(row)
      continue
    train_exp1.append(row)
  random.shuffle(removed)
  train_exp1 = train_exp1 + removed[:int(len(removed) * rate)]
  for row in test:
    h,r,t = row
    if h in V1 and t in V2:
      continue
    if h in V2 and t in V1:
      continue
    test_exp1.append(row)

#   for row in test:
#     h,r,t = row
#     if h in V1 and t in V2:
#       AliSalim_Test.append(row)
#     if h in V2 and t in V1:
#       AliSalim_Test.append(row)

  for row in test:
    valid_exp1.append(row)
    
    
  AliSalim_Test = [i for i in valid_exp1 if i not in test_exp1]
  return train_exp1,valid_exp1, test_exp1,AliSalim_Test 


def prepare_dataset4exprement(dataset):
  train,valid,test,WholeGraph = read_dataset(dataset)
  dataset_creator(train,test,valid)
  os.chdir('/content/dataset')
  n2n_py()
  os.chdir('/content')


def Prepare_data_into_files_for_exprement(train_exp1,valid_exp1, test_exp1,test3):
  #this is the last stage, takes files and create 2-3 datasets for running exprements
  pathlib.Path('dataset_test3').mkdir(parents=True, exist_ok=True) 
  # !mkdir dataset_test3
  dataset_creator(train_exp1,test3,test3,'dataset_test3')
  os.chdir('dataset_test3')
  # %cd dataset_test3
  n2n_py()
  os.chdir('..')

  #test2
  pathlib.Path('dataset_test2').mkdir(parents=True, exist_ok=True) 
  dataset_creator(train_exp1,test_exp1,valid_exp1,'dataset_test2')
  os.chdir('dataset_test2')
  n2n_py()
  os.chdir('..')


  #test1
  pathlib.Path('dataset_test1').mkdir(parents=True, exist_ok=True) 
  dataset_creator(train_exp1,valid_exp1,test_exp1,'dataset_test1')
  os.chdir('dataset_test1')
  n2n_py()
  os.chdir('..')
  
  
  shutil.copytree('/content/dataset_test1', '/content/OpenKE/benchmarks/dataset_test1') 
  shutil.copytree('/content/dataset_test2', '/content/OpenKE/benchmarks/dataset_test2') 
  shutil.copytree('/content/dataset_test3', '/content/OpenKE/benchmarks/dataset_test3') 
  
  try: 
    shutil.rmtree('/content/dataset_test1')
  except:
    pass
  try: 
    shutil.rmtree('/content/dataset_test2')
  except:
    pass

  try: 
    shutil.rmtree('/content/dataset_test3')
  except:
    pass


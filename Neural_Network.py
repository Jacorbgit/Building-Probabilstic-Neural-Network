import pandas as pd
import numpy as np
from collections import OrderedDict
#from my_function import create_random_matrix
from tool_box import *
import matplotlib.pyplot as plt
import math
import warnings
import seaborn as sns


def run_this():
  print("initial setup")
# titanic = sns.load_dataset('titanic')
# summary(titanic)
# head(titanic,25)
# print("numbr of rows in dataframe: ", nrow(titanic))
#----------------------------------------------------------------------- Sample_data creation
def sample_data_creation():
  global df
  global training_data
  #global training_data
  df = pd.DataFrame(np.random.rand(200, 5), columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5'])
  # Add a binary column
  df['Binary'] = np.random.choice([0, 1], size=(200,))
  training_data = df
  #return df, training_data
# -------------------------------Skip user input and let values be initiialized below...

def network_initialization():
  global layer_1_weights, layer_2_weights, layer_3_weights
  global layer_1_biases, layer_2_biases, layer_3_bias
  global num_nodes, feature_n
  feature_n = int(ncol(training_data) - 1)
  num_nodes = [3,5]
  num_nodes = [int(x) for x in num_nodes]
  matrix_dictionary = OrderedDict()
  layer_1_weights = create_random_matrix(feature_n,num_nodes[0])
  matrix_dictionary[1] = layer_1_weights
  layer_2_weights = create_random_matrix(num_nodes[0],num_nodes[1])
  matrix_dictionary[2] = layer_2_weights
  layer_3_weights = create_random_matrix(num_nodes[1], 1)
  matrix_dictionary[3] = layer_3_weights
  layer_1_biases = np.random.rand(num_nodes[0])
  layer_2_biases = np.random.rand(num_nodes[1])
  layer_3_bias = np.random.rand(1)
  
  print("Network Initialized...","\n\n\n\n")
  print("dim of layer 3 weight: ", layer_3_weights.shape)

# -------------------------------------------------------------------------------FORWARD PASS:

def forward_pass(dataframe_subset):
  print("  --------------------   FORWARD PASS   ---------------------", "\n\n\n\n")
  global a_L, layer_1_preact, layer_1_output, layer_2_preact, layer_2_output, final_activation
  global layer_1_activaions, layer_2_activations, layer_1_activation_cache, layer_2_activation_cache
  layer_1_activation_cache = np.empty((0, num_nodes[0]))
  layer_2_activation_cache = np.empty((0, num_nodes[1]))
  # layer_1_activations = []
  # layer_2_activations = []
  a_L = []
  for i in range(nrow(dataframe_subset)): # (nrow(dataframe_subset)):
    input_vector = dataframe_subset.iloc[i].to_numpy()
    #input_vector = np.array(input_vector)
    layer_1_preact = (np.dot(input_vector,layer_1_weights) + layer_1_biases)
    layer_1_output = relu(layer_1_preact)
    layer_2_preact = (np.dot(layer_1_output,layer_2_weights) + layer_2_biases)
    layer_2_output = relu(layer_2_preact)
    final_preactivation = (np.dot(layer_2_output, layer_3_weights) + layer_3_bias)
    #........................................there is a bug here because this should be a scalar..
    output = sigmoid(final_preactivation) # activation_L is the final output
    print("sigmodoid output is:  ", output)
    output = float(output)
    #output = round(output,8)
    a_L.append(output)
# -------------------------------------- NETWORK STATUS
 # global layer_1_activaions, layer_2_activations, layer_1_activation_cache, layer_2_activation_cache
 # layer_1_activation_cache = np.empty((0, num_nodes[0]))
 # layer_2_activation_cache = np.empty((0, num_nodes[1]))
 # layer_1_activations = []
 # layer_2_activations = []
    layer_1_activations = []
    layer_2_activations = []
    for j in range(num_nodes[0]):
      if layer_1_output[j] == 0:
        layer_1_activations.append(0)
      else:
       layer_1_activations.append(1)
    layer_1_activation_cache = np.vstack([layer_1_activation_cache, layer_1_activations])
    
    for j in range(num_nodes[1]):
      if layer_2_output[j] == 0:
        layer_2_activations.append(0)
      else:
       layer_2_activations.append(1)
    layer_2_activation_cache = np.vstack([layer_2_activation_cache, layer_2_activations])

  
  print("forward_pass successful, vector of expected outputs is: ", a_L, "\n", "\n", "\n")
  print("after foward pass dimensions of layer_2_matrix are:   ", dim(layer_2_weights))
  return a_L 

#-----------------------------NETWORK STATUS---
#------------------------------------------------------------------------

# def know_status():
#   global layer_1_activaions, layer_2_activations
#   layer_1_activations = []
#   layer_2_activations = []
  
#   for i in range(num_nodes[0]):
#     if layer_1_output[i] == 0:
#       layer_1_activations.append(0)
#     else:
#      layer_1_activations.append(1)
  
#   for i in range(num_nodes[1]):
#     if layer_2_output[i] == 0:
#       layer_2_activations.append(0)
#     else:
#      layer_2_activations.append(1)
#-----------------BACKPROPOGATION---------GRADIENT_DESCENT---------------
#-----------------------------------------------------------------------------

def loss(dataframe_subset):
  epsilon = 1e-15
  total = 0
  for i in range(nrow(dataframe_subset)): #(nrow(dataframe_subset)):
    total += ((label[i]*math.log(a_L[i] + epsilon) + (1-label[i])*math.log(1-a_L[i] - epsilon)))
  log_loss = -1*(1/nrow(dataframe_subset))*total
  log_loss =round(log_loss,4)
  print("loss found successfully, the loss for this subset is:  ", log_loss)
  return(log_loss)

def backpropogation(dataframe_subset):  # dataframe_subset is unsupervised
     # " | "
     # " | "
     # " V "
  global cost_wrt_layer_1_weights, cost_wrt_layer_1_activations, cost_wrt_layer_2_weights
  global cost_wrt_layer_2_activations, cost_wrt_layer_3_weights
  cost_wrt_layer_3_weights = []
  cost_wrt_layer_2_activations = []
  cost_wrt_layer_2_weights = np.empty((num_nodes[0], num_nodes[1]))
  cost_wrt_layer_1_activations = []
  cost_wrt_layer_1_weights = np.empty((feature_n, num_nodes[0]))
  cost = 0
  
  for j in range(length(layer_2_output)):
    for i in range(nrow(dataframe_subset)):
      cost += ((label[i]/a_L[i] )-(1-label[i])/(1-a_L[i])*(a_L[i]*\
                              (1-a_L[i]))*layer_3_weights[j])
    cost = float(cost)
    cost_wrt_layer_2_activations.append(cost)  # <- may need to reverse this cost list
    cost = 0
  #print("cost_wrt_layer_2_activation:  ", cost_wrt_layer_2_activations)
  
  for j in range(length(layer_3_weights)):
    for i in range(nrow(dataframe_subset)):
      cost += ((label[i]/a_L[i] )-(1-label[i])/(1-a_L[i])*\
                        (a_L[i]*(1-a_L[i]))*layer_2_output[j])
    cost = float(cost)
    cost_wrt_layer_3_weights.append(cost)
    cost = 0
    
  for i in range(num_nodes[0]):
    for j in range(num_nodes[1]):
      cost_wrt_layer_2_weights[i,j] = layer_1_output[i]*cost_wrt_layer_2_activations[j]

  for i in range(num_nodes[0]):
   for j in range((num_nodes[1])):
      cost += cost_wrt_layer_2_activations[i] * layer_2_activations[j] * layer_2_weights[i,j]
   cost = float(cost)
   cost_wrt_layer_1_activations.append(cost)
   cost = 0

  for k in range(nrow(dataframe_subset)):
    input_vector = dataframe_subset.iloc[k].to_numpy()
    for i in range(feature_n):
      for j in range(num_nodes[0]):
        cost_wrt_layer_1_weights[i,j] = input_vector[i]*cost_wrt_layer_1_activations[j]

  weight_3_viz = pd.DataFrame(cost_wrt_layer_3_weights)
  weight_2_viz = pd.DataFrame(cost_wrt_layer_2_weights)
  weight_1_viz = pd.DataFrame(cost_wrt_layer_1_weights)

  print("BACKPROPOGATION ran successfully and gradients found: !!!!!!!!!!!!!!", "\n\n\n")
  print("Derivative of loss (DOL) wrt to layer 3 weight: ","\n\n", weight_3_viz, "\n\n\n\n\n\n")
  print("DOL wrt layer 2: ","\n\n", weight_2_viz, "\n\n\n\n")
  print("DOL wrt layer 1: ","\n\n", weight_1_viz, "\n\n\n\n")
  print("DOL wrt layer 3:" , "\n\n", cost_wrt_layer_3_weights, "\n\n\n\n")
#-------------------------  UPDATE ----------------------------------
run_this()
print("this should print if nothing else")

def update():
  learning_rate = float(.002)
  global layer_1_weights, layer_2_weights, layer_3_weights, layer_1_biases, layer_2_biases 
  layer_1_weights = layer_1_weights - (cost_wrt_layer_1_weights * learning_rate)
  layer_2_weights = layer_2_weights - (cost_wrt_layer_2_weights * learning_rate)
  print("pre-update layer 3 weights:   ", layer_3_weights)
  print("class of layer_3 weights:   ", type(layer_3_weights))
  print("class of cost_wrt_layer_3_weights:   ", type(cost_wrt_layer_3_weights))
  print("shape of layer_3_weights is:   ", dim(layer_3_weights))
  layer_3_weights = layer_3_weights.flatten()
  layer_3_weights = layer_3_weights - (np.array(cost_wrt_layer_3_weights) * learning_rate)
  layer_1_biases  = layer_1_biases  - (np.array(cost_wrt_layer_1_activations) * learning_rate)
  layer_2_biases  = layer_2_biases  - (np.array(cost_wrt_layer_2_activations) * learning_rate)
  #layer_3_bias    = layer_3_bias    - cost_wrt_layer_3_activation
  print("update successful: new weights found")
  print("after updating dimensions of layer_ 1 weights are:  ", dim(layer_1_weights),"\n\n\n\n")
  print("after updating dimensions of layer_2 weights are: ", dim(layer_2_weights),"\n\n")
  print("after updating shape of layer_3 weights are: ", dim(layer_3_weights),"\n\n")
  print("layer_3 weights: ", layer_3_weights, "\n\n")
  print("after updating dimensions of layer_1 biases are:  ", dim(layer_1_biases),"\n\n\n\n")
  print("afte updating dimensions of layer_2 biases are: ", dim(layer_2_biases),"\n\n\n\n")
  return layer_1_weights, layer_2_weights, layer_3_weights, layer_1_biases, layer_2_biases
#--------------------------------------------------------- testing.....

sample_data_creation()
subset_1 = df.sample(n = 30)  # <- sampling 30 from the 200 observations
data_for_forward_pass = subset_1.drop("Binary", axis = 1) # <- data without labels (unsupervised) 
label = df['Binary']
loss_cache = []

epochs = int(3)

network_initialization()
# forward_pass(data_for_forward_pass)
# print("\n\n\n\n", "first pass complete")
# forward_pass(data_for_forward_pass)
# print("\n\n\n\n","second pass complete")



for i in range(epochs):
  forward_pass(data_for_forward_pass)
  loss_cache.append(loss(subset_1))
  backpropogation(subset_1)
  update()

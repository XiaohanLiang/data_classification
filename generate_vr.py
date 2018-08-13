# Generate numVR x VR points using the specified data file (.csv format)
# Saves result to specified output file
# numVR (optional) = Number of VR's required, if omitted then growing neural gas is used
# epochs  (optional)   = number of times the data is re-presented to training, defaykt 100
import mdp
import numpy as np
import sys
import os.path
from sentence_to_vec import *
sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------------------

srcDir = "./data/mnist/"
srcFilename = "mnist100.txt"
destFilename = "InputSpikingTime"
params = eval(open("settings.txt").read())
numVR  = params['numVR']
epochs = params['epochs']

# ------------------------------------------------------------------------------------------------------

#load observation data
obs = np.loadtxt(srcDir + '/' + srcFilename,delimiter=',')
obs = sentence_to_vec()
obs = np.array(obs)

numRows = obs.shape[0]
numFeatures = obs.shape[1]
print ('Data loaded:',obs.shape)

#shuffle (to create stationary data)
shuffledObs = mdp.numx.take(obs,mdp.numx_rand.permutation(numRows), axis=0)
#specifying 0 implies using the GNG algortihm to obtain an optimal number of VRs (nodes)
destPath  = "./" + "GNG-optimum-VR-set.csv"
if numVR==0:
    if os.path.isfile(destPath):
        os.system("mv "+destPath)

    print('Running growing neural gas to SUGGEST OPTIMUM NUM VRs')
    gng = mdp.nodes.GrowingNeuralGasNode()
    gng.train(shuffledObs)
    gng.stop_training()
    result  = gng.get_nodes_position()
    optimumVRs = result.shape[0]
    print ('VRs used:', optimumVRs)
    print(result)
    np.savetxt(destPath,result,delimiter=',',newline='\n')
else:
    print('Num VR specified as ', numVR, '- use std neural gas')
    gng = mdp.nodes.NeuralGasNode(num_nodes=numVR,max_epochs=epochs)
    gng.train(shuffledObs)
    gng.stop_training()
    result  = gng.get_nodes_position()
    np.savetxt(destPath,result,delimiter=',',newline='\n')
    print ('VRs found:',result.shape[0])


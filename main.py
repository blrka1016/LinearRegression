
# coding: utf-8

# In[11]:


import pandas as pd
from scipy.stats import multivariate_normal, norm
import math
import numpy as np # importing this way allows us to refer to numpy as np


# In[16]:


#A data object. takes input and output data frames, and partitions them into 
#training, validation and test sets. 
class Data:
    def __init__(self,inputdf,outputdf,trainingpercent,validationpercent):
        trainSize = int(len(outputdf)*trainingpercent)
        validSize = int(len(outputdf)*validationpercent)
        testSize = len(outputdf) - trainSize - validSize
        partitionIndex = np.arange(0,len(outputdf))
        np.random.shuffle(partitionIndex)
        trainIndices = partitionIndex[0:trainSize]
        validIndices = partitionIndex[trainSize:trainSize+validSize]
        testIndices = partitionIndex[trainSize+validSize:len(partitionIndex)]
        self.trainingInput = np.take(inputdf,trainIndices,axis=0)
        self.trainingOutput = np.take(outputdf,trainIndices)
        self.validationInput = np.take(inputdf,validIndices,axis=0)
        self.validationOutput = np.take(outputdf,validIndices)
        self.testInput = np.take(inputdf,testIndices,axis=0)
        self.testOutput = np.take(outputdf,testIndices)
        self.numVars = self.trainingInput.shape[1]
        self.numTrainingDataPoints = self.trainingInput.shape[0]


# In[93]:


#Centers & spreads. Contains several different methods for creating centers and spreads for the basis functions:
#Random: randomly initializes centers to values between -1 and 1, and spreads to values between 0 and 1.
#KMeans: clusters the input data into K clusters, 
#K = numBasisFunctions - 1
#then takes the centers and spreads of the data points in each cluster as the centers and spreads

class CentersSpreads:
    def __init__(self,numBasisFunctions,trainingInput,centersSpreadsCalcType):
        self.numBasisFunctions = numBasisFunctions
        self.trainingInput = trainingInput
        self.trainingInputSize = trainingInput.shape[0]
        self.numVars = trainingInput.shape[1]
        if(self.trainingInputSize>10000):
            self.trainingInput = self.trainingInput[np.random.choice(self.trainingInputSize, 5000, replace=False), :]
            self.trainingInputSize = 5000
        if(centersSpreadsCalcType=="kmeans"):
            (self.centers,self.spreads) = self.kMeansClusterRepeatUntilSame()
        elif (centersSpreadsCalcType=="randomPoints"):
            (self.centers,self.spreads) = self.Compute_centers_spreads_random_points()

                 
    #Assigns each data point in the training data to the closest of k points.
    #Returns the index of the cluster each data point is assigned to, 
    #and the mean (center) and variance (spread) of each cluster.
    def kmeansClusterOnce(self,kPoints):
        #Intialize the vector of K assignment of each data point. 
        #Initial value is K+1 for each entry (not an actual cluster)
        kAssignment = (np.ones([self.trainingInputSize]).astype(int))*(self.numBasisFunctions)
        #Loop over training datapoints and find the closest of the K samples.
        i=0
        while i<self.trainingInputSize:
            j = 0
            #print "clustering datapoint",i+1
            #Set the closest cluster (initially to K+1, since this is not a cluster)
            closestCluster = self.numBasisFunctions
            #Set the distance to the closest cluster
            closestClusterL2Norm = float('inf')
            while j<self.numBasisFunctions-1:
                #Calculate the distance between points
                l2norm = np.linalg.norm(kPoints[j] - self.trainingInput[i])
                #print "l2norm is",l2norm
                #if the distance is closer than the previous closest distance, change the closest cluster to this one
                if (l2norm<closestClusterL2Norm):
                    closestCluster = j
                    #print "new closest cluster",j
                    closestClusterL2Norm = l2norm
                    #print "norm",l2norm
                j = j+1
            #Assign the datapoint to the closest of the K points in the kAssignment vector
            kAssignment[i] = closestCluster
            i = i+1
        #print(kAssignment)
        #Loop over each cluster and then over each datapoint to find the center of the datapoints assigned to that cluster
        k = 0
        kNewCenters = np.zeros([self.numBasisFunctions-1,self.numVars])
        while k<self.numBasisFunctions-1:
            l=0
            clusterList = np.empty((0,self.numVars))
            while l<self.trainingInputSize:
                if(kAssignment[l]==k):
                    clusterList = np.vstack([clusterList, self.trainingInput[l,]])
                l=l+1
            #clusterCounts[k] = clusterList.shape[0]
            newCenter = np.mean(clusterList, axis=0)#sumCluster/float(countCluster)
            #print(newCenter)
            newSpread = np.cov(clusterList.T)
            #print(newSpread)
            kNewCenters[k,] = newCenter
            k = k+1
        #print "kNewCenters",kNewCenters
        #print clusterCounts
        return(kAssignment,kNewCenters)
    
    #Initializes k-means clustering by randomly selecting k points in the data. 
    #k = numBasisFunctions - 1 (because first basis function is intercept and always equals 1)
    #Clusters the data according to those points, and calculates the centroids of these clusters 
    #(using kmeansClusterOnce() function).
    #Re-clusters the data (using kmeansClusterOnce() function) with those centroids as the cluster points
    #Repeats this until the assignment vector does not change 
    #(i.e. until all data points are assigned to a cluster based on a centroid which is actuall the center of that cluster)
    def kMeansClusterRepeatUntilSame(self):
        #Select K initial points from the training input
        kPoints = self.trainingInput[np.random.choice(self.trainingInputSize, self.numBasisFunctions-1, replace=False), :]
        #print "initialPoints",kPoints
        #intialize assignment for all datapoints to the K+1th cluster
        kAssignment1 = np.ones([self.trainingInputSize])*(self.numBasisFunctions-1)
        #print "first clustering attempt..."
        (kAssignment2,kCenters) = self.kmeansClusterOnce(kPoints)
        counter = 2
        #print "Clustering..."
        while (not np.array_equal(kAssignment1,kAssignment2)):
            #print "Cluster attempt",counter
            kAssignmentOld = kAssignment2
            (kAssignment2,kCenters) = self.kmeansClusterOnce(kCenters)
            kAssignment1 = kAssignmentOld
            counter = counter+1
        #Calculate Spreads
        kSpreads = np.zeros([self.numBasisFunctions-1,self.numVars,self.numVars])
        k = 0
        while k<self.numBasisFunctions-1:
            #print "Calculating centers for cluster",k+1
            l=0
            clusterList = np.empty((0,self.numVars))
            while l<self.trainingInputSize:
                if(kAssignment2[l]==k):
                    clusterList = np.vstack([clusterList, self.trainingInput[l,]])
                l=l+1
            if (clusterList.shape[0]>1000):
                spread = np.cov(clusterList[np.random.choice(clusterList.shape[0], 1000, replace=False), :].T)
            else:
                spread = np.cov(clusterList.T)
            kSpreads[k,:,:] = spread
            k = k+1
        return (kCenters,kSpreads)
    
    #Generates centers & spreads so that all basis functions have the same spread (1) and the centers rise in half increments.
    #basisCenters matrix (size: m x i) has one row for each basis function, 
    #and the row represents the center of that basis function
    #basisSpreads matrix (size: m x i) has one row for each basis function, 
    #and the row represents the spread of that basis function
    #both have one column 
    #m is the number of basis functions
    #i is the number of input variables
    def Compute_centers_spreads_random_points(self):
        basisCenters = self.trainingInput[np.random.choice(self.trainingInputSize, self.numBasisFunctions-1, replace=False), :]
        basisSpreads = np.zeros([self.numBasisFunctions-1,self.numVars,self.numVars])
        sameSpread = np.cov(self.trainingInput[np.random.choice(self.trainingInputSize, 1000, replace=False), :].T)
        i = 0
        while i<self.numBasisFunctions-1:
            basisSpreads[i,:,:] = sameSpread
            i=i+1
        return (basisCenters,basisSpreads)


# In[18]:


#A design matrix object. Generates a design matrix based on a dataframe, a number of basis functions, 
#a matrix of centers and a matrix of spreads. 
#Can be used to generate the design matrix for training, validation or test data.
class DesignMatrix:
    
    def __init__(self,inputData,numberOfBasis,centers,spreads):
        #print "initializing design matrix..."
        self.numBasis = numberOfBasis
        self.inputData = inputData
        self.dataLength = self.inputData.shape[0]
        self.numVars = self.inputData.shape[1]
        self.centers = centers
        self.spreads = spreads
        self.spreadsInverses = np.zeros(self.spreads.shape)
        #print "assigning spreads & inverses..."
        i = 0
        while i<self.numBasis-1:
            self.spreadsInverses[i,:,:] = np.linalg.pinv(self.spreads[i,:,:])
            i = i+1
        #print "computing design matrix..."
        self.designMatrix = self.Compute_design_matrix()

    #Generates a cell (a scalar value) in the design matrix using a radial basis function
    #based on a row of the dataframe (one data point) (length i),
    #a vector which represents the center of the function (length i), 
    #and a vector which represents the spread of the function (length i).
    #i is the number of variables in the input data (i.e. the number of columns in the input data frame)
    def basisFunction(self,dataRow,centerVector,spreadMatrixInverse):
        retVal = np.exp(-1/2*np.matmul(np.matmul(np.transpose(dataRow-centerVector),spreadMatrixInverse),(dataRow-centerVector)))
        return retVal #must be a scalar

    #Computes a design matrix (nxm) by calculating each cell using the basisFunction.
    #The first column represents the intercept and is equal to 1 for each row 
    #n is the number of data points in the input data
    #m is the number of basis functions
    #centers and spreads should have m-1 values in each row
    def Compute_design_matrix(self):
        designMatrix = np.ones([self.dataLength,self.numBasis])
        #print "designMatrix shape is",designMatrix.shape
        i = 0
        while i < self.dataLength:
            j = 0
            designMatrix[i][j] = 1.
            j = 1
            while j < self.numBasis:
                designMatrix[i][j] = self.basisFunction(self.inputData[i,],self.centers[j-1,],self.spreadsInverses[j-1,:,:])
                j=j+1
            i = i+1
        return designMatrix


# In[50]:


#A weights object. Initializes to all zeros. Also contains functions to update the weights,
#using various gradient descent methods (stochastic, batch, minibatch), and functions to calculate
#the sum of squared error and root mean squared error for the weights, design matrix and output data.
class Weights:
    #Initializes the weights vector (length m) and the gradnorm 
    #(initialized to infinity, note this is not actually the gradnorm until the weights have been updated at least once)
    #m is the number of basis functions.
    def __init__(self,numberOfBasis):
        self.weights = np.zeros([numberOfBasis])
        self.gradnorm = float('inf')
 
    #MiniBatch Gradient Descent with Early Stopping
    def minibatchGDEarlyStopping(self,num_epochs,outputData,designMatrix,validOutput,validDesignMatrix,lmbda,learningRate,batchSize,tolerance,pointselectiontype):
        n = designMatrix.shape[0]
        RMSValidationExisting = 100
        toleranceCounter = 0
        filename = "Epochs_"+str(num_epochs)+"_lmbda_"+str(lmbda)+"_lr2_"+str(learningRate)+"_batchsize_"+str(batchSize)+"_tolerance_"+str(tolerance)+"_pointselection_"+pointselectiontype+"_basisfns_"+str(designMatrix.shape[1])+".txt"
        file = open(filename,"w") 
        file.write("epoch,trainRMS,validationRMS\n")
        if (learningRate == "adjustable"):
            adjustable = True
            learningRate = 1.0
        else:
            adjustable = False
        ep = 0
        for epoch in range(num_epochs):
            (designMatrix,outputData) = self.shufflesDataAndOutputEqually(designMatrix,outputData)
            for i in range(n/batchSize):
                lower_bound= i*batchSize
                upper_bound= min((i+1)*batchSize,n)
                Phi = designMatrix[lower_bound:upper_bound]
                t = outputData[lower_bound:upper_bound] 
                E_D = -1.*np.matmul((t - np.matmul(Phi,self.weights.T)).T,Phi)
                E = (E_D+lmbda*self.weights)/batchSize
                self.weights = self.weights-learningRate*E
            RMSValidationNew = self.rootMeanSquaredError(validOutput,validDesignMatrix)
            #print RMSValidationNew
            if RMSValidationNew > RMSValidationExisting:
                toleranceCounter = toleranceCounter + 1
                print "RMSValidation increased consecutively:",toleranceCounter
                RMSValidationExisting = RMSValidationNew
                if toleranceCounter > tolerance:
                    print "Validation RMS increased more than",tolerance,"in a row, stopping gradient descent."
                    break
            RMSValidationExisting = RMSValidationNew
            if(ep%5000 ==1):
                if adjustable==True and (learningRate - 1000./ep)>0.0000001:
                    learningRate = learningRate - 1000./ep
                    #print "Epoch:",ep,"New learning rate:",learningRate
                #else:
                    #print "Epoch:",ep
                #rmsTrain = self.rootMeanSquaredError(outputData,designMatrix,0.0)
                #print "RMS Train",rmsTrain
                #print "RMS Validation",RMSValidationNew
                #print self.weights
            rmsTrain = self.rootMeanSquaredError(outputData,designMatrix)
            rmsValidation = self.rootMeanSquaredError(validOutput,validDesignMatrix)
            file.write(str(ep)+","+str(rmsTrain)+","+str(rmsValidation)+"\n")
            ep = ep+1
        file.write(str(self.weights))
        file.close()
        #print "RMS Train",rmsTrain
        #print "RMS Validation",rmsValidation
        return (self.weights.flatten(),ep,rmsTrain,rmsValidation,np.linalg.norm(E))
    
    ##Define closed form solution function with regularization.
    ##Set lmbda equal to 0 for a closed form solution with no regularization.
    def closedFormSolutionReglrzn(self,designMatrix,outputData,lmbda):
        weightsVector = np.matmul(np.matmul(np.linalg.pinv(lmbda*np.identity(designMatrix.shape[1])+np.matmul(designMatrix.T,designMatrix)),designMatrix.T),outputData)
        return weightsVector

    #Shuffles the data and the output (identically) so that the batches for each epoch are not the same
    def shufflesDataAndOutputEqually(self,designMatrix,outputData):
        assert len(designMatrix) == len(outputData)
        p = np.random.permutation(len(designMatrix))
        return designMatrix[p], outputData[p]

    #Calculates the sum of squared errors with weight decay regularization for a given weights vector 
    #with the design matrix and the output data
    #Returns the sum of squared errors with weight decay regularization.
    #set lmbda equal to zero for sum of squared errors without weight decay regularization
    def sumSquaredErrorWithReg(self,outputData,designMatrix,lmbda):
        n = designMatrix.shape[0]
        error = np.dot(self.weights,designMatrix.T) - outputData
        #print error.shape
        sumsqderror = np.sum(np.multiply(error,error))
        #print sumsqderror
        weightDecayReg = lmbda*np.matmul(np.transpose(self.weights),self.weights)
        #print weightDecayReg
        return sumsqderror + weightDecayReg
    
    #Calculates the root mean squared error based on the sum of squared errors.
    def rootMeanSquaredError(self,outputData,designMatrix):
        sumSqdErr = self.sumSquaredErrorWithReg(outputData,designMatrix,0.0)
        rms = math.sqrt((sumSqdErr/float(designMatrix.shape[0])))
        return rms


# In[30]:


##Import synthetic Dataset
dfSyntheticInput = pd.read_csv('input.csv', encoding = 'utf8', header = None).values
dfSyntheticOutput = pd.read_csv('output.csv', encoding = 'utf8', header = None).values
synthetic = Data(dfSyntheticInput,dfSyntheticOutput,0.8,0.1)


# In[7]:


##Import LeToR Dataset
dfLetorInput = pd.read_csv('Querylevelnorm_X.csv', encoding = 'utf8', header = None).values
dfLetorOutput = pd.read_csv('Querylevelnorm_t.csv', encoding = 'utf8', header = None).values
Letor = Data(dfLetorInput,dfLetorOutput,0.8,0.1)


# In[31]:


##Demonstrate with no regularization or early stopping or lambda
centersSpreads = CentersSpreads(5,synthetic.trainingInput,"randomPoints")
trainDesignMatrix = DesignMatrix(synthetic.trainingInput,5,centersSpreads.centers,centersSpreads.spreads)
validationDesignMatrix = DesignMatrix(synthetic.validationInput,5,centersSpreads.centers,centersSpreads.spreads)
weights = Weights(5)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,0.1)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weights.minibatchGDEarlyStopping(400000,synthetic.trainingOutput,trainDesignMatrix.designMatrix,synthetic.validationOutput,validationDesignMatrix.designMatrix,0.0,0.5,trainDesignMatrix.designMatrix.shape[0],100000,"random")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeights
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[20]:


##Run intial model (Synthetic data) with M=5, lambda = 0.1, lr = 0.5, tolerance = 10, random centers, dataset covariance as spread
#centersSpreads = CentersSpreads(5,synthetic.trainingInput,"randomPoints")
#trainDesignMatrix = DesignMatrix(synthetic.trainingInput,5,centersSpreads.centers,centersSpreads.spreads)
#validationDesignMatrix = DesignMatrix(synthetic.validationInput,5,centersSpreads.centers,centersSpreads.spreads)
weights = Weights(5)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,0.1)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weights.minibatchGDEarlyStopping(400000,synthetic.trainingOutput,trainDesignMatrix.designMatrix,synthetic.validationOutput,validationDesignMatrix.designMatrix,0.1,0.5,trainDesignMatrix.designMatrix.shape[0],10,"random")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeights
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[90]:


##Run intial (Letor) model with M=5, lambda = 0.1, lr = 0.5, tolerance = 10, random centers, dataset covariance as spread
centersSpreadsL = CentersSpreads(5,Letor.trainingInput,"randomPoints")
trainDesignMatrixL = DesignMatrix(Letor.trainingInput,5,centersSpreadsL.centers,centersSpreadsL.spreads)
validationDesignMatrixL = DesignMatrix(Letor.validationInput,5,centersSpreadsL.centers,centersSpreadsL.spreads)
closedFormWeightsL = weights.closedFormSolutionReglrzn(trainDesignMatrixL.designMatrix,Letor.trainingOutput,0.1)
weightsL = Weights(5)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weightsL.minibatchGDEarlyStopping(400000,Letor.trainingOutput,trainDesignMatrixL.designMatrix,Letor.validationOutput,validationDesignMatrixL.designMatrix,0.1,0.5,trainDesignMatrixL.designMatrix.shape[0],10,"random")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeightsL
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[28]:


#Demonstrate an adaptive LR improves convergence: use adaptive learning rate with same other parameters as before
centersSpreads = CentersSpreads(5,synthetic.trainingInput,"randomPoints")
trainDesignMatrix = DesignMatrix(synthetic.trainingInput,5,centersSpreads.centers,centersSpreads.spreads)
validationDesignMatrix = DesignMatrix(synthetic.validationInput,5,centersSpreads.centers,centersSpreads.spreads)
weights = Weights(5)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,0.1)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weights.minibatchGDEarlyStopping(500000,synthetic.trainingOutput,trainDesignMatrix.designMatrix,synthetic.validationOutput,validationDesignMatrix.designMatrix,0.1,"adjustable",trainDesignMatrix.designMatrix.shape[0],3,"random")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeights
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[8]:


##Demonstrate K-Means improves results over random
centersSpreads = CentersSpreads(5,synthetic.trainingInput,"kmeans")
trainDesignMatrix = DesignMatrix(synthetic.trainingInput,5,centersSpreads.centers,centersSpreads.spreads)
validationDesignMatrix = DesignMatrix(synthetic.validationInput,5,centersSpreads.centers,centersSpreads.spreads)
weights = Weights(5)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,0.1)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weights.minibatchGDEarlyStopping(400000,synthetic.trainingOutput,trainDesignMatrix.designMatrix,synthetic.validationOutput,validationDesignMatrix.designMatrix,0.1,"adjustable",trainDesignMatrix.designMatrix.shape[0],3,"kmeans")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeights
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[ ]:


##Demonstrate K-Means improves results over random for LeToR
centersSpreadsL = CentersSpreads(5,Letor.trainingInput,"kmeans")
trainDesignMatrixL = DesignMatrix(Letor.trainingInput,5,centersSpreadsL.centers,centersSpreadsL.spreads)
validationDesignMatrixL = DesignMatrix(Letor.validationInput,5,centersSpreadsL.centers,centersSpreadsL.spreads)
closedFormWeightsL = weights.closedFormSolutionReglrzn(trainDesignMatrixL.designMatrix,Letor.trainingOutput,0.1)
weightsL = Weights(5)
(weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weightsL.minibatchGDEarlyStopping(400000,Letor.trainingOutput,trainDesignMatrixL.designMatrix,Letor.validationOutput,validationDesignMatrixL.designMatrix,0.1,"adjustable",trainDesignMatrixL.designMatrix.shape[0],3,"kmeans")
print "NumEpochs:",epochs
print "Weights:",weightsFinal
print "Closed Form Weights:",closedFormWeights
print "RMSTrain:",rmsTrain
print "RMSValidation:",rmsValidation
print "GradNorm:",gradNorm


# In[52]:


#Grid search for kmeans and lambda with adaptive LR
numBasisToTest = np.array([5,10,20])
lambdaToTest = np.array([0.0,0.01,0.1,0.5,0.8])
results = []
for m in numBasisToTest:
    if(trainDesignMatrix.designMatrix.shape[1] == m):
        centersSpreads = centersSpreads
    else:
        centersSpreads = CentersSpreads(m,synthetic.trainingInput,"kmeans")
    trainDesignMatrix = DesignMatrix(synthetic.trainingInput,m,centersSpreads.centers,centersSpreads.spreads)
    validationDesignMatrix = DesignMatrix(synthetic.validationInput,m,centersSpreads.centers,centersSpreads.spreads)
    for l in lambdaToTest:
        print "Testing",m,"basis functions and lambda=",l
        weights = Weights(m)
        closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,l)
        (weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weights.minibatchGDEarlyStopping(400000,synthetic.trainingOutput,trainDesignMatrix.designMatrix,synthetic.validationOutput,validationDesignMatrix.designMatrix,l,"adjustable",trainDesignMatrix.designMatrix.shape[0],3,"kmeans")
        print "NumEpochs:",epochs
        print "Weights:",weightsFinal
        print "Closed Form Weights:",closedFormWeights
        print "RMSTrain:",rmsTrain
        print "RMSValidation:",rmsValidation
        print "GradNorm:",gradNorm


# In[ ]:


#Grid search for kmeans and lambda with adaptive LR LeToR set
numBasisToTest = np.array([3,4,5,10,15,20])
lambdaToTest = np.array([0.0,0.01,0.1,0.5,0.8])
results = []
for m in numBasisToTest:
    centersSpreadsL = CentersSpreads(5,Letor.trainingInput,"kmeans")
    trainDesignMatrixL = DesignMatrix(Letor.trainingInput,m,centersSpreadsL.centers,centersSpreadsL.spreads)
    validationDesignMatrixL = DesignMatrix(Letor.validationInput,m,centersSpreadsL.centers,centersSpreadsL.spreads)
    for l in lambdaToTest:
        closedFormWeightsL = weights.closedFormSolutionReglrzn(trainDesignMatrixL.designMatrix,Letor.trainingOutput,0.1)
        weightsL = Weights(5)
        (weightsFinal,epochs,rmsTrain,rmsValidation,gradNorm) = weightsL.minibatchGDEarlyStopping(400000,Letor.trainingOutput,trainDesignMatrixL.designMatrix,Letor.validationOutput,validationDesignMatrixL.designMatrix,l,"adjustable",trainDesignMatrixL.designMatrix.shape[0],3,"kmeans")
        print "NumEpochs:",epochs
        print "Weights:",weightsFinal
        print "Closed Form Weights:",closedFormWeights
        print "RMSTrain:",rmsTrain
        print "RMSValidation:",rmsValidation
        print "GradNorm:",gradNorm


# In[55]:


#Tuning M and Lambda with the closed form solution
#BasisToTest is an array of possible values for M
#lambdaToTest is an array of possible values for lambda
#returns a list of tuples, where each tuple is the results for that M & lambda value combination.
def tuneWithClosedForm(data,basisToTest,lambdaToTest):
    results = []
    for m in numBasisToTest:
        print "Calculating clusters: ",m
        centers_Spreads = CentersSpreads(m,data.trainingInput,"kmeans")
        trainDesignMatrix = DesignMatrix(data.trainingInput,m,centers_Spreads.centers,centers_Spreads.spreads)
        validationDesignMatrix = DesignMatrix(data.validationInput,m,centers_Spreads.centers,centers_Spreads.spreads)
        for l in lambdaToTest:
            print "Testing",m,"basis functions and lambda=",l
            weights = Weights(m)
            closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,data.trainingOutput,l)
            weights.weights = closedFormWeights
            rmsTrain = weights.rootMeanSquaredError(data.trainingOutput,trainDesignMatrix.designMatrix)
            rmsValidation = weights.rootMeanSquaredError(data.validationOutput,validationDesignMatrix.designMatrix)
            print "RMSTrain:",rmsTrain
            print "RMSValidation:",rmsValidation
            results.append((m,l,rmsTrain,rmsValidation))
    return results


# In[94]:


numBasisToTest = np.array([3,5,10,15,25,30])
lambdaToTest = np.array([0.0,0.01,0.1,0.5,0.8])
tuneWithClosedForm(synthetic,numBasisToTest,lambdaToTest)


# In[96]:


numBasisToTest = np.array([3,5,10,20,25,30])
lambdaToTest = np.array([0.0,0.01,0.1,0.5,0.8])
tuneWithClosedForm(Letor,numBasisToTest,lambdaToTest)


# In[92]:


##Calculate test RMSE for synthetic dataset with 25 basis functions & lambda = 0.01
centers_Spreads_25_synth = CentersSpreads(25,synthetic.trainingInput,"kmeans")
trainDesignMatrix = DesignMatrix(synthetic.trainingInput,25,centers_Spreads_25_synth.centers,centers_Spreads_25_synth.spreads)
validDesignMatrix = DesignMatrix(synthetic.validationInput,25,centers_Spreads_25_synth.centers,centers_Spreads_25_synth.spreads)
testDesignMatrix = DesignMatrix(synthetic.testInput,25,centers_Spreads_25_synth.centers,centers_Spreads_25_synth.spreads)
weights = Weights(25)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrix.designMatrix,synthetic.trainingOutput,0.01)
weights.weights = closedFormWeights
rmsTrain = weights.rootMeanSquaredError(synthetic.trainingOutput,trainDesignMatrix.designMatrix)
rmsValid = weights.rootMeanSquaredError(synthetic.validationOutput,validDesignMatrix.designMatrix)
rmsTest = weights.rootMeanSquaredError(synthetic.testOutput,testDesignMatrix.designMatrix)
print rmsTrain
print rmsValid
print rmsTest


# In[91]:


##Demonstrate RMSE for synthetic dataset with 25 basis functions & lambda = 0.01 & random point initialization
centers_Spreads_25_synth_rand = CentersSpreads(25,synthetic.trainingInput,"randomPoints")
trainDesignMatrixR = DesignMatrix(synthetic.trainingInput,25,centers_Spreads_25_synth_rand.centers,centers_Spreads_25_synth_rand.spreads)
validDesignMatrixR = DesignMatrix(synthetic.validationInput,25,centers_Spreads_25_synth_rand.centers,centers_Spreads_25_synth_rand.spreads)
testDesignMatrixR = DesignMatrix(synthetic.testInput,25,centers_Spreads_25_synth_rand.centers,centers_Spreads_25_synth_rand.spreads)
weights = Weights(25)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrixR.designMatrix,synthetic.trainingOutput,0.01)
weights.weights = closedFormWeights
rmsTrain = weights.rootMeanSquaredError(synthetic.trainingOutput,trainDesignMatrixR.designMatrix)
rmsValid = weights.rootMeanSquaredError(synthetic.validationOutput,validDesignMatrixR.designMatrix)
rmsTest = weights.rootMeanSquaredError(synthetic.testOutput,testDesignMatrixR.designMatrix)
print rmsTrain
print rmsValid
print rmsTest


# In[88]:


##Calculate test RMSE for LeToR dataset with 20 basis functions & lambda = 0.1
centers_Spreads_25_letor = CentersSpreads(25,Letor.trainingInput,"kmeans")
trainDesignMatrixL = DesignMatrix(Letor.trainingInput,25,centers_Spreads_20_letor.centers,centers_Spreads_25_letor.spreads)
validDesignMatrixL = DesignMatrix(Letor.validationInput,25,centers_Spreads_20_letor.centers,centers_Spreads_25_letor.spreads)
testDesignMatrixL = DesignMatrix(Letor.testInput,25,centers_Spreads_20_letor.centers,centers_Spreads_25_letor.spreads)
weights = Weights(25)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrixL.designMatrix,Letor.trainingOutput,0.0)
weights.weights = closedFormWeights
rmsTrain = weights.rootMeanSquaredError(Letor.trainingOutput,trainDesignMatrixL.designMatrix)
rmsValid = weights.rootMeanSquaredError(Letor.validationOutput,validDesignMatrixL.designMatrix)
rmsTest = weights.rootMeanSquaredError(Letor.testOutput,testDesignMatrixL.designMatrix)
print rmsTrain
print rmsValid
print rmsTest


# In[89]:


##Demonstrate RMSE for LeToR dataset with 20 basis functions & lambda = 0.01 & random point initialization
centers_Spreads_20_synth_rand = CentersSpreads(20,Letor.trainingInput,"randomPoints")
trainDesignMatrixLR = DesignMatrix(Letor.trainingInput,20,centers_Spreads_20_synth_rand.centers,centers_Spreads_20_synth_rand.spreads)
validDesignMatrixLR = DesignMatrix(Letor.validationInput,20,centers_Spreads_20_synth_rand.centers,centers_Spreads_20_synth_rand.spreads)
weights = Weights(20)
closedFormWeights = weights.closedFormSolutionReglrzn(trainDesignMatrixLR.designMatrix,Letor.trainingOutput,0.0)
weights.weights = closedFormWeights
rmsTrain = weights.rootMeanSquaredError(Letor.trainingOutput,trainDesignMatrixLR.designMatrix)
rmsValid = weights.rootMeanSquaredError(Letor.validationOutput,validDesignMatrixLR.designMatrix)
print rmsTrain
print rmsValid


# In[66]:


#import necessary graphing libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[ ]:


#graph RMSE training and validation for first run of synthetic
df = pd.read_csv('Epochs_400000_lmbda_0.1_lr_0.5_batchsize_16000_tolerance_3.txt', encoding = 'utf8')
df = df.drop(df.index[len(df)-1])
logtransform = lambda x: math.log(float(x)+1)
#graph['trainRMS'] = graph['trainRMS'].apply(logtransform)
#graph['validationRMS'] = graph['validationRMS'].apply(logtransform)
df['epoch'] = df['epoch'].apply(logtransform)
df.rename(columns={'epoch': 'epoch (log transformed)'}, inplace=True)
plt.figure()
df.plot(x='epoch (log transformed)')
plt.show()


# In[ ]:


#Graph to illustrate training and validation RMSE
df_a = pd.read_csv('Epochs_400000_lmbda_0.1_lr_0.5_batchsize_16000_tolerance_10_pointselection_random.txt', encoding = 'utf8')
df_a = df_a.drop(df_a.index[len(df_a)-1])
df_a.rename(columns={'trainRMS': 'trainRMS_rand','validationRMS': 'validationRMS_rand' }, inplace=True)
df_b = pd.read_csv('Epochs_400000_lmbda_0.1_lr_0.5_batchsize_16000_tolerance_3.txt', encoding = 'utf8')
df_b = df_b.drop(df_b.index[len(df_b)-1])
df_b.rename(columns={'trainRMS': 'trainRMS_kmeans','validationRMS': 'validationRMS_kmeans' }, inplace=True)
df_c = pd.read_csv('Epochs_400000_lmbda_0.1_lr_adjustable_batchsize_16000_tolerance_3.txt', encoding = 'utf8')
df_c = df_c.drop(df_c.index[len(df_c)-1])
df_c.rename(columns={'trainRMS': 'trainRMS_adjustLR','validationRMS': 'validationRMS_adjustLR' }, inplace=True)
frames = [df_a, df_b[['trainRMS_kmeans','validationRMS_kmeans']], df_c[['trainRMS_adjustLR','validationRMS_adjustLR']]]
result = pd.concat(frames, axis=1)
result['epoch'] = result['epoch'].apply(logtransform)
result.rename(columns={'epoch': 'epoch (log transformed)'}, inplace=True)
plt.figure()
result.plot(x='epoch (log transformed)')
plt.show()


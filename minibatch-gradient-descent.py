import numpy as np
import random as rd
import matplotlib.pyplot as plt

#cost function
def J(X, y, beta):
  #my previous J definition np.dot(y-np.dot(X, beta).transpose(), y-np.dot(X, beta))
  return np.dot((y-np.dot(X, beta)).transpose(), (y-np.dot(X, beta)))

#mini-batch generation function
def build_mini_batches(rawdata, n):
  minibatches=[]

  #permute the raw data
  rawdata=np.random.permutation(rawdata[1:,[0,1,2,3]])

  #extract response parameters
  y=np.array(rawdata[0:,3], dtype=float)
  #extract feature parameters
  X_=np.array(rawdata[0:,[0,1,2]], dtype=float)
  #add column of 1's
  X_=np.concatenate((np.ones((y.size, 1)),X_), axis=1)

  N=y.size
  num_batches=N/n

  for i in range(int(num_batches)):
    X_minibatch=X_[i * n:(i + 1)*n, :]
    y_minibatch=y[i * n:(i + 1)*n]
    minibatches.append((X_minibatch, y_minibatch))
  return minibatches

#learning rate, taken from the assignment instructions
alpha=2.5*(10**-6)
#batch size, taken from the assignment instructions
n=10
#number of parameters
p=3
#number of iterations, taken from assignment instructions
iterations=20000

rawdata = np.loadtxt("Advertising_N200_p3.csv", delimiter=",", dtype=str)
y=np.array(rawdata[1:,3], dtype=float)
X_= np.array(rawdata[1:,[0,1,2]], dtype=float)
X_ = np.concatenate((np.ones((y.size, 1)),X_), axis=1)
beta=np.random.uniform(-1, 1, p+1)
costs=[]
betas=[]
betas.append(beta)

#here is where mini-batch gradient descent begins
for iter in range(iterations):
  #randomly load observations into the batches
  #print("===Running iteration",iter+1)
  #print("Building minibatches")
  minibatches=build_mini_batches(rawdata, n)
  #print("Minibatch creation complete")
  for minibatch in minibatches:
    X_minibatch, y_minibatch = minibatch

    term1=np.dot(X_minibatch, beta)
    term2=y_minibatch-term1
    term3=X_minibatch.transpose()
    term4=np.dot(2*alpha, term3)
    term5=np.dot(term4, term2)
    beta=beta+term5

  betas.append(beta)
  costs.append(J(X_, y, beta))

#final calculations
beta_hat=beta

figure, visual=plt.subplots(2, 1)
figure.set_size_inches(24, 18, forward=True)

#deliverable 1
#illustrate the effect of iteration number on the inferred regression coefficients
visual[0].plot(list(range(1,len(betas)+1)),betas, marker = '.')
visual[0].set_title('Deliverable 1')
visual[0].set_xlabel('Iterations')
visual[0].set_ylabel('Regression Coefficients')
visual[0].legend(['B0', 'B1', 'B2', 'B3'])

#deliverable 2
#illustrate the effect of iteration number on the cost
visual[1].plot(list(range(1,len(costs)+1)),costs, marker = '.')
visual[1].set_title('Deliverable 2')
visual[1].set_xlabel('Iterations')
visual[1].set_ylabel('Cost')
visual[1].legend(['Cost'])

plt.show

#deliverable 3
#provide the estimates for beta_hat
print("Deliverable 3 - best fit model parameters are",beta_hat,"\n")

#deliverable 4
#compute the MSE on the training set using the best-fit model parameters
y_hat=np.dot(X_, beta_hat)
#MSE=(np.subtract(y, y_hat)**2)/2
MSE=np.square(np.subtract(y, y_hat)).mean()
print("Deliverable 4 - Mean Squared Error=",MSE)

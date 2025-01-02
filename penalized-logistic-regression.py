import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tabulate import tabulate as tbl
from IPython.display import clear_output

def loaddatafromfile(filename):
  rawdata=np.loadtxt(filename, delimiter=",", dtype=str)
  X=np.array(rawdata[1:,0:rawdata.shape[1]-1], dtype=float)
  y=np.array(rawdata[1:,-1], dtype=str)
  feature_names=np.array(rawdata[0,0:rawdata.shape[1]-1], dtype=str)
  return X,y,feature_names

def build_design_matrix(X):
  return np.concatenate((np.ones((X.shape[0], 1)),X), axis=1)

def center_and_standardize_observations(observations):
  # standardize features
  X_tilde=np.transpose(observations)
  for i,parameter_vector in enumerate(X_tilde):
    mean=np.mean(parameter_vector,dtype=float)
    std_dev=np.std(parameter_vector, dtype=float)
    for j in range(parameter_vector.size):
      X_tilde[i,j]=(float(X_tilde[i,j])-float(mean))/std_dev
  X_tilde=np.transpose(X_tilde)
  return X_tilde

def build_indicator_response_matrix(y):
  Y=np.zeros((N, K),dtype=int)
  for index,val in enumerate(y):
    Y[index,int(val)]=1
  return Y

def build_parameter_matrix():
  return np.zeros((p+1, K),dtype=float)

def build_intercept_matrix():
  return np.zeros((p+1, K), dtype=float)

def tokenize_responses(classes,y):
  #replaces each unique class label with an integer class id
  for index,classs in enumerate(classes):
    y[y==classs]=index
  return y.astype(int)

def print_del1data(del1data):
  for k,data in enumerate(del1data):
    table=[]
    table.append(𝜆s)
    for row in data:
      table.append(row)
    print("class=",classes[k])
    print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))
  return

def extract_feature_matrix(rawdata):
  return np.array(rawdata[0:,0:rawdata.shape[1]-1], dtype=float)

def extract_response_vector(rawdata):
  return np.array(rawdata[0:,[-1]], dtype=int)

def permute_dataset(rawdata):
  return np.random.permutation(rawdata[0:,])

def calculate_column_means(matrix):
  data=np.transpose(matrix)
  means=[]
  for parameter_vector in data:
    mean=np.mean(parameter_vector,dtype=float)
    means.append(mean)
  return means

def calculate_column_stddevs(matrix):
  data=np.transpose(matrix)
  stddevs=[]
  for parameter_vector in data:
    stddev=np.std(parameter_vector,dtype=float)
    stddevs.append(stddev)
  return stddevs

def center_standardize_matrix(input,mean,stdev,standardize):
  matrix=np.transpose(input)
  i=0
  for vector in matrix:
    for j in range(vector.size):
      if standardize:
        matrix[i,j]=(float(matrix[i,j])-float(mean[i]))/stdev[i]
      else:
        matrix[i,j]=(float(matrix[i,j])-float(mean[i]))
    i+=1
  return np.transpose(matrix)

def extract_folds(dataset,fold,folds):
  fold_size=int(dataset.shape[0]/folds)
  start=fold*fold_size
  end=(fold+1)*fold_size if fold < folds -1 else None

  verification_rows=slice(start, end)
  verification_dataset=dataset[verification_rows,:]

  training_dataset1=dataset[slice(0, start),:]
  if end==None:
    training_rows=[slice(0, start)]
    training_dataset=training_dataset1
  else:
    training_dataset2=dataset[slice(end, None),:]
    training_rows=[slice(0, start),slice(end, None)]
    training_dataset=np.concatenate((training_dataset1,training_dataset2),axis=0)

  return training_rows,verification_rows,verification_dataset,training_dataset

def compute_probability_matrix(𝐗,𝐁):
  U=np.exp(np.dot(𝐗,𝐁))
  P=np.empty((𝐗.shape[0],K), dtype=float)
  for i,row in enumerate(P):
    for k,column in enumerate(row):
      P[i,k]=U[i,k]/np.sum(U[i,:])
  return P

def calculate_model_parameters(𝐁,𝛼,𝐗,𝐘,𝐏,𝜆,𝐙):
  return 𝐁 + np.multiply(𝛼,(np.dot(np.transpose(𝐗), (𝐘 - 𝐏)) - np.multiply((2 * 𝜆),(𝐁 - 𝐙))))

def perform_logistic_regression_cv(iterations,𝐗,𝐁,𝛼,𝜆,𝐘,𝐗_cv,training_rows,verification_rows):
  𝐙=build_intercept_matrix()
  for iteration in range(iterations):
    𝐏=compute_probability_matrix(𝐗_cv,𝐁)
    𝐁=calculate_model_parameters(𝐁,𝛼,𝐗,𝐘,extract_training_rows(𝐏,training_rows),𝜆,𝐙)
  return 𝐏,𝐁

def process_fold(combined_dataset,fold,folds,𝜆,𝐘):
  𝐗_cv=build_design_matrix(extract_feature_matrix(combined_dataset))
  training_rows,verification_rows,verification_dataset,training_dataset=extract_folds(combined_dataset,fold,folds)

  X_k_training=extract_feature_matrix(training_dataset)
  y_k_training=extract_response_vector(training_dataset)
  X_k_verification=extract_feature_matrix(verification_dataset)
  y_k_verification=extract_response_vector(verification_dataset)

  mean_X_k_training=calculate_column_means(X_k_training) #for each feature
  std_dev_X_k_training=calculate_column_stddevs(X_k_training) #for each feature
  mean_y_k_training=calculate_column_means(y_k_training) #for the response vector

  X_k_training=center_standardize_matrix(X_k_training,mean_X_k_training,std_dev_X_k_training,True)
  X_k_training=build_design_matrix(X_k_training)
  #y_k_training=center_standardize_matrix(y_k_training,mean_y_k_training,0,False)
  X_k_verification=center_standardize_matrix(X_k_verification,mean_X_k_training,std_dev_X_k_training, True)
  X_k_verification=build_design_matrix(X_k_verification)
  #y_k_verification=center_standardize_matrix(y_k_verification,mean_y_k_training,0,False)

  𝐁=build_parameter_matrix()

  training_𝐘1=𝐘[training_rows[0],:]
  if len(training_rows)>1:
    training_𝐘2=𝐘[training_rows[1],:]
    training_𝐘=np.concatenate((training_𝐘1,training_𝐘2),axis=0)
  else:
    training_𝐘=training_𝐘1

  𝐏,𝐁=perform_logistic_regression_cv(iterations,X_k_training,𝐁,𝛼,𝜆,training_𝐘,𝐗_cv,training_rows,verification_rows)
  CCE=compute_CCE(𝐘,𝐏,verification_rows)
  return CCE

def extract_training_rows(matrix, training_rows):
  return_matrix=matrix[training_rows[0],:]
  if len(training_rows)>1:
    matrix2=matrix[training_rows[1],:]
    return_matrix=np.concatenate((return_matrix,matrix2),axis=0)
  return return_matrix

def compute_CCE_back(𝐘,𝐏,verification_rows):
  𝐏k=𝐏[verification_rows,:]
  𝐘k=𝐘[verification_rows,:]
  Nm=𝐘k.shape[0]
  term1=np.dot(np.transpose(𝐘k),𝐏k)
  CCE=-abs(np.sum(np.sum(term1,axis=0))/(Nm*𝐘k.shape[1]))
  return CCE

def compute_CCE(𝐘,𝐏,verification_rows):
  𝐏k=𝐏[verification_rows,:]
  𝐘k=𝐘[verification_rows,:]
  Nm=𝐘k.shape[0]
  CCE=-np.sum(𝐘k * np.log(𝐏k))/Nm
  return CCE

def perform_logistic_regression(iterations,𝐗,𝐁,𝛼,𝜆,𝐘):
  𝐙=build_intercept_matrix()
  for iteration in range(iterations):
    𝐏=compute_probability_matrix(𝐗,𝐁)
    𝐁=calculate_model_parameters(𝐁,𝛼,𝐗,𝐘,𝐏,𝜆,𝐙)
  return 𝐏,𝐁

# fix tuning parameters 𝜆 and 𝛼
𝛼=10**-5
𝜆s=[10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
iterations=10**4

#load the data and setup the matrices
training_data_filename="TrainingData_N183_p10.csv"
X,y,feature_names=loaddatafromfile(training_data_filename)
N=X.shape[0]
p=feature_names.shape[0]
classes=np.unique(y)
K=classes.shape[0]
y=tokenize_responses(classes,y)
𝐗=center_and_standardize_observations(X)
𝐗_cv=X
𝐗=build_design_matrix(𝐗)
𝐘=build_indicator_response_matrix(y)

#deliverable 1 calculations
𝐁hat=np.empty((len(𝜆s)),dtype=object)
𝐏s=np.empty((len(𝜆s)),dtype=object)
for index,𝜆 in enumerate(𝜆s):
  𝐁=build_parameter_matrix()
  𝐏,𝐁=perform_logistic_regression(iterations,𝐗,𝐁,𝛼,𝜆,𝐘)
  𝐁hat[index]=𝐁
  𝐏s[index]=𝐏

# # deliverable 1 graph calcualtions, need to create 5 arrays of 10,9
del1data=np.empty((K), dtype=object)
for del1dataindex in range(len(del1data)):
  del1data[del1dataindex]=np.empty((p+1,0), dtype=object)

for 𝜆_index,𝜆_𝐁s in enumerate(𝐁hat):
  𝜆_𝐁s=np.transpose(𝜆_𝐁s)
  for index,class_𝐁 in enumerate(𝜆_𝐁s):
    del1data[index]=np.column_stack((del1data[index],class_𝐁))

figure, visual=plt.subplots(6, 1)
figure.set_size_inches(16, 36, forward = True)
figure.tight_layout(pad=5.0)

# deliverable 1 visualizations
for graphindex,classs in enumerate(classes):
  𝜆_axis=np.log10(𝜆s)
  𝜆_axis=np.round(𝜆_axis,0)
  𝜆_axis=np.array(𝜆_axis,dtype=str)
  𝜆_axis=np.char.add("1e-",𝜆_axis)
  graphdata=del1data[graphindex]
  for graphrow in graphdata:
    visual[graphindex].plot(𝜆_axis,graphrow,marker = '.')
  visual[graphindex].set_title('Deliverable 1 class='+classs)
  visual[graphindex].set_xlabel('Log10(Lambda)')
  visual[graphindex].set_ylabel('Bhat(j,k)')
  visual[graphindex].legend(np.asarray(feature_names))

#deliverable 2 calculations
folds=5
combined_dataset=np.column_stack((𝐗_cv,y))
combined_dataset=permute_dataset(combined_dataset)
𝐘=build_indicator_response_matrix(extract_response_vector(combined_dataset))
𝜆_CCEs=np.zeros(len(𝜆s))
for index,𝜆 in enumerate(𝜆s):
  CCE_folds=np.zeros(folds)
  for fold in range(folds):
    CCE_folds[fold]=process_fold(combined_dataset,fold,folds,𝜆,𝐘)
  𝜆_CCEs[index]=np.mean(CCE_folds)

#Deliverable 2 visualizations
visual[5].set_title('Deliverable 2')
visual[5].set_xlabel('Log10(Lambda)')
visual[5].set_ylabel('CV('+str(folds)+') Error')
visual[5].plot(np.char.add("1e-",np.array(np.round(np.log10(𝜆s),0),dtype=str)),𝜆_CCEs,marker = '.')
labels=["Categorical Cross Entropy"]
visual[5].legend(np.asarray(labels))
plt.show

#Deliverable 3
print("Deliverable 3")
print("Lowest cross-validation error of",𝜆_CCEs[np.argmin(𝜆_CCEs)],"found using 𝜆=",𝜆s[np.argmin(𝜆_CCEs)])

#Deliverable 4 calculations
#retraining the model with the best lamdba value
𝐘=build_indicator_response_matrix(y)
𝐁=𝐁hat[np.argmin(𝜆_CCEs)]
𝐏=𝐁hat[np.argmin(𝜆_CCEs)]

#run the classifier on the test data
testing_data_filename="TestData_N111_p10.csv"
Xtrain,ytrain,feature_names=loaddatafromfile(training_data_filename)
Xtest,ytest,feature_names=loaddatafromfile(testing_data_filename)

mean_X_training=calculate_column_means(Xtrain)
std_dev_X_training=calculate_column_stddevs(Xtrain)
Xtrain=center_standardize_matrix(Xtrain,mean_X_training,std_dev_X_training,True)
Xtest=center_standardize_matrix(Xtest,mean_X_training,std_dev_X_training, True)
Xtest=build_design_matrix(Xtest)

𝐏=compute_probability_matrix(Xtest,𝐁)
test_index=np.arange(1, Xtest.shape[0], 1, dtype=int)

print("Deliverable 4")
table=[]
headers=["Test Observation"]
headers.extend(classes)
headers.extend(["Most Probable Ancestry"])
headers.extend(["Explanation of Ancestry"])
table.append(headers)
for index,x in enumerate(𝐏):
  row=[str(index+1)]
  row.extend(x)
  #row.extend([ytest[index]])
  row.extend([str(classes[int(np.argmax(x))])])
  if ytest[index] == "Unknown":
    row.extend(["N/A"])
  else:
    max2=x.argsort()[-2:][::-1]
    explanation=ytest[index]+" background based on "+classes[max2[0]]+" and "+classes[max2[1]]+" ancestry"
    row.extend([explanation])
  table.append(row)
print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))

#Deliverable 5
print("Deliverable 5")
print("The Unknown samples have one ancestry that have probabilities of at least 90%")
print("In contract, the other samples do not have any ancestries over 90%")
print("This is explained by a combination of historical immigration and colonization")

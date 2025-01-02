import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tabulate import tabulate as tbl

def tokenize_gender(gender):
  if gender=="Male":
    return 0
  else:
    return 1

def tokenize_student_married(student_married):
  if student_married=="Yes":
    return 1
  else:
    return 0

def load_and_tokenize_observations(filename):
  rawdata=np.loadtxt(filename, delimiter=",", dtype=str)
  features=np.array(rawdata[0,[0,1,2,3,4,5,6,7,8]], dtype=str)
  processed_data=np.array(rawdata[1:,[0,1,2,3,4,5]], dtype=float)
  gender_student_married=np.array(rawdata[1:,[6,7,8]], dtype=str)
  # convert gender, student and married status into binary tokens
  for row in gender_student_married:
    row[0]=tokenize_gender(row[0])
    row[1]=tokenize_student_married(row[1])
    row[2]=tokenize_student_married(row[2])
  gender_student_married=gender_student_married.astype(float)
  y=np.array(rawdata[1:,9], dtype=float)
  processed_data=np.concatenate((processed_data, gender_student_married), axis=1)
  processed_data=np.column_stack((processed_data, y))
  return processed_data,features

def permute_dataset(rawdata):
  return np.random.permutation(rawdata[0:,[0,1,2,3,4,5,6,7,8,9]])

def extract_folds(dataset, fold):
  verification_rows=np.arange(fold*int(N/k),((fold+1)*int(N/k)))
  verification_dataset=centered_standardized_data[verification_rows,]
  training_rows=np.concatenate([np.arange(0, fold * int(N/k)), np.arange((fold + 1) * int(N/k), centered_standardized_data.shape[0])])
  training_dataset=centered_standardized_data[training_rows,]
  return verification_dataset, training_dataset

def center_and_standardize_observations(dataset):
  # standardize features
  X_=np.array(dataset[0:,[0,1,2,3,4,5,6,7,8]], dtype=float)
  X_=np.transpose(X_)
  i=0
  for parameter_vector in X_:
    mean=np.mean(parameter_vector,dtype=float)
    std_dev=np.std(parameter_vector, dtype=float)
    for j in range(parameter_vector.size):
      X_[i,j]=(float(X_[i,j])-float(mean))/std_dev
    i+=1
  X_=np.transpose(X_)
  # center response
  y=np.array(dataset[0:,9], dtype=float)
  y_mean=y.mean()
  for j in range(y.size):
    y[j]=y[j]-y_mean
  processed_dataset=np.column_stack((X_, y))
  return processed_dataset

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

def compute_gradient_descent(𝛽,𝜆,𝛼,X_,y):
  # step 4 from the assignment instructions: Update the parameter vector as
  # 𝛽 ≔ 𝛽 − 2𝛼[𝜆𝛽 − 𝐗𝑇(𝐲 − 𝐗𝛽)]
  term1=y-np.dot(X_,𝛽)
  term2=np.dot(np.transpose(X_), term1)
  term3=np.dot(𝜆,𝛽)
  term4=term3-term2
  term5=np.dot((2*𝛼),term4)
  term6=𝛽-term5
  term6=np.where(np.isfinite(term6), term6, 0)
  return term6

def extract_feature_matrix(rawdata):
  return np.array(rawdata[0:,[0,1,2,3,4,5,6,7,8]], dtype=float)

def extract_response_vector(rawdata):
  return np.array(rawdata[0:,[9]], dtype=float)

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

def MSE(X, y, 𝛽):
  #MSE=np.divide(np.dot((y-np.dot(X,𝛽)).transpose(),(y-np.dot(X,𝛽))),X.shape[0])
  term1=np.dot(X,𝛽)
  term2=y-term1
  term3=term2.transpose()
  term4=np.dot(term3, term2)
  term5=np.divide(term4,X.shape[0])
  MSE=term5
  return MSE

def MSE_v2(X, y, 𝛽):
  #from last assignment
  y_hat=np.dot(X, 𝛽)
  MSE=(np.subtract(y, y_hat)**2)/2
  MSE=np.square(np.subtract(y, y_hat)).mean()
  return MSE

def compute_cross_validation_errors(MSEs):
  CVEs=[]
  MSE_matrix=MSEs.transpose()
  for MSE in MSE_matrix:
    CVEs.append(np.mean(MSE))
  return CVEs

# setup the visualizations
figure, visual=plt.subplots(2, 1)
figure.set_size_inches(18, 18, forward=True)

# load the observation data
processed_data,feature_names=load_and_tokenize_observations("Credit_N400_p9.csv")
N=processed_data.shape[0]
p=processed_data.shape[1]-1

# step 1: choose learning rate 𝛼 and fix tuning parameter lamdba
𝛼=10**-5
𝜆s=[10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4]
iterations=10**5
k=5

# deliverable 1
centered_standardized_data=center_and_standardize_observations(processed_data)
X_=extract_feature_matrix(centered_standardized_data)
y=extract_response_vector(centered_standardized_data)
𝛽hat=np.empty(shape=[p,0])
for 𝜆 in 𝜆s:
  𝛽=np.random.uniform(-1, 1, (p,1))
  for iteration in range(iterations):
    𝛽=compute_gradient_descent(𝛽,𝜆,𝛼,X_,y)
  𝛽hat=np.column_stack((𝛽hat,𝛽))

#create visualization for deliverable 1
𝜆_axis=np.log10(𝜆s)
𝜆_axis=np.round(𝜆_axis,0)
𝜆_axis=np.array(𝜆_axis,dtype=str)
𝜆_axis=np.char.add("1e-",𝜆_axis)
for coefficient in range(0, 𝛽hat.shape[0]):
  visual[0].plot(𝜆_axis,𝛽hat[coefficient],marker = '.')
visual[0].set_title('Deliverable 1')
visual[0].set_xlabel('Log10(Lambda)')
visual[0].set_ylabel('Standardized Coefficients')
visual[0].legend(np.asarray(feature_names))

#deliverable 2
MSEs=np.zeros((k, len(𝜆s)))
MSE_i=0
for fold in range(k):
  #not needed I believe, permutated_data=permute_dataset(processed_data)
  verification_dataset,training_dataset=extract_folds(centered_standardized_data, fold)

  X_k_training=extract_feature_matrix(training_dataset)
  y_k_training=extract_response_vector(training_dataset)
  X_k_verification=extract_feature_matrix(verification_dataset)
  y_k_verification=extract_response_vector(verification_dataset)

  mean_X_k_training=calculate_column_means(X_k_training) #for each feature
  std_dev_X_k_training=calculate_column_stddevs(X_k_training) #for each feature
  mean_y_k_training=calculate_column_means(y_k_training) #for the response vector

  X_k_training=center_standardize_matrix(X_k_training,mean_X_k_training,std_dev_X_k_training,True)
  y_k_training=center_standardize_matrix(y_k_training,mean_y_k_training,0,False)
  X_k_verification=center_standardize_matrix(X_k_verification,mean_X_k_training,std_dev_X_k_training, True)
  y_k_verification=center_standardize_matrix(y_k_verification,mean_y_k_training,0,False)

  MSE_j=0
  for 𝜆 in 𝜆s:
    𝛽=np.random.uniform(-1, 1, (p,1))
    for iteration in range(iterations):
      𝛽=compute_gradient_descent(𝛽,𝜆,𝛼,X_k_training,y_k_training)
    MSEs[MSE_i,MSE_j]=MSE(X_k_verification,y_k_verification,𝛽)
    MSE_j+=1
  MSE_i+=1
MSEs=np.round(MSEs, 3)
CVEs=np.round(compute_cross_validation_errors(MSEs),3)

#create visualization for deliverable 2
x_axis=np.log10(𝜆s)
x_axis=np.round(x_axis,0)
x_axis=np.array(x_axis,dtype=str)
x_axis=np.char.add("1e-",x_axis)
y_axis=CVEs
for fold_errors in range(0, MSEs.shape[0]):
  visual[1].plot(x_axis,MSEs[fold_errors],marker = '.')
visual[1].set_title('Deliverable 2')
visual[1].set_xlabel('Log10(Lambda)')
visual[1].set_ylabel('CV(5) Error')
labels=[]
for label in range(k):
  labels.append("Fold "+str(label+1))
visual[1].legend(np.asarray(labels))

# deliverable 3
min_cv=np.argmin(CVEs)
print("Deliverable 3")
heading1, heading2, heading3="𝜆", "CV("+str(k)+")", "Lowest?"
table=[]
table.append([heading1, heading2, heading3])
for x in range(len(𝜆s)):
  table.append([𝜆s[x],CVEs[x],"Yes" if min_cv==x else "No"])
print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))

# deliverable 4
centered_standardized_data=center_and_standardize_observations(processed_data)
X_=extract_feature_matrix(centered_standardized_data)
y=extract_response_vector(centered_standardized_data)
𝛽hat=np.empty(shape=[p,0])
𝜆=𝜆s[min_cv]
𝛽=np.random.uniform(-1, 1, (p,1))
for iteration in range(iterations):
  𝛽=compute_gradient_descent(𝛽,𝜆,𝛼,X_,y)
𝛽hat=np.column_stack((𝛽hat,𝛽))
print("Deliverable 4")
heading1,heading2="Feature", "𝛽"
table=[]
table.append([heading1, heading2])
for x in range(len(𝛽hat)):
  table.append([feature_names[x],𝛽hat[x]])
print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))

plt.show

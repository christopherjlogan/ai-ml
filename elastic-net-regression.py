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

def permute_dataset(rawdata):
  return np.random.permutation(rawdata[0:,[0,1,2,3,4,5,6,7,8,9]])

def extract_folds(dataset, fold):
  verification_rows=np.arange(fold*int(N/k),((fold+1)*int(N/k)))
  verification_dataset=Xy_tilde[verification_rows,]
  training_rows=np.concatenate([np.arange(0, fold * int(N/k)), np.arange((fold + 1) * int(N/k), centered_standardized_data.shape[0])])
  training_dataset=Xy_tilde[training_rows,]
  return verification_dataset, training_dataset

def precompute_b(matrix):
  bb=np.zeros((p, 1), dtype=float)
  data=np.transpose(matrix)
  index=0
  for parameter_vector in data:
    b_k=np.sum(np.square(parameter_vector,dtype=float))
    bb[index][:]=b_k
    index+=1
  return bb

def compute_a_k(X_,y,𝛽,p,k):
  #𝑎𝑘 = x𝑘𝑇(𝐲 − 𝐗𝛽 + x𝑘𝛽𝑘)
  term1=np.dot(X_[:,k],float(𝛽[k]))#(400,)
  term1=np.reshape(term1,(term1.shape[0],1))
  term2=np.dot(X_,𝛽)#(400, 1)
  term3=np.transpose(X_[:,k])#(400,)
  term3=np.reshape(term3,(1,term3.shape[0]))
  return np.dot(term3,(y-term2+term1))

def perform_coordinate_descent(𝛽,a_k,𝜆,𝛼,b_k):
  '''sign_x=1
  if a_k<0:
    sign_x=-1'''
  term1=abs(a_k)-np.divide(np.multiply(𝜆,1-𝛼),2)
  if term1<0:
    term1=0
  with np.errstate(divide='raise'):
    try:
      returnval=np.divide(np.dot(np.sign(a_k),term1),(b_k+np.dot(𝜆,𝛼)))
      if np.isnan(returnval) or np.isinf(returnval):
        return 0
      else:
        return returnval
    except:
      return 0

def print_𝛽hats(𝛽hats):
  for 𝛼 in range(len(𝛼s)):
    table_data=np.empty(shape=[p,0])
    for 𝜆 in range(len(𝜆s)):
      table_data=np.column_stack((table_data,𝛽hats[𝜆,𝛼]))
    #table_data=np.transpose(table_data)
    table=[]
    table.append(feature_names)
    for 𝜆 in range(len(𝜆s)):
      table.append(𝛽hats[𝜆,𝛼])
    print("𝛼=",𝛼s[𝛼])
    print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))
  return

def MSE(X, y, 𝛽):
  #MSE=np.divide(np.dot((y-np.dot(X,𝛽)).transpose(),(y-np.dot(X,𝛽))),X.shape[0])
  term1=np.dot(X,𝛽)
  term2=y-term1
  term3=term2.transpose()
  term4=np.dot(term3, term2)
  term5=np.divide(term4,X.shape[0])
  MSE=term5
  return MSE

def lowest_index(row):
  lowest_value=np.inf
  lowest_index=0
  for x in range(len(row)):
      if row[x]<lowest_value:
         lowest_value=row[x]
         lowest_index=x
  return lowest_index

# ---------------end of functions---------------

# fix tuning parameters 𝜆 and 𝛼
𝛼s=[0,1/5,2/5,3/5,4/5,1]
𝜆s=[10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
iterations=10**3

# setup the visualizations
figure, visual=plt.subplots(7, 1)
figure.set_size_inches(16, 36, forward = True)
figure.tight_layout(pad=5.0)

# load the observation data
processed_data,feature_names=load_and_tokenize_observations("Credit_N400_p9.csv")
N=processed_data.shape[0]
p=processed_data.shape[1]-1

Xy_tilde=center_and_standardize_observations(processed_data)
X_=extract_feature_matrix(Xy_tilde)
y=extract_response_vector(Xy_tilde)

# Deliverable 1 calculations
𝛽hats=np.empty([len(𝜆s), len(𝛼s)], dtype=list)
𝛼_index=0
for 𝛼 in 𝛼s:
  𝜆_index=0
  for 𝜆 in 𝜆s:
    b=precompute_b(X_)
    𝛽=np.random.uniform(-1, 1, (p,1))
    for iteration in range(iterations):
      for k in range(p):
        a_k=compute_a_k(X_,y,𝛽,p,k)
        𝛽[k]=perform_coordinate_descent(𝛽,a_k,𝜆,𝛼,b[k])
    𝛽hats[𝜆_index][𝛼_index]=𝛽
    𝜆_index+=1
  𝛼_index+=1

# Deliverable 1 visualizations
for 𝛼 in range(len(𝛼s)):
  visual[𝛼].set_title('Deliverable 1 alpha='+str(𝛼s[𝛼]))
  visual[𝛼].set_xlabel('Log10(lambda)')
  visual[𝛼].set_ylabel('B^')
  chart_data=np.empty(shape=[p,0])
  for 𝜆 in range(len(𝜆s)):
    chart_data=np.column_stack((chart_data,𝛽hats[𝜆,𝛼]))
  chart_data=np.transpose(chart_data)
  visual[𝛼].plot(np.char.add("1e-",np.array(np.round(np.log10(𝜆s),0),dtype=str)),chart_data,marker = '.')
  visual[𝛼].legend(np.asarray(feature_names))

# Deliverable 2 calculations
k=5
centered_standardized_data=center_and_standardize_observations(processed_data)
𝛽hats=np.zeros([len(𝛼s),len(𝜆s)], dtype=list)
CVEs=np.zeros((len(𝛼s),len(𝜆s)))
min_cve=np.inf
min_cve_i=0
min_cve_j=0
MSE_fold=np.zeros(k)
MSE_i=0
for 𝛼 in 𝛼s:
  MSE_j=0
  for 𝜆 in 𝜆s:
    for fold in range(k):

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

      b=precompute_b(X_k_training)
      𝛽=np.random.uniform(-1,1,(p,1))
      for iteration in range(iterations):
        for parameter in range(p):
          a_k=compute_a_k(X_k_training,y_k_training,𝛽,p,parameter)
          𝛽[parameter]=perform_coordinate_descent(𝛽,a_k,𝜆,𝛼,b[parameter])
      MSE_fold[fold]=MSE(X_k_verification,y_k_verification,𝛽)
    𝛽hats[MSE_i][MSE_j]=𝛽
    CVEs[MSE_i,MSE_j]=np.mean(MSE_fold)
    if CVEs[MSE_i,MSE_j]<min_cve:
      min_cve=CVEs[MSE_i,MSE_j]
      min_cve_i=MSE_i
      min_cve_j=MSE_j
    MSE_j+=1
  MSE_i+=1
CVEs=np.round(CVEs, 3)

#create visualization for deliverable 2
visual[6].set_title('Deliverable 2')
visual[6].set_xlabel('Log10(Lambda)')
visual[6].set_ylabel('CV('+str(k)+') Error')
for plot in CVEs:
  visual[6].plot(np.char.add("1e-",np.array(np.round(np.log10(𝜆s),0),dtype=str)),plot,marker = '.')
labels=[]
for 𝛼 in 𝛼s:
  labels.append("alpha="+str(𝛼))
visual[6].legend(np.asarray(labels))

# Deliverable 3
print("Deliverable 3")
print("Minimum CVE of",CVEs[min_cve_i][min_cve_j],"was for 𝛼=",𝛼s[min_cve_i],"and 𝜆=",𝜆s[min_cve_j])
table=[]
headers=["𝛼|𝜆"]
headers.extend(𝜆s)
table.append(headers)
alpha_index=0
for alpha_row in CVEs:
  row=[𝛼s[alpha_index]]
  row.extend(alpha_row)
  table.append(row)
  alpha_index+=1
print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))

# Deliverable 4 calculations
centered_standardized_data=center_and_standardize_observations(processed_data)
X_=extract_feature_matrix(centered_standardized_data)
y=extract_response_vector(centered_standardized_data)
𝛽hat=np.empty(shape=[p,0])
𝜆=𝜆s[min_cve_j]
𝛼=𝛼s[min_cve_i]
𝛽=np.random.uniform(-1, 1, (p,1))
b=precompute_b(X_)
for k in range(p):
  for iteration in range(iterations):
    a_k=compute_a_k(X_,y,𝛽,p,k)
    𝛽[k]=perform_coordinate_descent(𝛽,a_k,𝜆,𝛼,b[k])
𝛽hat=np.column_stack((𝛽hat,𝛽))

# Deliverable 4 visualizations
print("Deliverable 4")
best_lasso_𝜆=lowest_index(CVEs[0,:])
best_ridge_𝜆=lowest_index(CVEs[-1,:])

heading1,heading2,heading3,heading4="Feature", "𝛽 for 𝛼="+str(𝛼s[min_cve_i])+",𝜆="+str(𝜆s[min_cve_j])+"(lowest CVE)", "𝛽 for 𝛼=0,𝜆="+str(𝜆s[best_lasso_𝜆])+"(best lasso)","𝛽 for 𝛼=1,𝜆="+str(𝜆s[best_ridge_𝜆])+"(best ridge)"
table=[]
table.append([heading1,heading2,heading3,heading4])
best_lasso_𝛽=𝛽hats[0][best_lasso_𝜆]
best_ridge_𝛽=𝛽hats[𝛽hats.shape[0]-1][best_ridge_𝜆]
for x in range(len(𝛽)):
  table.append([feature_names[x],𝛽[x],best_lasso_𝛽[x],best_ridge_𝛽[x]])
print(tbl(table, headers='firstrow', tablefmt='fancy_grid'))

plt.show

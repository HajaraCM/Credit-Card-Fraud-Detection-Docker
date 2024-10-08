#!/usr/bin/env python
# coding: utf-8

# ## Credit Card  Anamoly Detection
# 
# ### Context
# It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
# 
# ### Content
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# ### Inspiration
# Identify fraudulent credit card transactions.
# 
# Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
# 

# In[1]:


# data manipulation
import numpy as np
import pandas as pd

#Library for Data Visualization.


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
# Library for Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv(r"C:\Users\HP\Downloads\credit\creditcard.csv")
#display top 5 rows in the data
data.head()


# In[3]:


#display top 5 rows in the data
data.tail()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.describe().T


# In[8]:


data.isnull().sum()


# In[9]:


# Calculate the number of unique values in each column
for column in data.columns:
    print(f"{column} - Number of unique values : {data[column].nunique()}")
    print("=============================================================")


# In[10]:


data['Time'].unique()


# In[11]:


# In[15]:


# Get the numerical columns
numerical_cols = data.columns

# Number of columns for the subplot grid
num_cols = 2

# Calculate the number of rows needed
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols




# In[16]:




# In[17]:


def convert_and_adjust_time(float_time):
    # Step 1: Convert the float to an integer and then to a zero-padded string
    time_str = f"{int(float_time):06d}"
    
    # Split the string into hours, minutes, and seconds
    hours = int(time_str[:2])    # First two characters for hours
    minutes = int(time_str[2:4]) # Next two for minutes
    seconds = int(time_str[4:])   # Last two for seconds

    # Step 2: Handle the overflow of seconds
    if seconds >= 60:
        minutes += seconds // 60  # Convert excess seconds to minutes
        seconds = seconds % 60     # Keep the remainder for seconds
        
    # Handle overflow of minutes
    if minutes >= 60:
        hours += minutes // 60
        minutes = minutes % 60

    # Ensure hours are non-negative (for proper time representation)
    if hours < 0:
        hours = 0


    # Step 3: Create a proper time 
    return pd.to_datetime(f"{hours}:{minutes}:{seconds}",format='%H:%M:%S')

# Step 4: Apply the conversion function to the 'Time' column
data['Time'] = data['Time'].apply(convert_and_adjust_time)

#Step 2: Create new columns for hours and minutes
data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute



# In[18]:


data.info()


# In[19]:


data.drop('Time', axis=1, inplace=True)


# In[20]:


data[data.duplicated()]


# In[21]:


data.drop_duplicates(inplace=True)


# In[22]:


data.shape


# ## Model Prediction
# 
# 
# 
# 
# 
# Now it is time to start building the model .The types of algorithms we are going to use to try to do anomaly detection on this dataset are as follows
# 
# ### Isolation Forest Algorithm :
# One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.
# 
# This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.
# 
# Typical machine learning methods tend to work better when the patterns they try to learn are balanced, meaning the same amount of good and bad behaviors are present in the dataset.
# 
# How Isolation Forests Work
# The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.
# 
# The way that the algorithm constructs the separation is by first creating isolation trees, or random decision trees. Then, the score is calculated as the path length to isolate the observation.
# 
# 
# ### Local Outlier Factor(LOF) Algorithm
# The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.
# 
# The number of neighbors considered, (parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.

# In[23]:


pca_columns = [f'V{i}' for i in range(1, 29)]
non_pca_columns = ['Amount','Hour', 'Minute']


# In[24]:


X=data.drop('Class', axis=1)
y=data['Class']


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[26]:


# Scale non-PCA columns in the training set
scaler = StandardScaler()
X_train[non_pca_columns] = scaler.fit_transform(X_train[non_pca_columns])

# Transform the test set using the same scaler
X_test[non_pca_columns] = scaler.transform(X_test[non_pca_columns])


# In[27]:


# Combine scaled non-PCA columns with PCA columns for both training and test sets
X_train_combined = pd.concat([X_train[pca_columns], X_train[non_pca_columns]], axis=1)
X_test_combined = pd.concat([X_test[pca_columns], X_test[non_pca_columns]], axis=1)


# In[28]:


# Step 2: Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_combined, y_train)


# In[29]:


from collections import Counter
print("The number of classes before fit {}",format(Counter(y_train)))
print("The number of classes after fit {}",format(Counter(y_resampled)))


# In[30]:


df=pd.DataFrame(X_resampled, columns=X_train.columns)


# In[31]:


df.isnull().sum()


# In[32]:


df


# In[33]:


rf_model = RandomForestClassifier(n_estimators=42)
rf_model.fit(X_resampled,y_resampled)
y_pred = rf_model.predict(X_test_combined)
rf_acc=accuracy_score(y_test,y_pred)
print("Accuracy is:",rf_acc*100)
print("Results from RandomForest Classifier")
result = classification_report(y_pred,y_test,output_dict=True)
result = pd.DataFrame(result).transpose()
result.style.background_gradient(cmap="PuBuGn")


# In[36]:


print(confusion_matrix(y_test,y_pred))


# In[37]:


y_train_pred = rf_model.predict(X_resampled)
print('train accuracy score:',accuracy_score(y_resampled, y_train_pred))


# In[38]:




# In[39]:




# In[40]:





# In[41]:




# In[42]:



# In[43]:



# In[44]:



# In[45]:




# In[46]:



# In[47]:











# In[49]:


x_new=df[['Amount','Hour','Minute']]


# In[50]:


inertia = []
k_values = range(1, 11)  # Test for 1 to 10 clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_new)
    inertia.append(kmeans.inertia_)



# In[51]:


import joblib

# Save the model
joblib.dump(rf_model,'model.pkl')


# In[52]:


model = joblib.load('model.pkl')


# In[53]:


# Fit KMeans on the resampled training data
clusterer = KMeans(n_clusters=3, random_state=42)
clusterer.fit(x_new)  
print(f"Features for KMeans: {x_new}")  


# In[54]:


# Add cluster labels to the resampled training data
x_new['Cluster'] = clusterer.labels_ 
x_new.head()  # Display first few rows of cluster assignments


# In[55]:


# Check unique cluster labels assigned
print(f"Cluster labels: {np.unique(clusterer.labels_)}")  # Should show [0, 1, 2]

# Count the number of unique clusters
n_clusters = len(np.unique(clusterer.labels_))
print(f"Number of clusters found: {n_clusters}")  # Should be 3


# In[56]:


if len(df) == len(x_new):
    # Concatenate along the columns
    combined_df = pd.concat([df.reset_index(drop=True), 
                              x_new['Cluster'].reset_index(drop=True)], axis=1)
else:
    print("The number of rows in x_resampled_dataframe and x_new do not match!")

# Display the combined DataFrame
print(combined_df.head())


# In[57]:


combined_df.isnull().sum()


# In[58]:


# Save the model
joblib.dump(clusterer,'cluster.pkl')
cluster=joblib.load('cluster.pkl')


# In[86]:


def predict_fraud(amount,hour, minute):
    # Create a DataFrame for the new transaction
    new_transaction = pd.DataFrame({
        'Amount': [amount],
        'Hour': [hour],
        'Minute': [minute],
    })

    # Standardize the new transaction using the scaler
    new_transaction_scaled = scaler.transform(new_transaction[non_pca_columns])
    
    # Get the scaled hour, minute, and amount
    scaled_amount = new_transaction_scaled[0][0]
    scaled_hour = new_transaction_scaled[0][1]
    scaled_minute = new_transaction_scaled[0][2]
    # Check for existing transactions that match closely (old user)
    similar_transactions = df[    
        (np.abs(df['Amount'] - scaled_amount) < 1e-5)& # Use tolerance for Amount
        (np.abs(df['Hour'] - scaled_hour) < 1e-5) &  # Use scaled values and tolerance
        (np.abs(df['Minute'] - scaled_minute) < 1e-5) ]

    if not similar_transactions.empty:
        print("Old User: Using existing PCA components.")
        # Get PCA components from the most recent matching transaction
        pca_components = similar_transactions.loc[similar_transactions.index[-1], [f'V{i+1}' for i in range(28)]].values
    else:
        print("New User: Using clustering to find PCA components.")
        
        # Predict the cluster for the new transaction
        cluster_label = cluster.predict(new_transaction_scaled)[0]
        print(f"Predicted Cluster: {cluster_label}")

        # Retrieve the members of the predicted cluster
        cluster_members = combined_df[combined_df['Cluster'] == cluster_label]

        if cluster_members.empty:
            print("No cluster members found for prediction.")
            return

        # Find the most similar transaction in the cluster based on hour, minute, and amount
        distances = np.abs(cluster_members['Hour'] - scaled_hour) + np.abs(cluster_members['Minute'] - scaled_minute) + np.abs(cluster_members['Amount'] - amount)
        similar_index = distances.idxmin()

        # Get PCA components from the most similar transaction
        pca_components = cluster_members.loc[similar_index, [f'V{i+1}' for i in range(28)]].values
        print(f"PCA Components for Prediction: {pca_components}")

    # Combine PCA components with the scaled hour, minute, and amount for the model input
    input_for_model = np.concatenate((pca_components, new_transaction_scaled[0]))
    print(f"Input for Model: {input_for_model}")

    # Predict using the Random Forest model
    prediction = model.predict([input_for_model])
    probability = model.predict_proba([input_for_model])[0][1]

    # Print the result
    print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
    print(f"Probability of Fraud: {probability:.4f}")

# Example usage
predict_fraud (45.65,1,17)


# In[60]:


# Save the 
joblib.dump(scaler,'scaler.pkl')


# In[61]:



# In[ ]:





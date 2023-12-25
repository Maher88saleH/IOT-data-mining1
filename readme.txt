import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
data = pd.read_csv('C:\\IoT Network Intrusion Dataset Undersampled.csv')
print(data.head())

num_rows, num_columns = data.shape

# Print the results
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')

# Specify the columns you want to plot
data['Pkt_Size_Avg'] = pd.to_numeric(data['Pkt_Size_Avg'], errors='coerce')
data['AM/PM'] = pd.to_numeric(data['AM/PM'], errors='coerce')
data['Normal'] = pd.to_numeric(data['Normal'], errors='coerce')
columns_to_plot = ['Pkt_Size_Avg', 'AM/PM', 'Normal']



# Plot each specified column
for column in columns_to_plot:
    plt.plot(data[column], label=column)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Title of the Plot')

# Show legend
plt.legend()
# Show the plot
plt.show()
# Plot the histogram
plt.hist(data['Normal'], bins=10, color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('AM/PM')
plt.ylabel('Frequency')
plt.title(f'Histogram of AM/PM')

# Show the plot
plt.show()


columns_to_drop = ['Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts']

# Drop the specified columns
data.drop(columns=columns_to_drop, inplace=True)
# Drop duplicate rows
data_no_duplicates = data.drop_duplicates()
print(data.head())
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')
# the information of datasets without duplicated
print(data.info())
# Drop the specified of list columns
list = ['Dst_0', 'Dst_1', 'Dst_2', 'Dst_3']
data_no_duplicates1 = data.drop_duplicates(subset=list, keep='first', inplace=True)
print(data.info())
print(data.describe())
# Count the null cells in each column
null_counts = data.isnull().sum()

# Display the count of null cells for each column
print(null_counts)

# Drop rows with null values
data_cleaned = data.dropna()
data.dropna(inplace=True)
# Display the DataFrame without null values
print(data_cleaned)
print(data.describe())

# Fill null values in multiple columns with the mean of each column
columns_to_fill = ['AM/PM', 'Src_Port']
data[columns_to_fill] = data[columns_to_fill].apply(lambda x: x.fillna(x.std()))

# Display the DataFrame with null values filled with means in specified columns
print(data.describe())


# Plot the noisy data
att = ['Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min']
plt.figure(figsize=(20, 10))
data[att].boxplot()
plt.title('noise in data')
plt.show()

df = pd.DataFrame(data)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)  # Specify the number of components you want
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the selected principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Concatenate the original DataFrame with the new principal components
result_df = pd.concat([df, pc_df], axis=1)

# Display the result
print(f'Apply PCA####################################################################')
print(result_df)
# Visualize the results

plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Protocol')
plt.ylabel('AM/PM')
plt.show()
print(f'##################/////////////############################################//////////////////////######')
# Separate features and labels
X = df.drop('AM/PM', axis=1)
y = df['AM/PM']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an undersampling strategy
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1)

# Create a pipeline with undersampling and SVM
pipeline = make_pipeline(undersampler, svm_classifier)

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f'************************************')
# Get predicted probabilities for the positive class
y_prob = pipeline.decision_function(X_test)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f'************************************+++++++++++++++++++++++++++')
data = data.dropna()
X = data.drop('Normal', axis=1)  # Adjust 'target_column' to your target variable
y = data['Normal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
y_pred = qda_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
 xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
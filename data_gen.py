import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()

# Convert the dataset to a DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Output the number of rows in the dataset
print(f"Number of rows in the dataset: {len(data)}")

# Shuffle the dataset
data = data.sample(frac=1, random_state=30).reset_index(drop=True)

# Split the dataset into train (75%), test (15%), and inference (10%)
train_size = 0.75
test_size = 0.15
inference_size = 0.10

# First split: Train and Remaining (test + inference)
train_data, remaining_data = train_test_split(data, train_size=train_size, random_state=42)

# Second split: Test and Inference from the remaining data
test_data, inference_data = train_test_split(remaining_data,
                                             test_size=inference_size / (test_size + inference_size),
                                             random_state=42)

# Separate features and target for the inference dataset
inference_features = inference_data.drop(columns=['target'])
#inference_labels = inference_data['target']

# Save the datasets to CSV files in the 'data/' directory
train_data.to_csv('data/iris_train_data.csv', index=False)
test_data.to_csv('data/iris_test_data.csv', index=False)
inference_features.to_csv('data/iris_inference_data.csv', index=False)
#inference_labels.to_csv('data/iris_inference_labels.csv', index=False)

print("Datasets have been split and saved in the 'data/' directory.")

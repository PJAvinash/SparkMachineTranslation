import os
import pandas as pd

# Function to read data from files
def read_data(x_file, y_file):
    with open(x_file, 'r', encoding='utf-8') as f:
        x_data = f.readlines()
    with open(y_file, 'r', encoding='utf-8') as f:
        y_data = f.readlines()
    return x_data, y_data

# Define file paths
data_dir = '/Users/pjavinash/Documents/Avinash/UTD_MS/4th_sem/CS6320_NLP/Project/SparkMachineTranslation/project/data/wili-2018'  # Directory containing data files
x_train_file = os.path.join(data_dir, 'x_train.txt')
y_train_file = os.path.join(data_dir, 'y_train.txt')
x_test_file = os.path.join(data_dir, 'x_test.txt')
y_test_file = os.path.join(data_dir, 'y_test.txt')

# Read training data
x_train_data, y_train_data = read_data(x_train_file, y_train_file)

# Read testing data
x_test_data, y_test_data = read_data(x_test_file, y_test_file)

# Create DataFrames for training and testing data
train_df = pd.DataFrame({'Sentence': x_train_data, 'Label': y_train_data})
test_df = pd.DataFrame({'Sentence': x_test_data, 'Label': y_test_data})

# Save DataFrames as parquet files
output_dir = data_dir  # Directory to save parquet files
os.makedirs(output_dir, exist_ok=True)
train_parquet_file = os.path.join(output_dir, 'langdetection_train.parquet')
test_parquet_file = os.path.join(output_dir, 'langdetection_test.parquet')
train_df.to_parquet(train_parquet_file, index=False)
test_df.to_parquet(test_parquet_file, index=False)

print("Train and test data saved as train.parquet and test.parquet respectively.")

import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Plot each column of a CSV file.')
parser.add_argument('filename', type=str, help='Path to the CSV file')

# Parse arguments
args = parser.parse_args()

# Read the CSV file provided by the user
df = pd.read_csv(args.filename, header=None)

# Plot each column in the DataFrame
plt.figure(figsize=(10, 6))

for column in df.columns:
    plt.plot(df.index, df[column], label=f'Column {column+1}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Each Column')
plt.legend()
plt.show()


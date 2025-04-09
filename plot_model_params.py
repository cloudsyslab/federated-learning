import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to process the file and plot the data
def plot_data(filename):
    # Load data from file into DataFrame
    df = pd.read_csv(filename, header=None)  # Assuming CSV file without headers initially

    # Set the first row as column headers
    df.columns = df.iloc[0]
    df = df[1:]  # Remove the first row from the data

    # Plot each column except the first one
    for column in df.columns[1:]:  # Exclude the first column
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column], label=column)
        plt.title(f'Plot of {column}')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Plot columns from a CSV file (excluding the first one).")
    parser.add_argument('filename', type=str, help="Path to the input CSV file")
    
    args = parser.parse_args()
    
    # Call the function to plot the data
    plot_data(args.filename)

if __name__ == "__main__":
    main()


import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def select_top_n_per_x(x_values, y_values, n):
    # Create a dictionary to store y values per x value
    data_dict = defaultdict(list)

    # Group y values per x value
    for x, y in zip(x_values, y_values):
        data_dict[x].append(y)

    # Select top n y values per x value
    selected_x = []
    selected_y = []
    for x, y_list in data_dict.items():
        selected_x.extend([x] * min(n, len(y_list)))  # Pick top n or less if y list is shorter
        selected_y.extend(sorted(y_list, reverse=True)[:n])

    return selected_x, selected_y

def merge_csv_data(csv_files):
    merged_x = []
    merged_y = []

    for csv_file in csv_files:
        # Read the CSV file with tab delimiter
        with open(csv_file, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            data = list(reader)

        # Extract the x and y values from the CSV data
        x = []
        y = []
        for row_num, row in enumerate(data):
            for value in row:
                x.append(row_num)
                y.append(float(value))

        # Select top n y values per x value
        x, y = select_top_n_per_x(x, y, 5)  # Adjust the number of selected values as desired

        # Merge the data from current file with the overall data
        merged_x.extend(x)
        merged_y.extend(y)

    return merged_x, merged_y

def plot_merged_csv_data(csv_files_set1, csv_files_set2):
    # Merge data from the first set of CSV files
    merged_x1, merged_y1 = merge_csv_data(csv_files_set1)

    # Merge data from the second set of CSV files
    merged_x2, merged_y2 = merge_csv_data(csv_files_set2)

    # Plot the merged data from the first set
    plt.scatter(merged_x1, merged_y1, s=3, c='blue', label='Ramped Half-and-Half')  # Adjust label, color, and dot size as desired

    # Plot the merged data from the second set
    plt.scatter([x + 0.5 for x in merged_x2], merged_y2, s=3, c='red', label='Baseline')  # Adjust label, color, and dot size as desired

    plt.xlabel('Generation')  # Adjust x-axis label
    plt.ylabel('Fitness')  # Adjust y-axis label
    plt.legend()
    plt.savefig('merged_scatter_plot.png', dpi=300)  # Adjust the 'dpi' parameter for higher resolution
    plt.show()

csv_files_1 = [
    "rhh/fitnesses-2023-06-14-16:23:49.csv",
    "rhh/fitnesses-2023-06-14-18:14:11.csv",
    "rhh/fitnesses-2023-06-14-20:08:12.csv",
    "rhh/fitnesses-2023-06-14-22:33:59.csv",
    "rhh/fitnesses-2023-06-15-00:28:51.csv",
]

csv_files_2 = [
    "baseline/fitnesses-2023-06-14-01_52_45_run1.csv",
    "baseline/fitnesses-2023-06-14-04_30_12_run2.csv",
    "baseline/fitnesses-2023-06-14-06_13_22_run3.csv",
    "baseline/fitnesses-2023-06-14-08_41_52_run4.csv",
    "baseline/fitnesses-2023-06-14-11_27_58_run5.csv",
]

if __name__ == '__main__':
    plot_merged_csv_data(csv_files_1, csv_files_2)

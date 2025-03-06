import matplotlib.pyplot as plt
import csv

# Plot dataset high
file_path_high = "../saved_dataset/dataset_high.csv"

z_high = []

with open(file_path_high, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        z_high.append(float(row[0]))

plt.figure(1)
plt.scatter(range(len(z_high)), z_high)
plt.xlabel('')
plt.ylabel('Z Coordinate')
plt.title('Dataset High')


# Plot dataset low
file_path_low = "../saved_dataset/dataset_low.csv"

y_low, z_low = [], []

with open(file_path_low, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        y_low.append(float(row[0]))
        z_low.append(float(row[1]))

plt.figure(2)
plt.scatter(y_low, z_low)
plt.xlabel('Y Coordinate')
plt.ylabel('Z Coordinate')
plt.title('Dataset Low')

plt.show()

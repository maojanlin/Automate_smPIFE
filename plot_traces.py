import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_pattern(pattern_csv_file):
    f = open(pattern_csv_file, 'r')
    matrix_data = []
    for line in f:
        list_data = [np.double(k) for k in line.strip().split(',')]
        matrix_data.append(list_data)
    f.close()
    return matrix_data


def plot_matrix(matrix_data, out_path):
    t = np.linspace(0, len(matrix_data[0])/10, len(matrix_data[0]), False)
    for idx, list_data in enumerate(matrix_data):
        plt.figure(figsize = (16,8))
        plt.plot(t, matrix_data[idx], linewidth=1)
        #plt.show()
        plt.savefig(out_path + "/" + str(idx+1) + '.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pattern_csv_file', help='donor pattern csv file')
    parser.add_argument('-out', '--out_path', default="./", help='output_directory_path')
    args = parser.parse_args()

    out_path = args.out_path
    pattern_csv_file = args.pattern_csv_file
    matrix_data = read_pattern(pattern_csv_file)
    
    plot_matrix(matrix_data, out_path)


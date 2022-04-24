import sys
import pandas as pd
import numpy as np
import mykmeanssp
import os

RESULT_FILE = "result.txt"
INPUT_FILE = "my_input.txt"


def get_cmd_arguments():
    args = sys.argv
    if len(args) == 5:
        return int(args[1]), 200, float(args[2]), args[3], args[4]
    elif len(args) == 6:
        return int(args[1]), int(args[2]), float(args[3]), args[4], args[5]


def read_input_file(file_name):
    vectors = pd.read_csv(file_name, header=None)
    vectors.to_csv(INPUT_FILE, index=False, header=False)
    return vectors.values.tolist()


def choose_centroids(all_vectors, k):
    np.random.seed(0)
    n = len(all_vectors)
    d_of_vectors = [0] * n
    chosen_idx = np.random.choice(n)
    centroids = [all_vectors[chosen_idx]]
    centroids_idxs = [chosen_idx]
    for centroid_idx in range(k-1):
        for vector_idx in range(n):
            d_of_vectors[vector_idx] = min(np.linalg.norm(np.subtract(c, all_vectors[vector_idx]))**2
                                           for c in centroids)
        d_sum = sum(d_of_vectors)
        chosen_idx = np.random.choice(range(n), p=[d/d_sum for d in d_of_vectors])
        centroids.append(all_vectors[chosen_idx])
        centroids_idxs.append(chosen_idx)
    pd.DataFrame(centroids).to_csv("first_centroids.txt", index=False, header=False)
    return centroids_idxs


def print_output():
    with open(RESULT_FILE, "r") as final_centroids:
        print("".join([str(line) for line in final_centroids])[:-1])


def main():
    k, goal, file_name = get_cmd_arguments()
    if goal == "spk":
        mykmeanssp.algorithm(k, "T")
        all_vectors = read_input_file(RESULT_FILE)
        centroids_idxs = choose_centroids(all_vectors)
        print(",".join([str(s) for s in centroids_idxs]))
        os.remove(RESULT_FILE)
    mykmeanssp.algorithm(k, goal)
    print_output()
    os.remove(INPUT_FILE)
    os.remove(RESULT_FILE)


if __name__ == '__main__':
    main()

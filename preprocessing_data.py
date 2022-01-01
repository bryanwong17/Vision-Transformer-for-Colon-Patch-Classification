from os import listdir
from os.path import isfile, join
import csv
import pandas as pd

# D:0
# M:1
# N:2

def main():
    PATH_TRAIN = "sample_data_colon/big_data/test"
    # PATH_TEST = "sample_data_colon/test"

    files_N = [f for f in listdir(PATH_TRAIN + "/0n") if isfile(join(PATH_TRAIN + "/0n", f))]
    files_D = [f for f in listdir(PATH_TRAIN + "/1d") if isfile(join(PATH_TRAIN + "/1d", f))]
    files_M = [f for f in listdir(PATH_TRAIN + "/2m") if isfile(join(PATH_TRAIN + "/2m", f))]

    columns = ["image_id", "label"]

    with open(PATH_TRAIN + "/test.csv", 'w', newline="") as f:
        writer = csv.writer(f)

        # write the name of column
        writer.writerow(columns)

        # write the data for N
        for i in range(len(files_N)):
            temp = []
            temp.append(files_N[i])
            temp.append(0)
            writer.writerow(temp)

        # write the data for D
        for i in range(len(files_D)):
            temp = []
            temp.append(files_D[i])
            temp.append(1)
            writer.writerow(temp)
        
        # write the data for M
        for i in range(len(files_M)):
            temp = []
            temp.append(files_M[i])
            temp.append(2)
            writer.writerow(temp)

    # to random data
    df = pd.read_csv(PATH_TRAIN + "/test.csv")
    df = df.sample(frac=1)
    df.to_csv(PATH_TRAIN + "/test.csv", index=False)


if __name__ == "__main__":
    main()




import os
from pandas import read_csv
import numpy as np


if __name__ == '__main__':
    train_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/training/n-augmentation_0_maxdegree_14_resize_256_256_flip_0/kaggle"
    valid_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/valid/n-augmentation_0_maxdegree_14_resize_256_256_flip_0/kaggle"
    test_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/testing/n-augmentation_0_maxdegree_14_resize_256_256_flip_0/kaggle"
    csv = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/kaggle/train/trainLabels.csv"

    csv_df = read_csv(csv)
    max_degree=14
    angles = [x for x in range(-max_degree, -9)]
    angles.extend([x for x in range(10, max_degree + 1)])
    angles.extend([x for x in range(-9, 10)])
    for path in [train_path, valid_path, test_path]:
        jpg_list=os.listdir(path)
        levels=[]
        for i, jpg in enumerate(jpg_list):
            jpg = jpg.replace("_flipped", "")
            jpg = jpg.replace(".jpeg", "")
            jpg = jpg.replace(".jpg", "")
            for angle in angles:
                jpg = jpg.replace("_rot_%i" % angle, "")
            try:
                row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
                level = csv_df.iloc[row_number].at['level']
                levels.append(level)
            except:
                pass

        if train_path == path:
            levels = np.bincount(np.asarray(levels, dtype=np.int32), minlength=5)
            print(f"The train dataset contains {len(jpg_list)} images. The absolute frequencies of each DR level of the"
                  f"labelled data are as follows:\nNo DR: {levels[0]}\nMild DR: {levels[1]}\nModerate DR: {levels[2]}"
                  f"\nSevere DR: {levels[3]}\nProliferative DR: {levels[4]}\n")

        elif test_path == path:
            levels = np.bincount(np.asarray(levels, dtype=np.int32), minlength=5)
            print(f"The test dataset contains {len(jpg_list)} images. The absolute frequencies of each DR level "
                  f"are as follows:\nNo DR: {levels[0]}\nMild DR: {levels[1]}\nModerate DR: {levels[2]}"
                  f"\nSevere DR: {levels[3]}\nProliferative DR: {levels[4]}\n")

        else:
            levels = np.bincount(np.asarray(levels, dtype=np.int32), minlength=5)
            print(f"The valid dataset contains {len(jpg_list)} images. The absolute frequencies of each DR level"
                  f"are as follows:\nNo DR: {levels[0]}\nMild DR: {levels[1]}\nModerate DR: {levels[2]}"
                  f"\n[Severe DR: {levels[3]}\nProliferative DR: {levels[4]}\n")


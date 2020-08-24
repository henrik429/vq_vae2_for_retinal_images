import os

if __name__ == '__main__':
    training_kaggle_path = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/kaggle/train/train"
    test_kaggle_path = "/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/kaggle/test/test"

    data_size = len(os.listdir(training_kaggle_path)) + len(os.listdir(test_kaggle_path))

    print("length training kaggle dataset: {} \nlength training kaggle dataset: {} ".format(len(os.listdir(training_kaggle_path)),
                                                                                            len(os.listdir(test_kaggle_path))))

    train_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_train_images"
    valid_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_valid_images"
    test_path = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_test_images"

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for i, jpeg in enumerate(os.listdir(test_kaggle_path)):
        if i / data_size <= 0.7:
            os.system(f"cp {test_kaggle_path}/{jpeg} {train_path}/{jpeg}")
        else:
            os.system(f"cp {test_kaggle_path}/{jpeg} {valid_path}/{jpeg}")

    length_training = len(os.listdir(train_path))

    for i, jpeg in enumerate(os.listdir(training_kaggle_path)):
        if (length_training + i) / data_size <= 0.7:
            os.system(f"cp {training_kaggle_path}/{jpeg} {train_path}/{jpeg}")
        elif (length_training + i) / data_size <= 0.85:
            os.system(f"cp {training_kaggle_path}/{jpeg} {valid_path}/{jpeg}")
        else:
            os.system(f"cp {training_kaggle_path}/{jpeg} {test_path}/{jpeg}")


    print(len(os.listdir(train_path)))
    print(len(os.listdir(valid_path)))
    print(len(os.listdir(test_path)))




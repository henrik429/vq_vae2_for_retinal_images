"""
to run this, modify the config file as you wish and run `snakemake --cores <cores>`
note that the raw annotations file need to be renamed to '../data/processed/annotations/ODIR_Annotations.csv'
"""

configfile: "./utils/workflow_config.json"
n_augmentation = config["N_AUGMENTATION"]
maxdegree = config["MAX_ROTATION_ANGLE"]
path_prefix = config['PATH_PREFIX']
networkname = config['NETWORKNAME']
port = config['PORT']
resize1 = config['RESIZE'][0]
resize2 = config['RESIZE'][1]
flip = config['FLIP']


import os

rule all:
    input:
        expand("{port}/vis.done", port = config['PORT']),
        expand("{port}/fid.done", port = config['PORT'])
    run:
        path = os.path.abspath(str(path_prefix) + "/models/" + str(networkname))
        # os.system(f"rm {port}/vis.done")
        shell("tensorboard --logdir %s --port {port}" % path)


rule visualization:
    input:
        expand("{port}/classifier.done", port = config['PORT']),
        test_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']),
        csv="/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/kaggle/train/trainLabels.csv"
    output:
        touch(expand("{port}/vis.done", port = config['PORT']))
    run:
        shell("python test_model.py -t {input.test_data} -csv {input.csv} -pp {path_prefix} -nn {networkname} -md {maxdegree}")


rule classifier:
    input:
        expand("{port}/fid.done", port = config['PORT']),
        train_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
                 n_augmentation = config['N_AUGMENTATION'],
                 maxdegree = config['MAX_ROTATION_ANGLE'],
                 resize1 = config['RESIZE'][0],
                 resize2 = config['RESIZE'][1],
                 flip = config['FLIP']),
        test_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']),
         valid_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/valid/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']),
        csv="/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/kaggle/train/trainLabels.csv"
    output:
        touch(expand("{port}/classifier.done", port = config['PORT']))
    run:
        shell("python classifier.py -i {input.train_data} -t {input.test_data} -v {input.valid_data} -csv {input.csv} -pp {path_prefix} -nn {networkname} -md {maxdegree}")


rule calculate_fid:
    input:
        expand("{port}/training.done", port = config['PORT']),
        train_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']),
        valid_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/valid/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
         n_augmentation = config['N_AUGMENTATION'],
         maxdegree = config['MAX_ROTATION_ANGLE'],
         resize1 = config['RESIZE'][0],
         resize2 = config['RESIZE'][1],
         flip = config['FLIP'])
    output:
        touch(expand("{port}/fid.done", port = config['PORT']))
    run:
        shell("python fid.py -i {input.train_data} -v {input.valid_data} -pp {path_prefix} -nn {networkname} ")


rule training:
    input:
        train_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
                 n_augmentation = config['N_AUGMENTATION'],
                 maxdegree = config['MAX_ROTATION_ANGLE'],
                 resize1 = config['RESIZE'][0],
                 resize2 = config['RESIZE'][1],
                 flip = config['FLIP']),
        valid_data=expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/valid/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP'])
    output:
        touch(expand("{port}/training.done", port = config['PORT']))
    run:
        shell("python train_model.py -i {input.train_data} -v {input.valid_data} -pp {path_prefix} -nn {networkname}" )


rule preprocess_training_images:
    input:
        "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_train_images"
    output:
        directory(expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/training/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
                 n_augmentation = config['N_AUGMENTATION'],
                 maxdegree = config['MAX_ROTATION_ANGLE'],
                 resize1 = config['RESIZE'][0],
                 resize2 = config['RESIZE'][1],
                 flip = config['FLIP']))
    run:
         shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -f {flip}")


rule preprocess_testing_images:
    input:
        "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_test_images"
    output:
       directory(expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/testing/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']))
    run:
        shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -f {flip}")


rule preprocess_valid_images:
    input:
        "/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/raw/kaggle/trainingPLUStesting/kaggle_valid_images"
    output:
       directory(expand("/data/analysis/ag-reils/ag-reils-shared-students/henrik/data/processed/valid/n-augmentation_{n_augmentation}_maxdegree_{maxdegree}_resize_{resize1}_{resize2}_flip_{flip}/kaggle/",
             n_augmentation = config['N_AUGMENTATION'],
             maxdegree = config['MAX_ROTATION_ANGLE'],
             resize1 = config['RESIZE'][0],
             resize2 = config['RESIZE'][1],
             flip = config['FLIP']))
    run:
        shell("python utils/preprocessing.py {input} {output} -na {n_augmentation} -mra {maxdegree} -r {resize1} {resize2} -f {flip}")




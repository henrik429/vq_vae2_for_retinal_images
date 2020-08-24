import os
import torch
import numpy as np
from skimage import io
from pandas import read_csv
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.interpolate import UnivariateSpline

from utils.utils import setup
from utils.models import VAE, VQ_VAE, VQ_VAE_2, Mode, classifier


if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/models.py", config="./config.json")
    train_path = FLAGS.valid
    device = FLAGS.device if torch.cuda.is_available() else "cpu"
    mode = Mode.vq_vae if FLAGS.mode == 1 else Mode.vq_vae_2
    maxdegree = FLAGS.maxdegree
    csv_df = read_csv(FLAGS.csv)
    levels = 5

    network_dir = f'{FLAGS.path_prefix}/models/{FLAGS.network_name}'
    batch_size = FLAGS.batch_size
    os.makedirs(f"{network_dir}/classifier/", exist_ok=True)

    image_list = os.listdir(train_path)

    try:
        image_list.remove(".snakemake_timestamp")
    except ValueError:
        pass

    test_path = FLAGS.test
    first_image = io.imread(train_path + image_list[0])
    image_size = first_image.shape[1]
    reduction_factor = image_size // FLAGS.size_latent_space

    if mode == Mode.vq_vae:
        num_emb=FLAGS.num_emb
        emb_dim=FLAGS.emb_dim
        size_latent_space = FLAGS.size_latent_space ** 2
        vae = VQ_VAE(
            hidden_channels=FLAGS.hidden_channels,
            num_emb=num_emb,
            emb_dim=emb_dim,
            reduction_factor=reduction_factor
        )

    elif mode == Mode.vq_vae_2:
        num_emb = {"top": FLAGS.num_emb_top, "bottom": FLAGS.num_emb_bottom}
        emb_dim = {"top": FLAGS.emb_dim_top, "bottom": FLAGS.emb_dim_bottom}
        size_latent_space = {"top": FLAGS.size_latent_space_top ** 2,
                             "bottom": FLAGS.size_latent_space_bottom ** 2}
        reduction_factor = {"top": image_size // FLAGS.size_latent_space_bottom // FLAGS.size_latent_space_top,
                            "bottom": image_size // FLAGS.size_latent_space_bottom}
        vae = VQ_VAE_2(
            hidden_channels=FLAGS.hidden_channels,
            num_emb=num_emb,
            emb_dim=emb_dim,
            reduction_factor_bottom=reduction_factor["bottom"],
            reduction_factor_top=reduction_factor["top"]
        )

    elif mode == Mode.vae:
        vae = VAE(
            hidden_channels=FLAGS.hidden_channels,
            z_dim=FLAGS.z_dim,
            image_size=image_size,
            reduction_factor=reduction_factor,
            encoding_channels=FLAGS.encodings_channels
        )

    vae.load_state_dict(torch.load(f"{network_dir}/{FLAGS.network_name}.pth"))
    vae.to(device=device)
    vae.eval()

    angles = [x for x in range(-maxdegree, -9)]
    angles.extend([x for x in range(10, maxdegree + 1)])
    angles.extend([x for x in range(-9, 10)])

    data = []
    targets = []
    print("Load train data and generate their targets...")
    for idx, jpg in tqdm(enumerate(image_list)):
        original_jpg = jpg
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        target = np.zeros(levels)
        target[level] = 1.0
        targets.append(target)
        im = io.imread(train_path + original_jpg)
        data.append(im)

    data = torch.tensor(data).permute(0, 3, 1, 2).float()
    targets = torch.tensor(targets).float()
    data_size = targets.shape[0]
    n_batches = data_size//batch_size

    if mode == Mode.vq_vae:
        predictor = classifier(size_flatten_encodings=size_latent_space *emb_dim, num_targets=levels).to(device)
    elif mode == Mode.vq_vae_2:
        predictor = classifier(size_flatten_encodings=size_latent_space["bottom"] * emb_dim["bottom"], num_targets=levels).to(device)
    else:
        predictor = classifier(size_flatten_encodings=FLAGS.z_dim, num_targets=5).to(device)

    kwargs = {"lr":5e-5}
    optimizer = torch.optim.Adam(predictor.parameters(), **kwargs)
    lossarray = []
    n_epochs = 1000
    writer = SummaryWriter(f"{network_dir}/classifier_tensorboard/")
    print("Start Training")
    for epoch in tqdm(range(n_epochs)):
        for i in range(n_batches):
            train_data = data[i * batch_size:(i + 1) * batch_size].to(device)
            if mode == Mode.vq_vae:
                encodings, _ = vae.encode(train_data)
            elif mode == Mode.vq_vae_2:
                encodings, _, _, _ = vae.encode(train_data)
            else:
                encodings, _, _ = vae.encode(train_data)

            encodings = encodings.reshape((encodings.shape[0], np.prod(encodings.shape[1:])))
            optimizer.zero_grad()
            predictions = predictor(encodings)
            loss = F.binary_cross_entropy(predictions, targets[i * batch_size:(i + 1) * batch_size].to(device))
            loss.backward()
            optimizer.step()

            if i % 100 == 0:  # append loss every 10 batches
                lossarray.append(loss.item())
                writer.add_scalar("classification loss", loss.item(), global_step=batch_size*epoch+i)

    x = np.arange(len(lossarray))
    spl = UnivariateSpline(x, lossarray)
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    plt.savefig(f"{network_dir}/classifier/loss_curve.png")

    print("Test classifier")
    image_list = os.listdir(train_path)
    try:
        image_list.remove(".snakemake_timestamp")
    except ValueError:
        pass

    image_list = image_list
    first_image = io.imread(train_path + image_list[0])
    W, H = first_image.shape[:2]
    data = []
    targets = []

    print("Load test data and generate their targets...")
    for idx, jpg in tqdm(enumerate(image_list)):
        original_jpg = jpg
        jpg = jpg.replace("_flipped", "")
        jpg = jpg.replace(".jpeg", "")
        jpg = jpg.replace(".jpg", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['image'] == jpg].index[0]
        level = csv_df.iloc[row_number].at['level']
        target = np.zeros(levels)
        target[level] = 1.0
        targets.append(target)
        im = io.imread(train_path + original_jpg)
        data.append(im)

    data = torch.tensor(data).permute(0, 3, 1, 2).float()
    test_targets = torch.tensor(targets).float()
    targets = np.asarray(targets)
    data_size = targets.shape[0]
    n_batches = data_size // batch_size

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(n_batches):
            test_data = data[i * batch_size:(i + 1) * batch_size].to(device)
            if mode == Mode.vq_vae:
                encodings, _ = vae.encode(test_data)
            elif mode == Mode.vq_vae_2:
                encodings, _, _, _ = vae.encode(test_data)
            else:
                encodings, _, _ = vae.encode(test_data)

            encodings = encodings.reshape((encodings.shape[0], np.prod(encodings.shape[1:])))
            predictions = predictor(encodings)
            _, predicted = torch.max(predictions.data, 1)
            total += batch_size
            _, batch_targets = torch.max(test_targets[i * batch_size:(i + 1) * batch_size], 1)
            batch_targets.to(device)
            for i in range(batch_size):
                if predicted[i] == batch_targets[i]:
                    correct += 1

    print('Accuracy of the network on the %i test images: %d %%' % (data_size, 100 * correct / total))
    assert predictions.shape == targets.shape
    # ROC-Curve/AUC
    outputs = predictions.cpu().detach().numpy()
    targets = targets.float().numpy()
    colors = ['navy', 'green', 'orange', 'red', 'yellow']

    tpr = dict()  # Sensitivity/False Positive Rate
    fpr = dict()  # True Positive Rate / (1-Specifity)
    auc = dict()

    # A "micro-average": quantifying score on all classes jointly
    tpr["micro"], fpr["micro"], _ = roc_curve(targets.ravel(), outputs.ravel())
    auc["micro"] = roc_auc_score(targets.ravel(), outputs.ravel(), average='micro')
    print('AUC score, micro-averaged over all classes: {0:0.2f}'.format(auc['micro']))

    plt.figure()
    plt.step(tpr['micro'], fpr['micro'], where='post')
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(
        'AUC score, micro-averaged over all classes: AP={0:0.2f}'
            .format(auc["micro"]), fontsize=13, fontweight='bold')
    plt.show()
    plt.savefig(f'{network_dir}/classifier/ROC_curve_micro_averaged.png')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(5):
        tpr[i], fpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        auc[i] = roc_auc_score(targets[:, i], outputs[:, i])

    plt.figure(figsize=(7, 9))
    lines = []
    labels = []

    l, = plt.plot(tpr["micro"], fpr["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-averaged ROC-AUC = {0:0.2f})'.format(auc["micro"]))
    diagnoses = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    for i, color in zip(range(levels), colors):
        if i in auc.keys():
            l, = plt.plot(tpr[i], fpr[i], color=color, lw=0.5)
            lines.append(l)
            labels.append('ROC for class {0} (ROC-AUC = {1:0.2f})'.format(diagnoses[i], auc[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.title('ROC curve of all features', fontsize=13, fontweight='bold')
    plt.legend(lines, labels, loc=(0, 0), prop=dict(size=8))
    plt.savefig(f'{network_dir}/classifier/ROC_curve_of_all_classes.png')
    plt.show()
    plt.close()

    # Precision-Recall Plots
    precision = dict()
    recall = dict()
    average_precision = dict()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.ravel(), outputs.ravel())
    average_precision["micro"] = average_precision_score(targets, outputs, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]), fontsize=13, fontweight='bold')
    plt.show()
    plt.savefig(f'{network_dir}/classifier/PR_curve_micro_averaged.jpg')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(levels):
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        average_precision[i] = average_precision_score(targets[:, i], outputs[:, i])

    plt.figure(figsize=(7, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (average precision = {0:0.2f}'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(levels), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=0.5)
        lines.append(l)
        labels.append('Precision-recall for class {0} (AP = {1:0.2f})'.format(diagnoses[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall curve of all features')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=9))
    plt.show()
    plt.savefig(f'{network_dir}/classifier/PR_curve_of_all_features.jpg')
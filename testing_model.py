import os
import argparse
import time
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2 as cv
from models.ccnet import ccnet
from models import MyDataset
from utils import getFileNames
import pickle

def load_pretrained_model(model_path, num_classes, comp_weight):
    model = ccnet(num_classes=num_classes, weight=comp_weight)
    try:
        state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained model loaded with unexpected keys ignored.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Attempting to filter and load matching keys only.")
        
        # Manually filter state_dict
        pretrained_dict = state_dict
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print("Pretrained model loaded with partial matching keys.")

    model.cuda()
    model.eval()
    return model


def feature_extraction(data_loader, model):
    features, ids = [], []

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0].cuda()
        target = target.cuda()

        codes = model.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()

        if batch_id == 0:
            features = codes
            ids = labels
        else:
            features = np.concatenate((features, codes), axis=0)
            ids = np.concatenate((ids, labels))

    return np.array(features), np.array(ids)

def test_model(model, test_set_file, train_set_file, path_rst, batch_size):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')
    os.makedirs(path_rst, exist_ok=True)
    os.makedirs(path_hard, exist_ok=True)

    # Load datasets
    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    # Extract features
    featDB_train, iddb_train = feature_extraction(data_loader_train, model)
    featDB_test, iddb_test = feature_extraction(data_loader_test, model)

    print('Feature extraction completed.')
    print('Train feature DB shape:', featDB_train.shape)
    print('Test feature DB shape:', featDB_test.shape)

    # Calculate matching scores
    scores = []
    labels = []
    for i in range(len(featDB_test)):
        for j in range(len(featDB_train)):
            cos_dis = np.dot(featDB_test[i], featDB_train[j])
            dis = np.arccos(np.clip(cos_dis, -1, 1)) / np.pi
            scores.append(dis)
            labels.append(1 if iddb_test[i] == iddb_train[j] else -1)

    # Save matching scores
    scores_path = os.path.join(path_rst, 'scores_VeriEER.txt')
    with open(scores_path, 'w') as f:
        for score, label in zip(scores, labels):
            f.write(f'{score} {label}\n')

    # EER and Rank-1 Accuracy Calculation
    print('Calculating EER and Rank-1 accuracy...')
    # Call external scripts (if applicable)
    os.system(f'python ./getGI.py {scores_path} scores_VeriEER')
    os.system(f'python ./getEER.py {scores_path} scores_VeriEER')

    # Rank-1 accuracy
    cnt, correct = 0, 0
    for i in range(len(featDB_test)):
        dis = [scores[cnt + j] for j in range(len(featDB_train))]
        cnt += len(featDB_train)
        idx = np.argmin(dis)
        if iddb_test[i] == iddb_train[idx]:
            correct += 1
        else:
            # Save misclassified pairs
            testname = getFileNames(test_set_file)[i]
            trainname = getFileNames(train_set_file)[idx]

            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)

            # Resize or pad images to ensure the same height
            height = min(im_test.shape[0], im_train.shape[0])  # Use the smaller height
            im_test_resized = cv.resize(im_test, (int(im_test.shape[1] * height / im_test.shape[0]), height))
            im_train_resized = cv.resize(im_train, (int(im_train.shape[1] * height / im_train.shape[0]), height))

            # Concatenate the resized images
            img = np.concatenate((im_test_resized, im_train_resized), axis=1)

            # Save the concatenated image
            save_path = os.path.join(path_hard, f'{np.min(dis):6.4f}_{testname[-13:-4]}_{trainname[-13:-4]}.png')
            cv.imwrite(save_path, img)

    rank_acc = (correct / len(featDB_test)) * 100
    print(f'Rank-1 accuracy: {rank_acc:.3f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pretrained CO3Net")
    parser.add_argument("--pretrained_model", type=str, default='./results/checkpoint/net_params.pth', help="Path to the pretrained model")
    parser.add_argument("--train_set_file", type=str, required=True, help="Path to the training dataset file")
    parser.add_argument("--test_set_file", type=str, required=True, help="Path to the testing dataset file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading")
    parser.add_argument("--num_classes", type=int, default=600, help="Number of classes in the dataset")
    parser.add_argument("--comp_weight", type=float, default=0.8, help="Competition weight parameter")
    parser.add_argument("--path_rst", type=str, default='./results/rst_test', help="Path to save the results")

    args = parser.parse_args()

    model = load_pretrained_model(args.pretrained_model, args.num_classes, args.comp_weight)
    test_model(model, args.test_set_file, args.train_set_file, args.path_rst, args.batch_size)

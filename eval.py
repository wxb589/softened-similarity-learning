import os

import torch
import torchvision
import numpy as np
from model import SSLResnet
from tqdm import tqdm
from configparser import ConfigParser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Market1501Dataset
from utils import get_id
from utils import compute_map


def main():
    curpath = os.path.dirname(os.path.realpath(__file__))
    cfgpath = os.path.join(curpath, "config.ini")

    config = ConfigParser()
    config.read(cfgpath, encoding="utf-8")

    transform = transforms.Compose([
        transforms.Resize([768, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print('Loading model...')
    net = SSLResnet(torchvision.models.resnet50(pretrained=False))
    net.load_state_dict(torch.load(config['Test']['model_path']))
    device = 'cuda'
    net.eval()
    net.to(device)
    print('Done loading model.')

    test_loader, query_loader = load_data(config['Test Data'], transform, load_img=True)

    print('Getting label, cam...')
    test_label, test_cam = get_id(config['Test Data']['path'], 'test')
    query_label, query_cam = get_id(config['Test Data']['path'], 'query')
    print('Done getting label, cam.')

    print('Extracting features...')
    test_features = extract_feature(net, test_loader)
    query_features = extract_feature(net, query_loader)
    print('Done extracting features.')

    # query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    # test_cam = np.array(test_cam)
    test_label = np.array(test_label)

    print('Start evaluate...')
    scores = np.zeros(10)
    prab = tqdm(total=len(query_label))
    for i in range(len(query_label)):
        score = evaluate(query_features[i], query_label[i], test_features, test_label)
        scores = scores + score
        prab.update(1)
    prab.close()

    CMC = scores / len(query_label)
    print('Done evaluate.')
    print('Model accuracy：')
    print('Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f}'.format(
        CMC[0].item(), CMC[4].item(), CMC[9].item()))
    return CMC


def extract_feature(model, dataloader):
    feats = torch.FloatTensor()
    ex_f = tqdm(total=len(dataloader))
    with torch.no_grad():
        for batch_index, (labels, images, cams) in enumerate(dataloader):
            images = images.to('cuda')
            _, feat = model(images)
            feat = feat.to('cpu')
            feats = torch.cat((feats, feat), 0)
            ex_f.update(1)
        ex_f.close()
    return feats


def load_data(params, transform=None, load_img=True):
    test_set = Market1501Dataset(params['path'], data='test', load_img=load_img, transform=transform)
    query_set = Market1501Dataset(params['path'], data='query', load_img=load_img, transform=transform)
    if load_img:
        test_loader = DataLoader(test_set, 32, shuffle=False)
        query_loader = DataLoader(query_set, 32, shuffle=False)
    else:
        test_loader = DataLoader(test_set, len(test_set), shuffle=False)
        query_loader = DataLoader(query_set, len(query_set), shuffle=False)
    return test_loader, query_loader


def evaluate(query_features, query_labels, test_features, test_labels):
    goal = np.zeros(10)
    query_features = query_features.view(-1, 1)
    results = torch.cdist(query_features, test_features)
    ids = results.argmin(dim=1).to('cpu')
    test_labels = test_labels[ids]

    for i in range(10):
        if query_labels == test_labels[i]:
            goal[i] = 1

    return goal


# def evaluate(qf, ql, qc, tf, tl, tc):
#     query = qf.view(-1, 1)  # 输出维度（自动判断，1）  （2048,1）
#     score = torch.mm(tf, query)  # 矩阵gf与矩阵query相乘  （batch_size,1）
#     score = score.squeeze(1).cpu()  # 第二维降低一维并转换为CPU类型的张量
#     score = score.numpy()
#
#     index = np.argsort(score)  # 返回score数组值从小到大的索引值
#     index = index[::-1]  # 将index反转，例：a='python',a[::-1]='nohtyp'  score从大到小引索
#
#     query_index = np.argwhere(tl == ql)  # 返回gl和ql相等的元组的引索 是否为一个人
#     camera_index = np.argwhere(tc == qc)  # 返回gc和qc相等的元组的引索 是否是一个摄像头
#
#     good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
#     junk_index1 = np.argwhere(tl == -1)  # 返回gl中为-1的元组的引索
#     junk_index2 = np.intersect1d(query_index, camera_index)  # 找到两个数组中的交集，并返回两个输入数组的排序的唯一值
#     junk_index = np.append(junk_index2, junk_index1)  # 将junk_index1附加到junk_index2后
#
#     ap_tmp, CMC_tmp = compute_map(index, good_index, junk_index)
#     return ap_tmp, CMC_tmp


if __name__ == '__main__':
    main()

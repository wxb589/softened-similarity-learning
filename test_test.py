import torch
import torchvision
import torchvision.transforms as transforms
import os

from torch.utils.data import DataLoader
from dataset import Market1501Dataset
from model import SSLResnet
from tqdm import tqdm
from configparser import ConfigParser
import numpy as np
from utils import single_list


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

    net = SSLResnet(torchvision.models.resnet50(pretrained=False))
    net.load_state_dict(torch.load(config['Test']['model_path']))

    test_loader, query_loader = load_data(config['Test Data'], transform, load_img=True)
    test_filepath = os.path.join(config['Test']['output_path'], 'test.pth')
    query_filepath = os.path.join(config['Test']['output_path'], 'query.pth')

    # create_embeddings(net, test_filepath, test_loader, 'test')
    # create_embeddings(net, query_filepath, query_loader, 'query')

    print('Calculating accuracy...')
    test_loader, query_loader = load_data(config['Test Data'], transform, load_img=False)
    # evaluate(query_loader, query_filepath, test_loader, test_filepath)

    test_features = torch.load(test_filepath)
    query_features = torch.load(query_filepath)
    test_labels = next(iter(test_loader))[0]
    query_labels = next(iter(query_loader))[0]
    test_cams = next(iter(test_loader))[1]
    query_cams = next(iter(query_loader))[1]

    CMC = torch.IntTensor(len(test_labels)).zero_()
    ap = 0.0

    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate(query_features[i], query_labels[i], query_cams[i], test_features, test_labels,
                                   test_cams)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_labels)))


def load_data(params, transform=None, load_img=True):
    test_set = Market1501Dataset(params['path'], data='test', load_img=load_img, transform=transform)
    query_set = Market1501Dataset(params['path'], data='query', load_img=load_img, transform=transform)
    if load_img:
        test_loader = DataLoader(test_set, 8, shuffle=False)
        query_loader = DataLoader(query_set, 8, shuffle=False)
    else:
        test_loader = DataLoader(test_set, len(test_set), shuffle=False)
        query_loader = DataLoader(query_set, len(query_set), shuffle=False)
    return test_loader, query_loader


def create_embeddings(net, output, dataloader, data):
    device = 'cuda'
    net.eval()
    net.to(device)
    if data == 'test':
        print('Creating test embeddings...')
    elif data == 'query':
        print('Creating query embeddings...')
    with torch.no_grad():
        batch_output = []

        prab = tqdm(total=len(dataloader))
        for batch_index, (labels, images, cams) in enumerate(dataloader):
            images = images.to(device)
            _, embedding = net(images)
            embedding = embedding.to('cpu')
            batch_output.append(embedding)
            prab.update(1)
        prab.close()

        vectors = torch.cat(batch_output)
        torch.save(vectors, output)


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


# def evaluate(test_loader, query_loader, test_filepath, query_filepath):
#     """
#     CMC mAP计算v3.2
#     """
#     test_vectors = torch.load(test_filepath)
#     query_vectors = torch.load(query_filepath)
#     query_labels = next(iter(query_loader))[0]
#     test_labels = next(iter(test_loader))[0]
#
#     results = torch.cdist(query_vectors, test_vectors)  # 得到查询图像与测试图像之间的欧氏距离(query * test)
#     ids = np.argsort(results, axis=1)  # 按照从小到大排序
#     test_labels = torch.LongTensor(test_labels)
#     test_labels = test_labels[ids]  # 得分从大到小的测试集label排序
#
#     score, scores = np.zeros(10), np.zeros(10)
#     # ap, num_true_precision, num_current_query = 0, 0, 0
#     precision, recall, AP = [], [], []  # 准确率召回率初始化
#
#     prab = tqdm(total=len(query_labels))
#     for query_label_index in range(len(query_labels)):  # 依次检索查询图像
#         query_label = query_labels[query_label_index]  # 当前检索的query_label
#         test_label = test_labels[query_label_index]  # 当前检索的query_label在测试集上按照得分排序的test_label
#
#         for index in range(10):  # 计算CMC得分和预测正确的次数
#             if query_label == test_label[index]:
#                 score[index:] = 1
#
#         scores = scores + score  # CMC总得分
#         score, num_true_precision, num_current_query, ap = np.zeros(10), 0, 0, 0  # 参数清零
#         prab.update(1)
#     prab.close()
#     CMC = scores / len(query_labels)  # CMC得分
#     print(
#         'Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f}'.format(CMC[0].item(), CMC[4].item(), CMC[9].item(), ))
#     print('Done calculating accuracy.')
#     return CMC


if __name__ == '__main__':
    main()

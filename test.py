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
    evaluate(test_loader, query_loader, test_filepath, query_filepath)


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


# def evaluate(test_loader, query_loader, test_filepath, query_filepath):
#     """
#     CMC计算v1.0
#     """
#     test_vectors = torch.load(test_filepath)
#     query_vectors = torch.load(query_filepath)
#     results = torch.cdist(query_vectors, test_vectors)
#     ids = results.argmin(dim=1).to('cpu')
#     all_labels = next(iter(query_loader))[0]
#     test_labels = next(iter(test_loader))[0]
#     test_labels = torch.LongTensor(test_labels)
#     test_labels = test_labels[ids]
#     accuracy = (all_labels == test_labels).sum().item() / len(all_labels)
#     print("Top-1 Accuracy: %.2f" % accuracy)


def evaluate(test_loader, query_loader, test_filepath, query_filepath):
    """
    CMC mAP计算v3.2
    """
    test_vectors = torch.load(test_filepath)
    query_vectors = torch.load(query_filepath)
    query_labels = next(iter(query_loader))[0]
    test_labels = next(iter(test_loader))[0]

    results = torch.cdist(query_vectors, test_vectors)  # 得到查询图像与测试图像之间的欧氏距离(query * test)
    ids = np.argsort(results, axis=1)  # 按照从小到大排序
    test_labels = torch.LongTensor(test_labels)
    test_labels = test_labels[ids]  # 得分从大到小的测试集label排序

    score, scores = np.zeros(10), np.zeros(10)
    # ap, num_true_precision, num_current_query = 0, 0, 0
    precision, recall, AP = [], [], []  # 准确率召回率初始化

    prab = tqdm(total=len(query_labels))
    for query_label_index in range(len(query_labels)):  # 依次检索查询图像
        query_label = query_labels[query_label_index]  # 当前检索的query_label
        test_label = test_labels[query_label_index]  # 当前检索的query_label在测试集上按照得分排序的test_label

        # for i in range(len(test_label)):  # 当前检索的query在测试集中的数目(速度太慢，爪巴)
        #     if query_label == test_label[i]:
        #         num_current_query += 1

        # num_current_query = single_list(test_label, query_label)  # 当前检索的query在测试集中的数目

        for index in range(10):  # 计算CMC得分和预测正确的次数
            if query_label == test_label[index]:
                score[index:] = 1
                # num_true_precision = num_true_precision + 1
                # precision.append(num_true_precision / (index + 1))  # 当前检索的query的准确率
                # recall.append(num_true_precision / num_current_query)  # 当前检索的query的召回率

                # if index == 0:
                #     ap = (1 / num_current_query) * (precision[0] + precision[0]) / 2
                # else:
                #     ap = ap + (1 / num_current_query) * (
                #                 (precision[num_true_precision - 1] + precision[num_true_precision - 2]) / 2)  # 计算AP

        scores = scores + score  # CMC总得分
        # AP.append(ap)
        # score, num_true_precision, num_current_query, ap = np.zeros(10), 0, 0, 0  # 参数清零
        precision, recall = [], []
        prab.update(1)
    prab.close()
    CMC = scores / len(query_labels)  # CMC得分
    # mAP = np.mean(AP)
    print(
        'Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f}'.format(CMC[0].item(), CMC[4].item(), CMC[9].item(), ))
    print('Done calculating accuracy.')
    return CMC


# def evaluate(test_loader, query_loader, test_filepath, query_filepath):
#     """
#     CMC计算v2.2
#     """
#     scores = np.zeros(10)
#     score = np.zeros(10)
#
#     test_vectors = torch.load(test_filepath)
#     query_vectors = torch.load(query_filepath)
#
#     query_labels = next(iter(query_loader))[0]
#     test_labels = next(iter(test_loader))[0]
#
#     prab = tqdm(total=len(query_labels))
#     for i in range(len(query_labels)):
#         qv = query_vectors[i]
#         qv = qv.view(-1, 256)
#         results = torch.cdist(qv, test_vectors)
#         results = results.view(-1)
#         ids = np.argsort(results)
#         tl = test_labels
#         tl = tl[ids]
#         for j in range(10):
#             if query_labels[i] == tl[j]:
#                 score[j:] = 1
#         scores = scores + score
#         score = np.zeros(10)
#         prab.update(1)
#     prab.close()
#
#     CMC = scores / len(query_labels)
#     print('Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f}'.format(CMC[0].item(), CMC[4].item(), CMC[9].item()))
#     return CMC


if __name__ == '__main__':
    main()

import os
import numpy as np
import torch


def softened_similarity_loss(outputs, targets, memory_tb, part_num=8, t=0.1, ld=0.6):
    prob = -ld * memory_tb.probability(outputs, targets, t)
    reliables = torch.LongTensor(memory_tb.reliables[targets]).T.tolist()
    reliables_prob = torch.stack(list(map(lambda r: memory_tb.probability(outputs, r, t), reliables))).T
    reliables_prob = -((1 - ld) / part_num) * torch.log(reliables_prob).sum(dim=1)
    return (prob + reliables_prob).sum()


def load_vectors():
    embeddings = []
    parts = []
    for f in os.listdir('tmp'):
        if f.startswith('embedding'):
            embeddings.append(torch.load(os.path.join('tmp', f)))
        if f.startswith('parts'):
            parts.append(torch.load(os.path.join('tmp', f)))
    parts = torch.cat(parts)
    parts = parts.chunk(8, 1)
    parts = list(map(lambda part: part.to('cpu').squeeze(1), parts))
    return torch.cat(embeddings).to('cpu'), parts


def euclidean_dist(x, y):
    m = x.size(0)
    n = y.size(0)
    d = x.size(1)

    if d != y.size(1):
        raise Exception('Invalid input shape.')

    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.sqrt()
    return dist


# def get_id(datalader):
#     label = []
#     cam = []
#     for index, (labels, images, cams) in enumerate(datalader):
#         label.append(labels)
#         cam.append(cams)
#     return label, cam

def get_id(path, data):
    """
    获取相机ID和图像的类别标签
    """
    cams = []
    labels = []

    if data == 'test':
        data_path = os.path.join(path, 'bounding_box_test')
    elif data == 'query':
        data_path = os.path.join(path, 'query')
    else:
        raise Exception('Invalid dataset type')

    images = os.listdir(data_path)

    for img in images:
        if not img.endswith('jpg'):
            continue

        # img_path = os.path.join(data_path, img)
        label, cam_seq, frame, bbox = img.split('_')
        cam = int(cam_seq[1])
        label = int(label)

        labels.append(label)
        cams.append(cam)

    return labels, cams


def single_list(arr, target):
    arr = arr.numpy().tolist()
    target = target.numpy().tolist()
    return arr.count(target)


import torch
import shutil
import numpy as np
import math as m
def compute_metric(predict, target,writer, step):
    pred_label = torch.argmax(predict, dim=1)
    tp = torch.count_nonzero(pred_label * target)
    fp = torch.count_nonzero(pred_label * (target - 1))
    tn = torch.count_nonzero((pred_label - 1) * (target - 1))
    fn = torch.count_nonzero((pred_label - 1) * (target))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    writer.add_scalar("bc/precision", precision,step)
    writer.add_scalar("bc/recall", recall,step)
    writer.add_scalar("bc/accuracy", accuracy,step)
    writer.add_scalar("bc/specificity", specificity,step)
    writer.add_scalar("bc/f1_score", f1,step)
    return tp.item(), fp.item(), tn.item(), fn.item(), precision.item(), recall.item(), accuracy.item(), specificity.item(), f1.item()


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])


class Rotation(object):  # randomly rotate a point cloud

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, points):
        # print('before',points.shape[0])
        degree = np.random.randint(0, 45)+1
        theta=m.pi/(180/degree)
        rotmtx = [Rx, Ry, Rz]
        R = rotmtx[np.random.randint(0, 2)](theta)

        points = points * R
        points = points - np.min(points, axis=0)
        points = np.round(points)
        # print('larger than block: ',np.count_nonzero(points>=self.block_size))
        points = np.delete(points, np.where(np.max(points, axis=1) >= self.block_size)[0], 0)
        points = np.delete(points, np.where(np.min(points, axis=1) < 0)[0], 0)
        # print('after',points.shape[0])
        del theta,rotmtx,R
        return points


class Random_sampling(object):  # randomly rotate a point cloud

    #def __init__(self):
          # np.random.random()/2#percent of point to be remove

    def __call__(self, points):
        # print('before',points.shape[0])
        rates = [0, 0.125, 0.175, 0.25, 0.3]
        rate = np.random.choice(rates, p=[0.5, 0.2, 0.1, 0.1,
                                               0.1])
        idx = np.random.randint(points.shape[0], size=int(points.shape[0] * (1 - rate)))
        points = points[idx, :]
        # print('after',points.shape[0])
        del idx,rates,rate
        return points

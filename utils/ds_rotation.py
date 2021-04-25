import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
import argparse
import shutil
from os import makedirs
from glob import glob
from tqdm import tqdm
import random
from os.path import join, basename, split, splitext
import numpy as np
import math as m
from multiprocessing import Pool
from pyntcloud import PyntCloud
import functools
import pandas as pd
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


def process(path,args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))

    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)
    pc = PyntCloud.from_file(ori_path)
    pc.points = pc.points.astype('float64', copy=False)
    coords = ['x', 'y', 'z']
    R=[Rx,Ry,Rz]
    points = pc.points[coords]
    ori_points=points.values
    #points=points.to_numpy()
    theta = m.pi / 4
    #i=2
    # ori_points=ori_points[:5,:]
    # print(ori_points)
    for i in range(3):
        target_path1 =target_path +('_45'+ coords[i] + args.target_extension)
        R_coef=R[i](theta)
        points=ori_points*R_coef
        points=points-np.min(points,axis=0)
        #print(points.shape)
        points=np.round(points)
        # print(np.max(points,axis=1))
        # print(np.max(points, axis=1) >= args.box)
        # print('point will be removed',np.where(np.max(points,axis=1)>=args.box)[0],'done')
        points = np.delete(points,np.where(np.max(points,axis=1)>=args.box)[0],0)
        points = np.delete(points, np.where(np.min(points, axis=1) < 0)[0], 0)
        # points=np.asarray(points)
        #print(points.shape)
        points=pd.DataFrame(data=points,columns=coords)
        if(len(points)>0):
            cloud=PyntCloud(points)
            cloud.to_file(target_path1)
def datasetRotation(args):
    assert os.path.exists(args.source), f'{args.source} does not exist'
    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source):] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    with Pool() as p:
        process_f = functools.partial(process, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len))
    logger.info(f'{files_len} models written to {args.dest}')



def singlePCRotation(args):
    pc = PyntCloud.from_file(args.source)
    print('original pc: ', pc)
    pc.points = pc.points.astype('float64', copy=False)
    coords = ['x', 'y', 'z']
    R = [Rx, Ry, Rz]

    points = pc.points[coords]
    ori_points = points.values
    theta = m.pi / (4)
    R_coef = R[2](theta)
    # theta = m.pi
    # R_coef = R_coef*R[0](theta)
    # theta = m.pi/120
    # R_coef = R_coef * R[2](-theta)
    points = ori_points * R_coef
    points = points - np.min(points, axis=0)
    points = np.round(points)
    points = np.delete(points, np.where(np.max(points, axis=1) >= args.box)[0], 0)
    points = np.delete(points, np.where(np.min(points, axis=1) < 0)[0], 0)
    points = pd.DataFrame(data=points, columns=coords)
    if (len(points) > 0):
        cloud = PyntCloud(points)
        cloud.to_file(args.dest)
        print('pc written',cloud)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--level', type=int, help='Number of level to downsample', default=1)
    parser.add_argument('--box', type=int, help='bounding box of pc', default=64)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')
    parser.add_argument("-portion", '--portion_points', type=float,
                        default=1,
                        help='percent of point to be removed')

    args = parser.parse_args()
    #datasetRotation(args)
    singlePCRotation(args)



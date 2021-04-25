import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import pandas as pd
import argparse
import functools
from tqdm import tqdm
from multiprocessing import Pool
from utils import octree_partition
#from . import octree_partition
import numpy as np

def arr_to_pc(arr, cols, types):
    d = {}
    for i in range(arr.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = arr[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    pc = PyntCloud(df)
    return pc


def process(path, args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))
    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)

    pc = PyntCloud.from_file(ori_path)
    bbox_min = [0, 0, 0]
    bbox_max = [args.vg_size, args.vg_size, args.vg_size]
    coords = ['x', 'y', 'z']
    points = pc.points[coords]
    print(np.max(points,axis=0))
    blocks, _ = octree_partition.partition_octree(points, bbox_min, bbox_max, args.level)
    #print(blocks[1][:10])
    cur_pc =  arr_to_pc(blocks[1], pc.points.columns[0:3], pc.points.dtypes)
    # # cloud = PyntCloud(cur_pc)
    # # points = points.astype('float32', copy=False)
    cur_pc.to_file(target_path+'1.ply')

    block=cur_pc.points
    block=(block-0.01)/2
    block = np.round(block)
    block = block.drop_duplicates()
    block = block.dropna()
    block = np.abs(block)
    #print(block[0:5],block.shape)


    downsampled_size = int(args.vg_size/2)
    pc.points = pc.points.astype('float64', copy=False)
    coords = ['x', 'y', 'z']
    points2 = pc.points[coords]
    #points2 = points2 - np.min(points2,axis=0)
    #points2 = points2 / np.max(points2,axis=0)
    points2=(points2-0.01)/2
    #points2 = points2 * (downsampled_size - 1)
    points2 = np.round(points2)
    points2 = points2.drop_duplicates()
    points2 = points2.dropna()
    downsampled_points = points2.astype('float32', copy=False)
    blocks2, _ = octree_partition.partition_octree(downsampled_points, [0, 0, 0],[downsampled_size, downsampled_size, downsampled_size], args.level)
    #print(len(blocks),len(blocks2),np.min(blocks[10],axis=0), np.max(blocks[10],axis=0), np.min(blocks2[10],axis=0),np.max(blocks2[10],axis=0))
    cur_pc2 = arr_to_pc(blocks2[1], pc.points.columns[0:3], pc.points.dtypes)
    #cloud2 = PyntCloud(cur_pc2)
    cur_pc2.to_file(target_path+'2.ply')
    # for i, block in enumerate(blocks):
    #     final_target_path = target_path + f'_{i:03d}{args.target_extension}'
    #     logger.debug(f"Writing PC {ori_path} to {final_target_path}")
    #     cur_pc = arr_to_pc(block, pc.points.columns, pc.points.dtypes)
    #     cur_pc.to_file(final_target_path)
    #print(cur_pc2.points[0:5],cur_pc2.points.shape)

def process2(path, args):
    ori_path = join(args.source, path)
    target_path, _ = splitext(join(args.dest, path))
    target_folder, _ = split(target_path)
    makedirs(target_folder, exist_ok=True)

    pc = PyntCloud.from_file(ori_path)
    coords = ['x', 'y', 'z']
    points = pc.points[coords]
    length=int(args.vg_size/64)
    cnt=0
    for d in range(length):
        for h in range(length):
            for w in range(length):
                center = points[
                    (points['x'] >= np.max([0, d * 64 ])) & (points['x'] < np.min([args.vg_size, d * 64 + 64])) & (
                            points['y'] >= np.max([0, h * 64 ])) & (
                            points['y'] < np.min([args.vg_size, h * 64 + 64])) & (
                            points['z'] >= np.max([0, w * 64 ])) & (
                            points['z'] < np.min([args.vg_size, w * 64 + 64]))]
                if(center.shape[0]>0):
                    cnt+=1
                    sample = points[
                        (points['x'] >= np.max([0, d * 64 - 8])) & (
                                    points['x'] < np.min([args.vg_size, d * 64 + 72])) & (
                                points['y'] >= np.max([0, h * 64 - 8])) & (
                                points['y'] < np.min([args.vg_size, h * 64 + 72])) & (
                                points['z'] >= np.max([0, w * 64 - 8])) & (
                                points['z'] < np.min([args.vg_size, w * 64 + 72]))]

                    sample=sample-np.min(sample,axis=0)
                    sample=(sample-0.01)/2
                    sample = np.round(sample)
                    sample = np.abs(sample)
                    sample = sample.drop_duplicates()
                    sample = sample.dropna()


                    center=center-np.min(center,axis=0)

                    #print(np.max(center, axis=0), np.max(sample, axis=0))

                    center_target_path = target_path + '_bl64' + f'_{cnt:04d}{args.target_extension}'
                    bbox_target_path = target_path + '_bl40' + f'_{cnt:04d}{args.target_extension}'
                    pc1=PyntCloud(sample)
                    pc2=PyntCloud(center)
                    pc1.to_file(bbox_target_path)
                    pc2.to_file(center_target_path)
    print(cnt,' blocks are written')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds pc superres block.py',
        description='Convert pc to block 64 and its downsample version + border area (border size =16)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--vg_size', type=int, help='Voxel Grid resolution for x, y, z dimensions', default=64)
    parser.add_argument('--level', type=int, help='Octree decomposition level.', default=3)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')

    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    assert args.vg_size > 0, f'vg_size must be positive'

    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source): ] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    with Pool() as p:
        process_f = functools.partial(process2, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len))
        # Without parallelism
        # list(tqdm((process_f(f) for f in files), total=files_len))

    logger.info(f'{files_len} models written to {args.dest}')
import os
from os.path import join, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import numpy as np
import argparse

def pc_2_voxelgrid(source, dest,level, target_extension='.ply', source_extension='.ply'):
    paths = glob(join(source, '**', f'*{source_extension}'), recursive=True)
    print(paths)
    files = [x[len(source):] for x in paths]
    for path in files:
        ori_path = join(source, path)
        target_path, _ = splitext(join(dest, path))
        target_path += ('_ds_by'+str(level)+target_extension)
        target_folder, _ = split(target_path)
        makedirs(target_folder, exist_ok=True)

        print(f"Writing PC {ori_path} to {target_path}")
        pc = PyntCloud.from_file(ori_path)
        pc.points = pc.points.astype('float64', copy=False)
        coords = ['x', 'y', 'z']
        points = pc.points[coords]
        print(type(points), 'len point: ', len(points))
        print('Max coordinate value: ',np.max(points))
        # points = points - np.min(points,axis=0)
        # points = points / np.max(points,axis=0)
        # points = points * (vg_size - 1)
        for i in range(level):
            points = (points - 0.01) / 2
            points = np.abs(points)
            points = np.round(points)
            points = points.drop_duplicates()

        print(type(points), 'Len point after unique: ', len(points))
        points=points.dropna()
        cloud=PyntCloud(points)
        #points=points.astype('float32', copy=False)
        cloud.to_file(target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    )

    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--level', type=int, help='Number of level to downsample', default=1)
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    parser.add_argument('--target_extension', help='Point cloud extension', default='.ply')

    args = parser.parse_args()
    pc_2_voxelgrid(args.source,args.dest,args.level)
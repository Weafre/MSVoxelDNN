import numpy as np
from utils.octree_partition import partition_octree
import time
from glob import glob
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
import pandas as pd
import torch
#import open3d as o3d
#VOXEL-OCTREE
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap

def get_bin_stream_blocks(path_to_ply, pc_level, departition_level):
    # co 10 level --> binstr of 10 level, blocks size =1
    level = int(departition_level)
    pc = PyntCloud.from_file(path_to_ply)
    points = pc.points.values
    no_oc_voxels = len(points)
    box = int(2 ** pc_level)
    blocks2, binstr2 = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return no_oc_voxels, blocks2, binstr2

def voxel_block_2_octree(box,oct_seq):
    box_size=box.shape[0]
    child_bbox=int(box_size/2)
    if(box_size>2):
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box=box[d * child_bbox:(d + 1) * child_bbox, h * child_bbox:(h + 1) * child_bbox, w * child_bbox:(w + 1) * child_bbox]
                    if(np.sum(child_box)!=0):
                        oct_seq.append(1)
                        voxel_block_2_octree(child_box, oct_seq)
                    else:
                        oct_seq.append(0)

    else:
        curr_octant=[int(x) for x in box.flatten()]
        oct_seq+=curr_octant
    return oct_seq

#FOR VOXEL
def input_fn_super_res(points, batch_size, dense_tensor_shape32, data_format, repeat=True, shuffle=True, prefetch_size=1):
    # Create input data pipeline.
    def gen():
        iterator=iter(points)
        done=False
        while not done:
            try:
                p = next(iterator)
            except StopIteration:
                done=True
            else:
                ds = np.abs(np.round((p - 0.01) / 2))
                ds = np.unique(ds,axis=0)
                yield (ds, p)
    p_max = np.array([64, 64, 64])
    dense_tensor_shape64 = np.concatenate([p_max, [1]]).astype('int64')
    dense_tensor_shape=[dense_tensor_shape32,dense_tensor_shape64]
    dataset = tf.data.Dataset.from_generator(generator=gen, output_types=(tf.int64,tf.int64),output_shapes= (tf.TensorShape([None, 3]),tf.TensorShape([None, 3])))
    if shuffle:
        dataset = dataset.shuffle(len(points))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(lambda x,y: pc_to_tf(x,y
                                               , dense_tensor_shape, data_format))
    dataset = dataset.map(lambda x,y: process_x(x,y, dense_tensor_shape))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset
# Main launcher
def input_fn_voxel_dnn(points, batch_size, dense_tensor_shape, data_format, repeat=True, shuffle=True, prefetch_size=1):
    print('point shape: ', points.shape)
    # Create input data pipeline.

    dataset = tf.data.Dataset.from_generator(lambda: iter(points), tf.int64, tf.TensorShape([None, 3]))
    if shuffle:
        dataset = dataset.shuffle(len(points))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(lambda x: pc_to_tf_voxel_dnn(x, dense_tensor_shape, data_format))
    dataset = dataset.map(lambda x: process_x_voxel_dnn(x, dense_tensor_shape))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset
def df_to_pc(df):
    points = df[['x', 'y', 'z']].values
    return points


def pa_to_df(points):
    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    types = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    assert 3 <= points.shape[1] <= 6
    for i in range(points.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = points[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def pc_to_tf(x,y, dense_tensor_shape, data_format):
    assert data_format in ['channels_last', 'channels_first']
    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        x = tf.pad(x, [[0, 0], [0, 1]])
    else:
        x = tf.pad(x, [[0, 0], [1, 0]])
    st0 = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape[0])

    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        y = tf.pad(y, [[0, 0], [0, 1]])
    else:
        y = tf.pad(y, [[0, 0], [1, 0]])
    st1 = tf.sparse.SparseTensor(y, tf.ones_like(y[:, 0]), dense_tensor_shape[1])
    return (st0,st1)

def process_x(x,y, dense_tensor_shape):

    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape[0])
    x = tf.cast(x, tf.float32)

    y = tf.sparse.to_dense(y, default_value=0, validate_indices=False)
    y.set_shape(dense_tensor_shape[1])
    y = tf.cast(y, tf.float32)
    return (x,y)

def pc_to_tf_voxel_dnn(points, dense_tensor_shape, data_format):
    x = points
    assert data_format in ['channels_last', 'channels_first']
    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        x = tf.pad(x, [[0, 0], [0, 1]])
    else:
        x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape)
    # print('st in pc to tf: ',st)
    return st

def process_x_voxel_dnn(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    # print('x in process x: ',x)
    return x


def get_shape_data(resolution, data_format):
    assert data_format in ['channels_last', 'channels_first']
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    if data_format == 'channels_last':
        dense_tensor_shape = np.concatenate([p_max, [1]]).astype('int64')
    else:
        dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    return p_min, p_max, dense_tensor_shape


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_pc(path):
    try:
        pc = PyntCloud.from_file(path)
        points=pc.points
        ret = df_to_pc(points)
        return ret
    except:
        return


def load_points(files, batch_size=32):
    files_len = len(files)
    with multiprocessing.Pool() as p:
        # logger.info('Loading PCs into memory (parallel reading)')
        points = np.array(list(tqdm(p.imap(load_pc, files, batch_size), total=files_len)))
    return points


# blocks to occupancy maps, only for running the first time to explore pc characteristic
def pc_2_block_oc3_test(blocks, bbox_max):
    no_blocks = len(blocks)
    blocks_oc = np.zeros((no_blocks, bbox_max, bbox_max, bbox_max, 1), dtype=np.float32)
    coor_min_max=np.zeros((no_blocks,6),dtype=np.uint32)
    lower_level_ocv=[]
    for i, block in enumerate(blocks):
        block = block[:, 0:3]
        # getting infor of block
        coor_min_max[i,:3] = np.min(block,axis=0)
        coor_min_max[i, 3:] = np.max(block,axis=0)
        bl_points = (block - 0.01) / 2
        bl_points = np.abs(bl_points)
        bl_points = np.round(bl_points)
        bl_points = np.unique(bl_points,axis=0)
        lower_level_ocv.append(len(bl_points))

        block = block.astype(np.uint32)
        blocks_oc[i, block[:, 0], block[:, 1], block[:, 2], 0] = 1.0
    return blocks_oc,coor_min_max,lower_level_ocv
def occupancy_map_explore_test(ply_path, pc_level, departition_level):
    no_oc_voxels, blocks, binstr = get_bin_stream_blocks(ply_path, pc_level, departition_level)
    bbox_max=int(2**(pc_level-departition_level))
    boxes,coor_min_max,lower_level_ocv = pc_2_block_oc3_test(blocks, bbox_max)
    return boxes, binstr, no_oc_voxels,coor_min_max,lower_level_ocv



#official function of pc2block oc3 and occupalncy map explore
def pc_2_block_oc3(blocks, bbox_max):
    no_blocks = len(blocks)
    blocks_oc = np.zeros((no_blocks, bbox_max, bbox_max, bbox_max, 1), dtype=np.float32)
    for i, block in enumerate(blocks):
        block = block[:, 0:3]
        block = block.astype(np.uint32)
        blocks_oc[i, block[:, 0], block[:, 1], block[:, 2], 0] = 1.0
    return blocks_oc

def occupancy_map_explore(ply_path, pc_level, departition_level):
    no_oc_voxels, blocks, binstr = get_bin_stream_blocks(ply_path, pc_level, departition_level)
    boxes = pc_2_block_oc3(blocks, bbox_max=64)
    return boxes, binstr, no_oc_voxels
#only for super resolution approach
def pc_2_block(ply_path, pc_level, departition_level):
    departition_level = int(departition_level)
    pc = PyntCloud.from_file(ply_path)
    points = pc.points[['x','y','z']]
    points=points.to_numpy()
    bbox_max = int(2 ** pc_level)
    block_oc = np.zeros((1, bbox_max, bbox_max, bbox_max, 1), dtype=np.float32)
    points = points.astype(np.uint32)
    block_oc[:,points[:, 0], points[:, 1], points[:, 2], 0] = 1.0
    no_box=int(2**departition_level)
    child_box_size=int(2**(pc_level-departition_level))
    child_blocks=[]
    for d in range(no_box):
        for h in range(no_box):
            for w in range(no_box):
                child_box = block_oc[:,d * child_box_size:(d + 1) * child_box_size,
                            h * child_box_size:(h + 1) * child_box_size,
                            w * child_box_size:(w + 1) * child_box_size, :]
                if(np.sum(child_box)!=0):
                    location=[d * child_box_size,h * child_box_size,w * child_box_size]
                    child_blocks.append([child_box,location])
    return block_oc,child_blocks

def pc_2_xyzblock(ply_path, pc_level, departition_level):
    departition_level = int(departition_level)
    pc = PyntCloud.from_file(ply_path)
    points = pc.points[['x','y','z']]
    points=points.to_numpy()
    points = points.astype(np.uint32)
    no_box=int(2**departition_level)
    child_box_size=int(2**(pc_level-departition_level))
    child_blocks=[]
    for d in range(no_box):
        for h in range(no_box):
            for w in range(no_box):
                child_box = points[(points[:,0]>=d*child_box_size) & (points[:,0]<(d+1)*child_box_size)&(points[:,1]>=h*child_box_size) & (points[:,1]<(h+1)*child_box_size)&(points[:,2]>=w*child_box_size) & (points[:,2]<(w+1)*child_box_size)]

                if(child_box.shape[0]!=0):
                    #child_box = child_box - np.min(child_box, axis=0)
                    location=[d * child_box_size,h * child_box_size,w * child_box_size]
                    child_blocks.append([child_box.shape[0],location])
    return points,child_blocks


def pc_2_block_octree(ply_path, pc_level, departition_level):
    level = int(departition_level)
    pc = PyntCloud.from_file(ply_path)
    points = pc.points[['x','y','z']]
    no_oc_voxels = len(points)
    box = int(2 ** pc_level)
    blocks10_64, _ = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)#partition 10 bits pc to block 64 + octree 4
    blocks10_32=[]
    for bl_points in blocks10_64:
        bl_points = (bl_points - 0.01) / 2
        bl_points = np.abs(bl_points)
        bl_points = np.round(bl_points)
        bl_points = np.unique(bl_points,axis=0)
        blocks10_32.append(bl_points)
    blocks10_64 = pc_2_block_oc3(blocks10_64, 64)
    blocks9_32 = pc_2_block_oc3(blocks10_32,32)

    points = (points - 0.01) / 2
    points = np.abs(points)
    points = np.round(points)
    points = np.unique(points,axis=0)
    box = int(2 ** (pc_level-1))
    blocks9_64, binstr9 = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level-1)#partition 9 bits pc to block 64 + octree 3
    blocks9_64=pc_2_block_oc3(blocks9_64,64)

    return binstr9, blocks9_64, blocks10_64,blocks9_32, no_oc_voxels


#Octree generation from occupancy map
def depth_partition_box(binstr,box,max_level, current_level,ocv):
    current_value=0
    if current_level==max_level:
        assert box.shape[0]==2
        flatted_box=box.flat[:]
        for i,bin in enumerate(flatted_box):
            current_value+=bin*2**i
            if bin==1:
                ocv+=1
        binstr.append(int(current_value))
    else:
        curr_bbox_max=2**(max_level-current_level+1)
        child_bbox_max=int(curr_bbox_max/2)
        current_value=0
        child_value=[]
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    curr_box=box[d*child_bbox_max:(d+1)*child_bbox_max,h*child_bbox_max:(h+1)*child_bbox_max,w*child_bbox_max:(w+1)*child_bbox_max,:]
                    #print(curr_box.shape)
                    if (np.sum(curr_box)!=0.):
                        child_value.append(1)
                        _,ocv=depth_partition_box(binstr,curr_box,max_level,current_level+1,ocv)
                    else:
                        child_value.append(0)
        for i,bin in enumerate(child_value):
            current_value+=bin*2**i
        binstr.append(current_value)
    return binstr,ocv
#for block based octree coding
def depth_first_search(last_level_cnt, current_pointer, current_level, binstr, fr_table, pos_seq, level):
    current_bin = format(binstr[current_pointer], '08b')
    if (current_level == level):
        for i in range(8):
            if (current_bin[i] == '1'):
                last_level_cnt += 1
    else:
        current_level += 1
        for i in range(8):
            if (current_bin[i] == '1'):
                current_pointer += 1
                fr_table[i, binstr[current_pointer]] += 1
                pos_seq[i].append(binstr[current_pointer])
                [last_level_cnt, current_pointer, fr_table, pos_seq] = depth_first_search(last_level_cnt,
                                                                                          current_pointer,
                                                                                          current_level, binstr,
                                                                                          fr_table, pos_seq, level)
    return last_level_cnt, current_pointer, fr_table, pos_seq
def discover2(binstr, level):
    current_pointer = 0
    current_level = 1
    fr_table = np.zeros([8, 257], dtype=np.float)
    pos_seq = [[], [], [], [], [], [], [], []]
    last_level_cnt = 0
    [last_level_cnt, _, fr_table, pos_seq] = depth_first_search(last_level_cnt, current_pointer,
                                                                              current_level, binstr, fr_table, pos_seq,
                                                                              level)
    fr_table = fr_table.astype(int)
    return last_level_cnt, fr_table, pos_seq


def normalize_from_mesh(input_path,output_path,vg_size):
    pc_mesh = PyntCloud.from_file(input_path)
    print(pc_mesh)
    mesh=pc_mesh.mesh
    pc_mesh.points = pc_mesh.points.astype('float64', copy=False)
    pc_mesh.mesh = mesh

    pc = pc_mesh.get_sample("mesh_random", n=10000000, as_PyntCloud=True)
    coord=['x', 'y', 'z']
    points = pc.points.values
    print(np.min(points, axis=0), np.max(points, axis=0),len(points))
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (vg_size - 1)
    points = np.round(points)
    pc.points[coord]=points
    if len(set(pc.points.columns) - set(coord)) > 0:
        pc.points = pc.points.groupby(by=coord, sort=False).mean()
    else:
        pc.points = pc.points.drop_duplicates()
    pc.to_file(output_path)

    print(np.min(pc.points, axis=0), np.max(pc.points, axis=0),len(pc.points))
    print('Normalized pc from ', input_path,' to ', output_path)

def normalize_pc(input_path,output_path,vg_size):
    pc = PyntCloud.from_file(input_path)

    coord=['x', 'y', 'z']
    points = pc.points.values
    points=points[:,:3]
    print(points.shape)
    print('original pc',np.min(points, axis=0), np.max(points, axis=0),len(points))
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (vg_size - 1)
    points = np.round(points)
    points = np.unique(points, axis=0)
    points=pd.DataFrame(points,columns=coord)
    #points=points.drop_duplicates()
    new_pc=PyntCloud(points)
    new_pc.to_file(output_path)

    print('new pc',np.min(new_pc.points,axis=0), np.max(new_pc.points, axis=0),len(new_pc.points))
    print('Normalized pc from ', input_path,' to ', output_path)


def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  # On GPU, softmax followed by cumsum can lead to the final value being
  # slightly bigger than 1, so we clamp.
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0

'''
def removing_noises(path, departition_level, pc_level, rate):

    pc=o3d.io.read_point_cloud(path)
    ori_points=np.asarray(pc.points)
    no_points=len(pc.points)
    print('Starting filter out noise points')
    print('File contains: ',no_points, ' points')
    distances=[]
    pcd_tree = o3d.geometry.KDTreeFlann(pc)
    for i in range(len(pc.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[i], 16)
        distances.append(D_distance(ori_points[idx[:],:3]))
    removal=int(rate*no_points)
    #print(np.asarray(distances).argsort()[:-10])
    points=ori_points[np.asarray(distances).argsort()[:-removal][::-1],:]

    # block partitioning
    level = int(departition_level)
    box = int(2 ** pc_level)
    blocks, binstr2block = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    oc_blocks = pc_2_block_oc3(blocks, bbox_max=64)
    # noise octree representation
    noisy_points=ori_points[np.asarray(distances).argsort()[-removal:][::-1],:]
    no_noisy_points=len(noisy_points)
    _, noisy_binstr = timing(partition_octree)(noisy_points, [0, 0, 0], [box, box, box], pc_level)
    return oc_blocks, binstr2block, noisy_binstr,no_noisy_points

def D_distance(points):
    anchor=points[0,:]
    sub=np.square(points-anchor)
    total=np.sum(sub,axis=1)
    root=np.sqrt(total)
    return np.average(root[1:])

'''
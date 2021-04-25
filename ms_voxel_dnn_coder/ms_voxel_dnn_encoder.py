import numpy as np
import os
import argparse
import time
from utils.inout import occupancy_map_explore, pmf_to_cdf
from utils.metadata_endec import save_compressed_file
import gzip
import pickle
from training.voxel_dnn_training_torch import VoxelDNN
import torchac
import torch
import torch.nn as nn
from training.ms_voxel_cnn_training import index_hr, index_lr, MSVoxelCNN


def encoder(args):
    global Models, device, VoxelDNN8, lowest_bits
    pc_level, ply_path, output_path, signaling, downsample_level, saved_model_path, voxeldnn8_path = args
    departition_level = pc_level - 6

    # getting encoding input data
    lowest_bits = 0
    blocks, binstr, no_oc_voxels = occupancy_map_explore(ply_path, pc_level, departition_level)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # For output signal
    sequence_name = os.path.split(ply_path)[1]
    sequence = os.path.splitext(sequence_name)[0]
    output_path = output_path + str(sequence) + '/' + signaling + '/'
    os.makedirs(output_path, exist_ok=True)
    outputMSVoxelDNN = output_path + 'blocks.bin'
    outputVoxelDNN = output_path + 'baseblocks.bin'
    metadata_file = output_path + 'metadata.bin'
    heatmap_file = output_path + 'heatmap.pkl'

    start = time.time()
    # Restore MSVoxelDNN
    Models = []
    for lv in range(downsample_level):
        for gr in range(8):
            low_res = int(64 // (2 ** (lv + 1)))
            model = MSVoxelCNN(2, 1, low_res, 4, gr)
            ckp_path = saved_model_path + 'G' + str(gr) + '_lres' + str(low_res) + '/' + "best_model.pt"
            print(ckp_path)
            checkpoint = torch.load(ckp_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval().to(device)
            Models.append(model)
    print('Sucessfully loaded ', len(Models), ' models')
    heatmap = []
    no_oc_voxels = 0

    # restore torch voxelCNN
    VoxelDNN8 = VoxelDNN(depth=8, height=8, width=8, residual_blocks=2, n_filters=64)
    checkpoint = torch.load(voxeldnn8_path + 'best_model.pt')
    VoxelDNN8.load_state_dict(checkpoint['state_dict'])
    VoxelDNN8.eval().to(device)

    # Encoding block
    i = 0
    print('Encoding ', blocks.shape[0], ' blocks ')
    with open(outputMSVoxelDNN, 'wb') as bitoutms, open(outputVoxelDNN, 'wb') as bitoutvx:
        for block in blocks:
            print('Encoding block ', i, ' over ', blocks.shape[0], ' blocks ', end='\r')
            encode_log, ocv = blockCoder(block, bitoutms, bitoutvx, downsample_level)
            heatmap.append(encode_log)
            no_oc_voxels += ocv
            i += 1

    with open(heatmap_file, 'wb') as f:
        pickle.dump(heatmap, f)
    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)

    total_no_ocv = no_oc_voxels

    basebits = int(os.stat(outputVoxelDNN).st_size) * 8
    file_size = int(os.stat(outputMSVoxelDNN).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8
    total_size = file_size + metadata_size + basebits
    avg_bpov = (total_size) / total_no_ocv
    print('Encoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)
    print('Occupied Voxels: %04d' % total_no_ocv)
    print('Blocks bitstream: ', outputMSVoxelDNN)
    print('Baseblock bitstream: ', outputVoxelDNN)
    print('Metadata bitstream', metadata_file)
    print('Encoding information: ', heatmap_file)
    print('Average bits per occupied voxels: %.04f' % avg_bpov)
    print('Percent of base level: %.04f' % (basebits / total_size))
    print('Total file size: ', total_size)


def blockCoder(block, bitout_msvoxeldnn, bitout_voxeldnn, downsample_level):
    global device, lowest_bits
    d, h, w, _ = block.shape
    block = torch.from_numpy(block).view(1, 1, d, h, w)
    ocv = torch.sum(block).item()
    heatmap = []
    for lv in range(downsample_level):
        ds_sampler = nn.Sequential(*[nn.MaxPool3d(kernel_size=2) for _ in range(lv + 1)]).to(device)
        curr_sampler = nn.Sequential(*[nn.MaxPool3d(kernel_size=2) for _ in range(lv)]).to(device)
        curr_block = curr_sampler(block.detach().clone().to(device))
        ds_block = ds_sampler(block.detach().clone().to(device))
        total_bits = 0
        for gr in range(8):
            predicted_probs = predictor(curr_block, ds_block, lv, gr)
            bytestream = torchacCoder(curr_block, predicted_probs, ds_block, gr)
            total_bits += len(bytestream * 8)
            bitout_msvoxeldnn.write(bytestream)

        heatmap.append([total_bits, torch.sum(curr_block).item()])
        if lv == downsample_level - 1:
            bits = baseLevelCoder(ds_block, bitout_voxeldnn)
            lowest_bits += bits
            heatmap.append([bits, torch.sum(ds_block).item()])
    return heatmap, ocv


def predictor(curr_block, ds_block, curr_lv, group):
    global Models, device
    _, _, d, h, w = curr_block.shape
    if group == 0:
        input = ds_block
    else:
        index = index_hr(group - 1, d, h, w)
        input = curr_block[:, :, index[0][:, None, None], index[1][:, None], index[2]]

        _, _, ld, lh, lw = input.shape
        index_0 = index_lr(group - 1, ld, lh, lw)
        if (index_0 is not None):
            input[:, :, index_0[0][:, None, None], index_0[1][:, None], index_0[2]] = 0
        if (group == 5):
            input[:, :, 1:ld:2, 1:lh:2, :] = 0
    input = input.to(device)
    group_prediction = Models[curr_lv * 8 + group](input)
    return torch.nn.Softmax(dim=0)(group_prediction[0])


def torchacCoder(curr_block, predicted_probs, ds_block, group):
    _, _, d, h, w = curr_block.shape
    idx = np.unravel_index(group, (2, 2, 2))
    curr_block = curr_block[0, 0, idx[0]:d:2, idx[1]:h:2, idx[2]:w:2]
    pd, ph, pw = curr_block.shape
    ds_block = ds_block.type(torch.bool).view(pd, ph, pw)
    syms = curr_block[ds_block].type(torch.int16)
    probs = predicted_probs[:, ds_block].transpose(0, 1)

    predicted_cdf = pmf_to_cdf(probs)
    predicted_cdf = predicted_cdf.detach().cpu()
    filtered_curr_block = syms.detach().cpu()
    byte_stream = torchac.encode_float_cdf(predicted_cdf, filtered_curr_block, check_input_bounds=True)
    return byte_stream


def baseLevelCoder(box, bitout):
    global device, VoxelDNN8

    # Torch voxeldnn model
    probs = torch.nn.Softmax(dim=0)(VoxelDNN8(box)[0])
    probs = probs.permute(1, 2, 3, 0)
    block = box[0, 0, :, :, :]
    block = block.type(torch.int16).cpu()

    probs = pmf_to_cdf(probs.cpu())
    byte_stream = torchac.encode_float_cdf(probs, block, check_input_bounds=True)
    curr_bit = len(byte_stream) * 8
    if (bitout != 1):
        bitout.write(byte_stream)
    return curr_bit


# Main launcher
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')
    parser.add_argument("-depth", '--downsamplingdepth', type=int,
                        default=3,
                        help='max depth to downsample, depth = 3: base block is 8')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-output", '--outputpath', type=str, help='path to output files', default='t')
    parser.add_argument("-model", '--modelpath', type=str, help='path to input model file', default='t')

    parser.add_argument("-signaling", '--signaling', type=str, help='special character for the output', default='t')

    parser.add_argument("-model8", '--modelpath8', type=str, help='path to input model 8 .h5 file')
    args = parser.parse_args()
    encoder([args.octreedepth, args.plypath, args.outputpath, args.signaling, args.downsamplingdepth, args.modelpath,
             args.modelpath8])
# python3 -m  ms_voxel_dnn_coder.ms_voxel_dnn_encoder -level 10 -depth 3 -ply ../PCC2/TestPC/MPEG_thaidancer_viewdep_vox10.ply -output Output/ -signaling msvxdnn -model Model/MSVoxelDNN/ -model8 ../PCC2/Model/VoxelDNNTorch/BL8_tf0/
# ../PCC2/Model/VoxelDNNTorch/BL8_tf0/ TestPC/Microsoft_ricardo10_vox10_0011.ply
# python3 -m  ms_voxel_dnn_coder.ms_voxel_dnn_encoder -level 10 -depth 3 -ply ../PCC2/TestPC/Microsoft_ricardo10_vox10_0011.ply -output Output/ -signaling msvxdnn -model Model/MSVoxelDNN/ -model8 ../PCC2/Model/VoxelDNNTorch/BL8_tf0/

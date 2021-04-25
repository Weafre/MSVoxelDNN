import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import os
import time
from torchvision import datasets, transforms
import argparse
from torchsummary import summary
from pyntcloud import PyntCloud
from glob import glob
from utils.training_tools import save_ckp,load_ckp,compute_metric, Rotation, Random_sampling
import datetime
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

'''RESNET BLOCKS'''
def conv_bn(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, *args, **kwargs), nn.BatchNorm3d(out_channels))

class ResnetBlock(nn.Module):
  def __init__(self, in_filters, out_filters, *args, **kwargs):
    super().__init__()
    self.in_filters=in_filters
    self.out_filters=out_filters
    self.block=nn.Sequential(
        conv_bn(self.in_filters,self.out_filters,kernel_size=1,stride=1, padding=0,bias=False),
        nn.ReLU(),
        conv_bn(self.out_filters,self.out_filters,kernel_size=3,stride=1, padding=1,bias=False),
        nn.ReLU(),
        conv_bn(self.out_filters,self.in_filters,kernel_size=1,stride=1, padding=0,bias=False),
    )
  def forward(self,x):
    identity=x
    out=self.block(x)
    out=out+identity
    out=F.relu(out)
    return out
class ResNet_kkk(nn.Module):
  def __init__(self,in_filters,out_filters,nblocks,block):
    super().__init__()
    self.in_filters=in_filters
    self.out_filters=out_filters
    self.nblocks=nblocks
    self.block=block
    self.blocks=nn.Sequential(
        nn.Conv3d(self.in_filters,self.out_filters,kernel_size=7,stride=1,padding=3,bias=False),
        *[block(self.out_filters, self.out_filters) for _ in range(self.nblocks)]
    )
  def forward(self,x):
    out=self.blocks(x)
    return out

class ResNet_kk2k(nn.Module):
  def __init__(self,in_filters,out_filters,nblocks,block):
    super().__init__()
    self.in_filters=in_filters
    self.out_filters=out_filters
    self.nblocks=nblocks
    self.block=block
    self.blocks=nn.Sequential(
        nn.Conv3d(self.in_filters,self.out_filters,kernel_size=7,stride=1,padding=(3,3,3),bias=False),
        nn.Conv3d(self.out_filters,self.out_filters,kernel_size=3,stride=(1,1,2),padding=(1,1,1),bias=False),
        *[block(self.out_filters, self.out_filters) for _ in range(self.nblocks)]
    )
  def forward(self,x):
    out=self.blocks(x)
    return out

class ResNet_k2k2k(nn.Module):
  def __init__(self,in_filters,out_filters,nblocks,block):
    super().__init__()
    self.in_filters=in_filters
    self.out_filters=out_filters
    self.nblocks=nblocks
    self.block=block
    self.blocks=nn.Sequential(
        nn.Conv3d(self.in_filters,self.out_filters,kernel_size=7,stride=1,padding=(3,3,3),bias=False),
        nn.Conv3d(self.out_filters,self.out_filters,kernel_size=3,stride=(1,2,2),padding=(1,1,1),bias=False),
        *[block(self.out_filters, self.out_filters) for _ in range(self.nblocks)]
    )
  def forward(self,x):
    out=self.blocks(x)
    return out

class ResNet_2k2k2k(nn.Module):
  def __init__(self,in_filters,out_filters,nblocks,block):
    super().__init__()
    self.in_filters=in_filters
    self.out_filters=out_filters
    self.nblocks=nblocks
    self.block=block
    self.blocks=nn.Sequential(
        nn.Conv3d(self.in_filters,self.out_filters,kernel_size=7,stride=1,padding=(3,3,3),bias=False),
        nn.Conv3d(self.out_filters,self.out_filters,kernel_size=3,stride=(2,2,2),padding=(1,1,1),bias=False),
        *[block(self.out_filters, self.out_filters) for _ in range(self.nblocks)]
    )
  def forward(self,x):
    out=self.blocks(x)
    return out

'''Voxel CNN BLOCKS'''


class maskedConv3D(nn.Conv3d):
    def __init__(self, masktype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, _, kD, kH, kW = self.weight.size()

        self.mask.fill_(1)
        self.mask[:, :, kD // 2, kH // 2, kW // 2 + (masktype == 'B'):] = 0
        self.mask[:, :, kD // 2, kH // 2 + 1:, :] = 0
        self.mask[:, :, kD // 2 + 1:, :, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(maskedConv3D, self).forward(x)


class maskedResnet(nn.Module):
    def __init__(self, no_filters):
        super().__init__()
        self.no_filters = no_filters
        self.conv2a = nn.Conv3d(in_channels=2 * self.no_filters, out_channels=self.no_filters, kernel_size=1, stride=1,
                                padding=0)
        self.conv2b = maskedConv3D(masktype='B', in_channels=self.no_filters, out_channels=self.no_filters,
                                   kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv3d(in_channels=self.no_filters, out_channels=2 * self.no_filters, kernel_size=1, stride=1,
                                padding=0)

    def forward(self, x):
        identity = x
        out = self.conv2a(x)
        out = F.relu(out)
        out = self.conv2b(out)
        out = F.relu(out)
        out = self.conv2c(out)
        out += identity
        return out


class VoxelCNN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.pixelcnn = nn.Sequential(
            maskedConv3D(masktype='A', in_channels=input_channel, out_channels=64, kernel_size=7, stride=1, padding=3),
            maskedResnet(32),
            maskedConv3D(masktype='B', in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            maskedConv3D(masktype='B', in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        # print(x.size())
        batch, cin, d, h, w = x.size()
        # print(batch, cin, h,w)
        assert torch.sum(torch.isnan(x)) == 0
        out = self.pixelcnn(x)
        # print(out.size())
        out = out.view(batch, 2, d, h, w)
        # out = out.permute(0, 1, 3, 4, 2)
        # print(out.shape)
        return out

''' BUILDING PIPELINE, MERGE AND SPLIT'''
class MSVoxelCNN(nn.Module):
  def __init__(self, Mpatch, input_channel,input_size,no_resnet,group):
    super().__init__()
    self.Mpatch=Mpatch
    self.input_channel=input_channel
    self.input_size=input_size
    self.VoxelCNN=VoxelCNN(32)
    self.patch_size=self.input_size//self.Mpatch
    self.group = group
    if (self.group <= 1):
        self.Resnet = ResNet_kkk(1, 32, no_resnet,
                                 ResnetBlock)  # 1 is number of input channel ,32 is number of output, 12 is number of resnet block
    elif (self.group == 2):
        self.Resnet = ResNet_kk2k(1, 32, no_resnet, ResnetBlock)
    elif (self.group == 3 or self.group == 4):
        self.Resnet = ResNet_k2k2k(1, 32, no_resnet, ResnetBlock)
    else:
        self.Resnet = ResNet_2k2k2k(1, 32, no_resnet, ResnetBlock)
  def forward(self,x):

    #x=self.maxpooling(x)
    ResnetFeature = self.Resnet(x)
    patches = ResnetFeature.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
    unfold_shape = patches.size()
    patches_orig = torch.zeros(unfold_shape[0],2,unfold_shape[2],unfold_shape[3],unfold_shape[4],unfold_shape[5],unfold_shape[6],unfold_shape[7])
    for i in range (unfold_shape[2]):
      for j in range (unfold_shape[3]):
        for k in range (unfold_shape[4]):
          patches_orig[:,:,i,j,k,:,:,:]= self.VoxelCNN(patches[:,:,i,j,k,:,:,:])
    output_d = unfold_shape[2] * unfold_shape[5]
    output_h = unfold_shape[3] * unfold_shape[6]
    output_w = unfold_shape[4] * unfold_shape[7]
    patches_orig = patches_orig.permute(0,1,2,5,3,6,4,7).contiguous()
    patches_orig = patches_orig.view(unfold_shape[0],2,output_d, output_h, output_w)
    return patches_orig

class PCdataset(Dataset):
  def __init__(self, files, transforms=None):
    self.files=np.asarray(files)
    self.transforms=transforms
  def __len__(self):
    return len(self.files)
  def __getitem__(self, idx):
    pc=PyntCloud.from_file(self.files[idx])
    points=pc.points.to_numpy()[:,:3]

    if(self.transforms):
        points=self.transforms(points)
    try:
        points = np.unique(points, axis=0)
    except:
        return None
    points=torch.from_numpy(points).type(torch.LongTensor)

    #print(points.shape)
    v=torch.ones(points.shape[0])
    dense_block=torch.sparse.FloatTensor(torch.transpose( points,0,1),v, torch.Size([64,64,64])).to_dense().view(1,64,64,64)
    #print(dense_block.shape, torch.max(dense_block), torch.min(dense_block), torch.count_nonzero(dense_block))
    return dense_block
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def data_collector(training_dirs,params):
    total_files = []
    for training_dir in training_dirs:
        training_dir = training_dir + '**/*.ply'

        files = glob(training_dir, recursive=True)
        print('Total files: ',len(files))
        total_files_len = len(files)
        total_files = np.concatenate((total_files, files), axis=0)
        print('Selected ', len(files), ' from ', total_files_len, ' in ', training_dir)

    assert len(total_files) > 0
    rn.shuffle(total_files)  # shuffle file
    print('Total blocks for training: ', len(total_files))
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in total_files])
    files_train = total_files[files_cat == 'train']
    files_valid = total_files[files_cat == 'test']

    rotation = Rotation(64)
    sampling = Random_sampling()
    #rotation, sampling,
    transforms_ = Compose([rotation, sampling])
    #,transforms.ToTensor()

    training_set = PCdataset(files_train)
    training_generator = torch.utils.data.DataLoader(training_set,collate_fn=collate_fn, **params)

    # Validation data
    valid_set = PCdataset(files_valid)
    valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn=collate_fn,**params)
    return training_generator, valid_generator
def index_hr(group,d,h,w):#generate the index to select high resolution from block d,h,w for input
  index=[[np.arange(0,d,2),np.arange(0,h,2),np.arange(0,w,2)],
       [np.arange(0,d,2),np.arange(0,h,2),np.arange(0,w,1)],
       [np.arange(0,d,2),np.arange(0,h,1),np.arange(0,w,1)],
       [np.arange(0,d,2),np.arange(0,h,1),np.arange(0,w,1)],
       [np.arange(0,d,1),np.arange(0,h,1),np.arange(0,w,1)],
       [np.arange(0,d,1),np.arange(0,h,1),np.arange(0,w,1)],
       [np.arange(0,d,1),np.arange(0,h,1),np.arange(0,w,1)]]
  return index[group]
def index_lr(group, d,h,w):#generate the index of the value that will be assigned to 0
  assign_zeros=[None,
              None,
              [np.arange(0,d,1),np.arange(1,h,2),np.arange(1,w,2)],
              None,
              [np.arange(1,d,2),np.arange(0,h,2),np.arange(1,w,2)],
              [np.arange(1,d,2),np.arange(1,h,2),np.arange(0,w,1)],
              [np.arange(1,d,2),np.arange(1,h,2),np.arange(1,w,2)]]
  return assign_zeros[group]
def train(use_cuda, batch_size,low_res, max_epochs,group, output_model, dataset_path, valid_loss_min, model,
          optimizer, start_epoch, ds_level):
    #tensorboard writer:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_model + 'log' + current_time + '/train'
    test_log_dir = output_model + 'log' + current_time + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    #checkpoint setup
    checkpoint_path = output_model + "current_checkpoint.pt"
    best_model_path = output_model + "best_model.pt"
    eps = 1e-8
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2}

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    training_generator, valid_generator=data_collector(dataset_path,params)
    maxpool_n1 = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(ds_level - 1)]
    )
    maxpool_n = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(ds_level)]
    )

    train_loss = 0
    train_losses = []
    best_val_epoch = None
    output_period=len(training_generator)//20
    block_size=int(low_res*2)
    idx = np.unravel_index(group, (2, 2, 2))
    print('Start the training for group: ', group, ' with lower resoltuion is ', low_res)
    for epoch in range(start_epoch, max_epochs):
        for batch_idx, x in enumerate(training_generator):
            x=x.to(device)

            if (group == 0):
                input = maxpool_n(x)
            else:
                input = maxpool_n1(x)
                index = index_hr(group - 1, block_size, block_size, block_size)
                input = input[:, :, index[0][:, None, None], index[1][:, None], index[2]]

                _, _, ld, lh, lw = input.shape
                index_0 = index_lr(group - 1, ld, lh, lw)
                if (index_0 is not None):
                    input[:, :, index_0[0][:, None, None], index_0[1][:, None], index_0[2]] = 0
                #if(batch_idx==1):
                #    print(input.shape)
                if (group == 5):
                    input[:, :, 1:ld:2, 1:lh:2, :]=0
            if(epoch==0 and batch_idx==0):
                print('Input shape: ', input.shape)
            target = maxpool_n1(x.clone().detach())[:, :, idx[0]:block_size:2, idx[1]:block_size:2, idx[2]:block_size:2].view(x.shape[0], low_res, low_res,
                                                                                 low_res).type(torch.LongTensor)
            predict = model(input) + eps
            #print(predict.shape, torch.max(target), torch.min(target), target.shape)
            loss = F.cross_entropy(predict,target)  # predict: shape of input: https://pytorch.org/docs/stable/nn.functional.html#cross-entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            tp, fp, tn, fn, precision, recall, accuracy, specificity, f1 = compute_metric(predict, target,
                                                                                          train_summary_writer, len(
                    training_generator) * epoch + batch_idx)
            train_summary_writer.add_scalar("bc/loss", train_loss, len(training_generator) * epoch + batch_idx)
            if(batch_idx%output_period==0):
                print("Batch {} over {}:  \tloss : {:.6f}\t accuracy : {:.3f} tp : {:.2f} fp : {:.2f} tn : {:.2f} fn : {:.2f} f1 : {:.4f}".format(batch_idx,len(training_generator), train_loss, accuracy, tp, fp, tn, fn, f1), end='\r')
            del loss, target,predict
        train_losses.append(train_loss)
        #print(train_losses)


        # validation
        valid_loss = 0
        model.eval()
        for batch_idx, x in enumerate(valid_generator):
            x=x.to(device)
            if (group == 0):
                input = maxpool_n(x)
            else:
                input = maxpool_n1(x)
                index = index_hr(group - 1, block_size, block_size, block_size)
                input = input[:, :, index[0][:, None, None], index[1][:, None], index[2]]

                _, _, ld, lh, lw = input.shape
                index_0 = index_lr(group - 1, ld, lh, lw)
                if (index_0 is not None):
                    input[:, :, index_0[0][:, None, None], index_0[1][:, None], index_0[2]] = 0
                if (group == 5):
                    input[:, :, 1:ld:2, 1:lh:2, :]=0

            target = maxpool_n1(x.clone().detach())[:, :, idx[0]:block_size:2, idx[1]:block_size:2, idx[2]:block_size:2].view(
                x.shape[0], low_res, low_res,
                low_res).type(torch.LongTensor)
            output = model(input) + eps
            loss = F.cross_entropy(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            valid_loss=valid_loss
            del loss, target, output
        test_summary_writer.add_scalar("bc/loss", valid_loss, epoch)

        print('Training for group: ', group, ' downsample level: ', ds_level)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss))

        # saving model
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if valid_loss <= valid_loss_min or best_val_epoch ==None:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
            best_val_epoch=epoch
        if(epoch-best_val_epoch>=10):
            print('Early stopping detected')
            break




# /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/
if __name__ == "__main__":
    # Command line main application function.
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-scratch", '--scratch', type=lambda x: (str(x).lower() in ['true','1', 'yes']),
                        default=False,
                        help='Training from scratch or checkpoint')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True,
                        help='using cuda or not')
    parser.add_argument("-batch", '--batch', type=int,
                        default=8,
                        help='batch size')
    parser.add_argument("-blocksize", '--blocksize', type=int,
                        default=64,
                        help='training block size')
    parser.add_argument("-epoch", '--epoch', type=int,
                        default=10,
                        help='number of epochs')
    parser.add_argument("-group", '--group', type=int,
                        default=2,
                        help='building which model?')
    parser.add_argument("-downsample", '--dslevel', type=int,
                        default=2,
                        help='number of downsampling step until group 1 level')
    parser.add_argument("-nopatches", '--patches', type=int,
                        default=4,
                        help='Number of patches in spliting step')
    parser.add_argument("-noresnet", '--noresnet', type=int,
                        default=4,
                        help='Number of patches in spliting step')
    parser.add_argument("-inputmodel", '--savedmodel', type=str, help='path to saved model file')
    # parser.add_argument("-loss", '--loss_img_name', type=str, help='name of loss image')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-dataset", '--dataset', action='append', type=str, help='path to dataset ')
    parser.add_argument("-validation", '--validation', type=str, help='path to validation dataset ')
    parser.add_argument("-portion_data", '--portion_data', type=float,
                        default=1,
                        help='portion of dataset to put in training, densier pc are selected first')
    args = parser.parse_args()
    low_res=int(64/(2**args.dslevel))
    model = MSVoxelCNN(args.patches, 1, low_res, args.noresnet,args.group)
    maxpool = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(args.dslevel - 1)]
    )
    valid_loss_min = np.Inf
    device = torch.device("cuda" if args.usecuda else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    start_epoch = 0
    output_path=args.saving_model_path+'G'+str(args.group)+'_lres'+str(low_res)+'/'
    os.makedirs(output_path,exist_ok=True)
    if (args.scratch):
        print("Training from scratch \n")
        train(args.usecuda, args.batch,low_res, args.epoch, args.group, output_path, args.dataset,
              valid_loss_min, model, optimizer, start_epoch, args.dslevel)
    else:
        ckp_path = output_path + "current_checkpoint.pt"
        model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
        print('Successfully loaded model \n')
        train(args.usecuda, args.batch,low_res, args.epoch,args.group, output_path, args.dataset,
              valid_loss_min, model, optimizer, start_epoch, args.dslevel)
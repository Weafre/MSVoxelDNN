# VoxelCNN
import random as rn
import numpy as np

from utils.inout import input_fn_voxel_dnn, get_shape_data, get_files, load_points
import os
import sys
import argparse
import datetime
from utils.training_tools import save_ckp,load_ckp,compute_metric, Rotation, Random_sampling
import datetime
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from glob import glob
import math
from torchsummary import summary
random_seed = 42

np.random.seed(random_seed)
rn.seed(random_seed)

class maskedConv3D(nn.Conv3d):
    def __init__(self, masktype, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, _, kD, kH, kW = self.weight.size()

        self.mask.fill_(1)

        self.mask[:, :, kD // 2, kH // 2, kW // 2 + (masktype == 'B'):] = 0
        self.mask[:, :, kD // 2, kH // 2 + 1:, :] = 0
        self.mask[:, :, kD // 2 + 1:, :, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(maskedConv3D, self).forward(x)


class residualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.no_filters=h
        self.conva=nn.Conv3d(in_channels=2*h, out_channels=h,kernel_size=1, stride=1, padding=0 )
        self.convb=maskedConv3D(masktype='B',in_channels=h, out_channels=h, kernel_size=5, stride=1, padding= 2)
        self.convc = nn.Conv3d(in_channels=h, out_channels=2*h, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        out = self.conva(x)
        out = F.relu(out)
        out = self.convb(out)
        out = F.relu(out)
        out = self.convc(out)
        out += identity
        return out

class VoxelDNN(nn.Module):
    def __init__(self,depth = 64, height = 64, width = 64, n_channel = 1, output_channel = 2, residual_blocks = 2, n_filters = 64):
        self.depth = depth
        self.height = height
        self.width = width
        self.n_channel = n_channel
        self.output_channel = output_channel
        self.residual_blocks = residual_blocks
        self.n_filters = n_filters
        self.init__ = super(VoxelDNN, self).__init__()
        self.voxelcnn = nn.Sequential(
            maskedConv3D(masktype='A', in_channels=self.n_channel, out_channels=self.n_filters, kernel_size=7, stride=1, padding=3),
            *[residualBlock(self.n_filters//2) for _ in range(self.residual_blocks)],
            nn.ReLU(),
            nn.Conv3d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels=self.n_filters, out_channels=self.output_channel, kernel_size=1, stride=1, padding=0),
        )
    def forward(self,x):
        out=self.voxelcnn(x)
        return out


class PCdataset(Dataset):
  def __init__(self, files,block_size, transforms=None):
    self.files=np.asarray(files)
    self.transforms=transforms
    self.bz=block_size
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
    v=torch.ones(points.shape[0])
    #print('number of points: ', points.shape)
    dense_block=torch.sparse.FloatTensor(torch.transpose( points,0,1),v, torch.Size([self.bz,self.bz,self.bz])).to_dense().view(1,self.bz,self.bz,self.bz)
    #print('Number of ocv voxels: ', torch.sum(dense_block))
    return dense_block
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def data_collector(training_dirs,transform_idx,params):
    total_files = []
    for training_dir in training_dirs:
        training_dir = training_dir + '**/*.ply'

        files = glob(training_dir, recursive=True)
        print('Total files: ',len(files))
        total_files_len = len(files)
        total_files = np.concatenate((total_files, files), axis=0)
        print('Selected ', len(files), ' from ', total_files_len, ' in ', training_dir)

    #total_files=total_files[:5000]
    assert len(total_files) > 0
    rn.shuffle(total_files)  # shuffle file
    print('Total blocks for training: ', len(total_files))
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in total_files])
    files_train = total_files[files_cat == 'train']
    files_valid = total_files[files_cat == 'test']

    rotation = Rotation(64)
    sampling = Random_sampling()
    #rotation, sampling,
    both = Compose([rotation, sampling])
    #,transforms.ToTensor()
    transformations=[None, rotation, sampling, both]

    training_set = PCdataset(files_train,64, transformations[transform_idx])
    training_generator = torch.utils.data.DataLoader(training_set,collate_fn=collate_fn, **params)

    # Validation data
    valid_set = PCdataset(files_valid, 64, transformations[transform_idx])
    valid_generator = torch.utils.data.DataLoader(valid_set, collate_fn=collate_fn,**params)
    return training_generator, valid_generator


def train(use_cuda, batch_size, max_epochs, output_model, dataset_path, valid_loss_min, model,
          optimizer, start_epoch, block_size, transformations):
    #tensorboard writer:
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_model + 'log' + '/train'
    test_log_dir = output_model + 'log' + '/test'
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
    print('Optimizer state: ', optimizer)

    training_generator, valid_generator=data_collector(dataset_path,transformations,params)
    ds_level=int(math.log2((64//block_size)))

    maxpool_n = nn.Sequential(
        *[nn.MaxPool3d(kernel_size=2) for _ in range(ds_level)]
    ).to(device)

    train_loss = 0
    train_losses = []
    best_val_epoch = None
    output_period=len(training_generator)//20

    for epoch in range(start_epoch, max_epochs):
        for batch_idx, x in enumerate(training_generator):
            x=x.to(device)
            x_donwscale=maxpool_n(x)
            # x_donwscale = torch.ones_like(x_donwscale).to(device)

            if(epoch==0 and batch_idx==0):
                print('Input shape: ', x_donwscale.shape)
            target = x_donwscale
            db, _, dd, dh, dw = x_donwscale.shape
            target=target.view(db,dd,dh, dw).type(torch.LongTensor).to(x_donwscale.device)

            predict = model(x_donwscale) + eps
            loss = F.cross_entropy(predict, target)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            tp, fp, tn, fn, precision, recall, accuracy, specificity, f1 = compute_metric(predict, target,
                                                                                          train_summary_writer, len(
                    training_generator) * epoch + batch_idx)
            train_summary_writer.add_scalar("bc/loss", train_loss, len(training_generator) * epoch + batch_idx)
            if((batch_idx%output_period)==0):
                #ocv=1-torch.sum(x_donwscale)/(db*dd*dh*dw)
                print("Batch {} over {}:  \tloss : {:.6f}\t accuracy : {:.3f} tp : {:.2f} fp : {:.2f} tn : {:.2f} fn : {:.2f} F1 : {:.4f}".format(batch_idx,len(training_generator), train_loss, accuracy, tp, fp, tn, fn, f1), end='\r')
            del loss, target,predict
            #break
        train_losses.append(train_loss)
        #print(train_losses)


        # validation
        with torch.no_grad():
            valid_loss = 0
            model.eval()
            for batch_idx, x in enumerate(valid_generator):
                x = x.to(device)
                x_donwscale = maxpool_n(x)

                target = x_donwscale

                db, _, dd, dh, dw = x_donwscale.shape
                target = target.view(db, dd, dh, dw).type(torch.LongTensor).to(x_donwscale.device)

                predict = model(x_donwscale) + eps
                loss = F.cross_entropy(predict, target)


                #loss = F.cross_entropy(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                del loss, target, predict
            test_summary_writer.add_scalar("bc/loss", valid_loss, epoch)

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





if __name__ == "__main__":
    # Command line main application function.
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-blocksize", '--block_size', type=int,
                        default=64,
                        help='input size of block')
    parser.add_argument("-nfilters", '--n_filters', type=int,
                        default=64,
                        help='Number of filters')
    parser.add_argument("-batch", '--batch_size', type=int,
                        default=2,
                        help='batch size')
    parser.add_argument("-epochs", '--epochs', type=int,
                        default=2,
                        help='number of training epochs')
    parser.add_argument("-inputmodel", '--savedmodel', type=str, help='path to saved model file')
    #parser.add_argument("-loss", '--loss_img_name', type=str, help='name of loss image')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-dataset", '--dataset', action='append', type=str, help='path to dataset ')
    parser.add_argument("-portion_data", '--portion_data', type=float,
                        default=1,
                        help='portion of dataset to put in training, densier pc are selected first')
    parser.add_argument("-scratch", '--scratch', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False,
                        help='Training from scratch or checkpoint')
    parser.add_argument("-usecuda", '--usecuda', type=bool,
                        default=True,
                        help='using cuda or not')
    parser.add_argument("-lr", '--lr', type=int,
                        default=3,
                        help='index of learning rate: 1e-5,5e-5, 1e-4, 1e-3')
    parser.add_argument("-tf", '--transformations', type=int,
                        default=0,
                        help='0: none, 1: rotation, 2: random sampling, 3: both')

    args = parser.parse_args()
    block_size=args.block_size
    model = VoxelDNN(depth=block_size,height=block_size,width=block_size,n_channel=1,output_channel=2,residual_blocks=2,n_filters=args.n_filters)

    valid_loss_min = np.Inf
    device = torch.device("cuda" if args.usecuda else "cpu")
    model = model.to(device)
    summary(model, (1, block_size, block_size, block_size))
    print(model)

    lrs=[1e-6,1e-5,5e-5, 1e-4, 1e-3, 1e-2]
    lr=lrs[args.lr]
    print('Selected learning rate: ', lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    start_epoch = 0
    output_path=args.saving_model_path+'BL'+str(args.block_size)+'_tf'+str(args.transformations)+'/'
    os.makedirs(output_path,exist_ok=True)

    if (args.scratch):
        print("Training from scratch \n")
        train(args.usecuda, args.batch_size, args.epochs, output_path, args.dataset,
              valid_loss_min, model, optimizer, start_epoch, args.block_size,args.transformations)
    else:
        try:
            ckp_path = output_path + "best_model.pt"
            model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer) #load optimize and start epooch from check point


            print('Successfully loaded model \n')
            train(args.usecuda, args.batch_size, args.epochs, output_path, args.dataset,
                  valid_loss_min, model, optimizer, start_epoch, args.block_size, args.transformations)
        except:
            train(args.usecuda, args.batch_size, args.epochs, output_path, args.dataset,
                  valid_loss_min, model, optimizer, start_epoch, args.block_size, args.transformations)

#python3 -m training.voxel_dnn_training -blocksize 64 -nfilters 64 -inputmodel Model/voxelDNN_CAT1 -outputmodel Model/voxelDNN_CAT1 -dataset /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/ -dataset /datnguyen_dataset/database/CAT1/cat1_selected_vox10_oct4/ -batch 8 -epochs 30
#python3 -m training.voxel_mixing_context -epoch 50 -blocksize 64 -outputmodel Model/voxelDnnSuperRes/ -inputmodel Model/voxelDnnSuperRes -dataset /datnguyen_dataset/database/Microsoft/10bitdepth_selected_oct4/ -dataset /datnguyen_dataset/database/MPEG/selected_8i_oct4/  -dataset /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/ -batch 8 -nfilters 64

# python3 -m training.voxel_dnn_training_torch -usecuda 1 -dataset /datnguyen_dataset/database/Modelnet40/ModelNet40_200_pc512_oct3/ -dataset /datnguyen_dataset/database/Microsoft/10bitdepth_selected_oct4/ -dataset /datnguyen_dataset/database/MPEG/selected_8i_oct4/ -dataset /datnguyen_dataset/database/CAT1/cat1_selected_vox10_oct4/ -outputmodel Model/MSVoxelCNNP/ -epoch 100 --scratch=0  -batch 32  -nopatches 2 -group 1 -downsample 1 -noresnet 4
# # python3 -m training.voxel_dnn_training_torch -usecuda 1  -dataset /datnguyen_dataset/database/MPEG/selected_8i_oct4/  -outputmodel Model/VoxelDNNTorch/ -epoch 1 --scratch=1  -batch 32

# new platform :
# python3 -m training.voxel_dnn_training_torch -usecuda 1  -dataset ../../Datasets/ModelNet40_200_pc512_oct3/  -outputmodel Model/VoxelDNNTorch/ -epoch 3 --scratch=1  -batch 8 -tf 0 -lr 4
# python3 -m training.voxel_dnn_training_torch -usecuda 1  -dataset /Users/thanhdatnguyen/Documents/Works/INTERN/src/OctreeCoding/block_64/  -outputmodel Model/VoxelDNNTorch/ -epoch 3 --scratch=1  -batch 8 -tf 0 -lr 4 -nfilters 16 -blocksize 16
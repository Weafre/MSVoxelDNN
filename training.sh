#!/bin/bash
groups=(0 1 2 3 4 5 6 7 )
dslevels=(1 2 3)
batchsize=(32 32 64)
noise_range=(1.0 1.5 2.0)
dataset64=("Modelnet40/10bitdepth_2_oct4/" "Microsoft/10bitdepth_2_oct4/"  "MPEG/10bitdepth_2_oct4/"  "CAT1/10bitdepth_2_oct4/")

for gr in "${groups[@]}";
do
  for lv in "${dslevels[@]}";
    do
    python3 -m training.ms_voxel_cnn_training -usecuda 1 -dataset ${dataset64[0]} -dataset ${dataset64[1]} -dataset ${dataset64[2]} -dataset ${dataset64[3]} -outputmodel Model/MSVoxelCNN/ -epoch 20 --scratch=0  -batch ${batchsize[${lv}-1]}   -nopatches 2 -group $gr -downsample $lv -noresnet 4
    done
done

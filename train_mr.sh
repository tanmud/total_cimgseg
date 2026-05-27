for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 002 3d_fullres $FOLD \
      -tr nnUNetTrainer_2000epochs_NoMirroring \
      -pretrained_weights $MR_WEIGHTS
done
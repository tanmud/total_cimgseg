for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 001 3d_fullres $FOLD \
      -tr nnUNetTrainerNoMirroring \
      -pretrained_weights $WEIGHTS
done
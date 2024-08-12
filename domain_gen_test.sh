python3 domain_gen_main.py \
-classnum 3 \
-mode test \
-support 1 \
-neighbor 3 \
-cuda 2 \
-bs 16 \
-dataset /home/mlmlab13/3dgcn/classification/dataset/kitti_3 \
-load /home/mlmlab13/Domain_generalization/MSCN/pretrained_model/MSCN_with_unseen_domain/MSCN_with_unseen_domain_KITTI.pkl # If you set -n_tgt to 20, you can revise i_th to 0~19

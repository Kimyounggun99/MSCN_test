# model options: mscn, MSCN_with_only_SCL, dgcnn, pointnet, pointMLP, gcn, scn, pointNext.
# classnum: 2(sim2real) or 3(real2real)
# -support and -neighbor are essential for gcn, scn, mscn, and MSCN_with_only_SCL
# -normal is essential for PointNet, PointMLP, PointNext, and DGCNN
# -dataset: please revise train_dataset to specific dataset like kitti_3.
# -record: please add your path and the file name to save history during train phase.
# -save: please add your path and the file name to save model parameters.

python3 base_main.py \
-model mscn \
-classnum 3 \
-mode train \
-cuda 0 \
-support 1 \
-neighbor 3 \
-epoch 10 \
-bs 16 \
-dataset your_paths/MSCN/dataset/train_dataset \
-record your_paths/MSCN/train/train_log_name.log \
-save your_paths/MSCN/saved_model/model_name.pkl

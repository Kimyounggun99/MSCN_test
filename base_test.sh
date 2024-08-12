# model options: mscn, MSCN_with_only_SCL, dgcnn, pointnet, pointMLP, gcn, scn, pointNext.
# classnum: 2(sim2real) or 3(real2real)
# -support and -neighbor are essential for gcn, scn, mscn, and MSCN_with_only_SCL
# -normal is essential for PointNet, PointMLP, PointNext, and DGCNN
# -dataset: please revise test_dataset to specific dataset like kitti_3.
# -record: please add your path and the file name to save history during test phase.
# -load: please add a path of trained model.

python3 base_main.py \
-model mscn \
-classnum 3 \
-mode test \
-cuda 0 \
-bs 16 \
-support 1 \
-neighbor 3 \
-dataset your_paths/dataset/target_dataset \
-record your_paths/MSCN/test/train_log_name.log \
-load your_paths/MSCN/saved_model/model_name.pkl
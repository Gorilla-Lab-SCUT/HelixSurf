# SCENE_ID: id of scannet
# DATA_ROOT: the root to save dataset
# RESULT_ROOT: the root to store mvs results
# EPOCH: the trained epoch
# PRE_MVS_SUFFIX: the last mvs suffix
# MVS_SUFFIX: the directory name of mvs results
SCENE_ID="0616_00"
DATA_ROOT="./HelixSurf_data/scene_data"
RESULT_ROOT="./HelixSurf_data/mvs_results"
EPOCH="1"
PRE_MVS_SUFFIX="mvs"
MVS_SUFFIX="mvs_from_"$EPOCH"epoch"

# save directories
RESULT_DIR=$RESULT_ROOT/$SCENE_ID
SUPERPIXEL_DIR=$RESULT_DIR/superpixel
PRE_MVS_DIR=$RESULT_DIR/$PRE_MVS_SUFFIX
MVS_DIR=$RESULT_DIR/$MVS_SUFFIX

python -m gmvs.scripts.launch -rd $RESULT_DIR --mvs_suffix $MVS_SUFFIX --dn_input --input_depth_normal_dir $PRE_MVS_DIR/ep_$EPOCH
python -m gmvs.scripts.mvs_fusion_segmentaion --depth_normal_dir $MVS_DIR/depth_normal/ \
    --data_dir $DATA_ROOT/$SCENE_ID --superpixel_dir $SUPERPIXEL_DIR/ \
    --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh
# To run mvs fusion on cpu
# python -m gmvs.scripts.mvs_fusion_segmentaion_cpu --depth_normal_dir $MVS_DIR/depth_normal/ \
#     --data_dir $DATA_ROOT/$SCENE_ID --superpixel_dir $SUPERPIXEL_DIR/ \
#     --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh
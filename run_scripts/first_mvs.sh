# SCENE_ID: id of scannet
# DATA_ROOT: the root to save dataset
# DENSE_SUFFIX: suffix of dense output (`dense` defined in COLMAP)
# RESULT_ROOT: the root to store mvs results
# MVS_SUFFIX: the directory name of mvs results
SCENE_ID="0616_00"
DATA_ROOT="./HelixSurf_data/scene_data"
RESULT_ROOT="./HelixSurf_data/mvs_results"
DENSE_SUFFIX="result_colmap/dense"
MVS_SUFFIX="mvs"

# save directories
RESULT_DIR=$RESULT_ROOT/$SCENE_ID
SUPERPIXEL_DIR=$RESULT_DIR/superpixel
MVS_DIR=$RESULT_DIR/$MVS_SUFFIX

python -m gmvs.scripts.colmap2mvs --dense_folder $DATA_ROOT/$SCENE_ID/$DENSE_SUFFIX --save_folder $RESULT_DIR
python -m gmvs.scripts.extract_superpixel --img_dir $DATA_ROOT/$SCENE_ID/images --save_dir $SUPERPIXEL_DIR
python -m gmvs.scripts.launch -rd $RESULT_DIR --mvs_suffix $MVS_SUFFIX
python -m gmvs.scripts.mvs_fusion_segmentaion --depth_normal_dir $MVS_DIR/depth_normal/ \
    --data_dir $DATA_ROOT/$SCENE_ID --superpixel_dir $SUPERPIXEL_DIR/ \
    --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh --gen_mask --mask_dir planar_mask_mvs
# To run mvs fusion on cpu
# python -m gmvs.scripts.mvs_fusion_segmentaion_cpu --depth_normal_dir $MVS_DIR/depth_normal/ \
#     --data_dir $DATA_ROOT/$SCENE_ID --superpixel_dir $SUPERPIXEL_DIR/ \
#     --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh --gen_mask --mask_dir planar_mask_mvs
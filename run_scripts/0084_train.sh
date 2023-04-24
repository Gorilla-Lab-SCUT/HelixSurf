### change here as your directory to dataset and mvs results
scene_data_dir="./HelixSurf_data/scene_data"
mvs_data_dir="./HelixSurf_data/mvs_results"

# param list
scene_id=0084_00
train_suffix=0084_00_default
vis_suffix=0084_00_default
mvs_suffix=${1} # NOTE: for different mvs results

echo using scene_id: $scene_id train_suffix: $train_suffix vis_suffix: $vis_suffix mvs_suffix: $mvs_suffix **kwargs: ${2}

python scripts/train.py \
    --data_dir ${scene_data_dir} \
    --config configs/default.yaml --scene $scene_id \
    --train_dir ckpt/$train_suffix/ --print_every 10 \
    --export_mesh resultvis/$vis_suffix/sdf_mc.ply \
    --log_depth --plane_slidewindow_size 31 \
    --mvs_dir ${mvs_data_dir}/$scene_id/$mvs_suffix/ \
    ${2}


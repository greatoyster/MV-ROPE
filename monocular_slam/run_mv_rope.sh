export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
i=1
j=1


for ((scene_num=i; scene_num<=j; scene_num++))
do
    scene_path="/home/yangjq/Datasets/NOCS/real/real_test/scene_$scene_num"
    # scene_path="/home/yangjq/Datasets/dyna_rope_data/mvrope/scene_$scene_num"
    command="python ./evaluation_scripts/test_mv_rope.py \
             --datapath $scene_path \
             --mask \
             --nocs \
             --depth \
             --zero_depth \
             --buffer 128 \
             --frontend_window 8 \
             --keyframe_thresh 2 \
             --trajectory_path ./reconstructions/scene_$scene_num/traj_est.pkl \
             --reconstruction_path scene_$scene_num \
             --disable_vis
             "    
    eval $command
done

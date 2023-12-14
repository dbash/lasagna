
export CONTROLNET_PATH="./controlnet"
img_path="./digital_art/hamburger/0.png"
obj_name="hamburger"
lighting_dir=3

entry_bname=$(basename "$img_path")
out_dir="./results/${obj_name}/${entry_bname}/lighting_${lighting_dir}"

echo "Running script for image ${img_path} with ${obj_name} object with lighting direction ${lighting_dir}";
echo "saving to ${out_dir}";
python train_shading_sds.py --img_path "$entry" \
    --controlnet_path "$CONTROLNET_PATH_NO_PRIOR" \
    --exp_root "${out_dir}" \
    --exp_name "controlnet_lasagna" --prompt "A shading of a ${obj_name} with ${lighting_dir} lighting" \
    --neg_prompt ""\
    --cfg 10 --max_train_steps 800 \
    --reg_w 1 \
    --num_images 1 --seed 21 --multilayer 1 --batch_size 4;


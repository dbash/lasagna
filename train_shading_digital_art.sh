

export CONTROLNET_PATH="./controlnet/" # insert path to the downloaded ControlNet adaptor checkpoint
img_path="./digital_art/hamburger/0.png"
obj_name="hamburger"
lighting_dir=3

entry_bname=$(basename "$img_path")
out_dir="./results/${obj_name}/${entry_bname}/lighting_${lighting_dir}"

echo "Running script for image ${img_path} with ${obj_name} object with lighting direction ${lighting_dir}";
echo "saving to ${out_dir}";
python train_shading_sds.py --img_path "$img_path" \
    --controlnet_path "$CONTROLNET_PATH" \
    --exp_root "${out_dir}" \
    --exp_name "controlnet_lasagna" --prompt "A photo of a ${obj_name} with ${lighting_dir} lighting" \
    --neg_prompt ""\
    --cfg 12 --max_train_steps 500 \
    --reg_w 500 \
    --num_images 1 --seed 21 --multilayer 1 --batch_size 4;


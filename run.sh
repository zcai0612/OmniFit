INPUT_PATH="/path/to/input_mesh.obj"
OUTPUT_DIR="/path/to/output_dir"

python infer.py \
    --input_path $INPUT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_points 15000 \
    --with_scale \
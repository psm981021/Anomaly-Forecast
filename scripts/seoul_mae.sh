python main.py \
    --data_dir "data/data_seoul_only_v2" \
    --image_csv_dir "data/Seoul_V2.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L2.01" \
    --output_dir "output/seoul-L2.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "mae_image" \
    --location "seoul" \
    --regression "gap" \
    --pre_train


python main.py \
    --data_dir "data/data_seoul_only_v2" \
    --image_csv_dir "data/Seoul_V2.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L2.01" \
    --output_dir "output/seoul-L2.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "mae_image" \
    --location "seoul" \
    --regression "gap" \
    --pre_train
    
# scripts/seoul_mae.sh
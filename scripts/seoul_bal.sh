python main.py \
    --data_dir "data/data_Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "seoul-L5.03" \
    --output_dir "output/seoul-L5.03" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \
    --pre_train \
    --classifier \
    --balancing 

python main.py \
    --data_dir "data/data_Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "seoul-L5.03" \
    --output_dir "output/seoul-L5.03" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \
    --classifier \
    --balancing \


# scripts/seoul_bal.sh

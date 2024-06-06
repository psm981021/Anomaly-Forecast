# python main.py \
#     --data_dir "data/data_Seoul" \
#     --image_csv_dir "data/Seoul.csv" \
#     --batch 8 \
#     --epochs 5000 \
#     --patience 40 \
#     --model_idx "seoul-L5.01" \
#     --output_dir "output/seoul-L5.01" \
#     --test_list=[] \
#     --gpu_id 0 \
#     --device cuda:0 \
#     --seed 0 \
#     --lr 0.001 \
#     --loss_type "stamina" \
#     --location "seoul" \
#     --regression "gap" \
#     --pre_train \
#     --balancing 

# python main.py \
#     --data_dir "data/data_Seoul" \
#     --image_csv_dir "data/Seoul.csv" \
#     --batch 8 \
#     --epochs 5000 \
#     --patience 40 \
#     --model_idx "seoul-L5.01" \
#     --output_dir "output/seoul-L5.01" \
#     --test_list=[] \
#     --gpu_id 0 \
#     --device cuda:0 \
#     --seed 0 \
#     --lr 0.06 \
#     --loss_type "stamina" \
#     --location "seoul" \
#     --regression "gap" \
#     --balancing 

python main.py \
    --data_dir "data/data_Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "seoul-L5.01" \
    --output_dir "output/seoul-L5.01" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \
    --classification \
    --balancing 

# scripts/seoul_bal.sh
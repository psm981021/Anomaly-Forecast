# python main.py \
#     --data_dir "/dev/shm/data_Seoul" \
#     --image_csv_dir "data/Seoul.csv" \
#     --batch 8 \
#     --epochs 3000 \
#     --patience 40 \
#     --model_idx "seoul-L1.01" \
#     --output_dir "output/seoul-L1.01/" \
#     --test_list=[] \
#     --gpu_id 0 \
#     --device cuda:0 \
#     --seed 0 \
#     --lr 0.001 \
#     --loss_type "ce_image" \
#     --location "seoul" \
#     --regression "gap" \
#     --pre_train

python main.py \
    --data_dir "data/data_Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L1.01" \
    --output_dir "output/seoul-L1.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.02 \
    --loss_type "ce_image" \
    --location "seoul" \
    --regression "gap" \
    --classification



# scripts/seoul_ce.sh
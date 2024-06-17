# python main.py \
#     --data_dir "data/Seoul" \
#     --image_csv_dir "data/Seoul.csv" \
#     --batch 8 \
#     --epochs 4000 \
#     --patience 40 \
#     --model_idx "seoul-L3.03" \
#     --output_dir "output/seoul-L3.03/" \
#     --test_list=[] \
#     --gpu_id 1 \
#     --device cuda:1 \
#     --seed 0 \
#     --lr 0.001 \
#     --loss_type "ed_image" \
#     --location "seoul" \
#     --regression "gap" \
#     --pre_train

python main.py \
    --data_dir "data/data_Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 4000 \
    --patience 40 \
    --model_idx "seoul-L3.03" \
    --output_dir "output/seoul-L3.03/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "ed_image" \
    --location "seoul" \
    --regression "gap" \
    --classification


# scripts/seoul_ed.sh

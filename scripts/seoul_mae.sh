# python main.py \
#     --data_dir "data/Seoul" \
#     --image_csv_dir "data/Seoul.csv" \
#     --batch 8 \
#     --epochs 4000 \
#     --patience 40 \
#     --model_idx "seoul-L2.01" \
#     --output_dir "output/seoul-L2.01/" \
#     --test_list=[] \
#     --gpu_id 0 \
#     --device cuda:0 \
#     --seed 0 \
#     --lr 0.001 \
#     --loss_type "mae_image" \
#     --location "seoul" \
#     --regression "gap" \
#     --pre_train


python main.py \
    --data_dir "data/Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 4000 \
    --patience 40 \
    --model_idx "seoul-L2.01" \
    --output_dir "output/seoul-L2.01/" \
    --test_list=[] \
    --gpu_id 0\
    --device cuda:0\
    --seed 0 \
    --lr 0.02 \
    --loss_type "mae_image" \
    --location "seoul" \
    --regression "gap" \
    --do_eval


# scripts/seoul_mae.sh
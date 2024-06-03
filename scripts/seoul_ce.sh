python main.py \
    --data_dir "data/data_seoul_only_v2" \
    --image_csv_dir "data/Seoul_V2.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L4.03" \
    --output_dir "output/seoul-L4.03/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.01 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \

# scripts/seoul_ce.sh
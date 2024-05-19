
python main.py \
    --data_dir "data/data_seoul_only" \
    --image_csv_dir "data/서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --epochs 1000 \
    --patience 50 \
    --batch 8 \
    --model_idx "seoul-V.1.01" \
    --output_dir "output/seoul-V.1.01/" \
    --test_list=[]\
    --gpu_id 1 \
    --device "cuda:1" \
    --seed 0 \
    --wandb

# scripts/seoul.sh
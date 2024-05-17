
python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 100 \
    --model_idx "gangwon-V1" \
    --test_list=[]\
    --gpu_id 0 \
    --seed 9486 \
    --wandb

python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 100 \
    --model_idx "gangwon-V2" \
    --test_list=[]\
    --gpu_id 0 \
    --seed 3449 \
    --wandb

python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 100 \
    --model_idx "gangwon-V3" \
    --test_list=[]\
    --gpu_id 0 \
    --seed 9365 \
    --wandb

# scripts/gangwon.sh
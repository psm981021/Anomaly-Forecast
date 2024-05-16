# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/.csv" \
#     --batch 8 \
#     --test_list []
#     --model_idx "test-projection" \


python main.py \
    --data_dir "data/data_seoul_only" \
    --image_csv_dir "data/서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 4 \
    --epochs 5 \
    --test_list=[]\
    --model_idx "seoul-test" \
    --use_multi_gpu \
    --wandb

#scripts/Version1.sh
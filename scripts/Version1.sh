# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/.csv" \
#     --batch 8 \
#     --test_list []
#     --model_idx "test-projection" \


python main.py \
    --data_dir "data/data_seoul_only" \
    --image_csv_dir "data/seoul_sample.csv" \
    --batch 4 \
    --epochs 5 \
    --test_list=[]\
    --model_idx "seoul-test" \
    --use_multi_gpu
    # --wandb

#scripts/Version1.sh
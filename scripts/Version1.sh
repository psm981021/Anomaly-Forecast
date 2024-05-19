# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/data_sample.csv" \
#     --batch 8 \
#     --test_list []
#     --model_idx "test-projection" \


python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 100 \
    --model_idx "tests-V3" \
    --test_list=[]\
    --gpu_id 0 \
    --seed 9486 \
    
#scripts/Version1.sh
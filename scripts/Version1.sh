# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/data_sample.csv" \
#     --batch 8 \
#     --test_list []
#     --model_idx "test-projection" \


python main.py \
    --data_dir "data/radar_test" \
    --image_csv_dir "data/data_sample_train.csv" \
    --epochs 1000 \
    --patience 50 \
    --batch 8 \
    --model_idx "test_debug" \
    --output_dir "output/test" \
    --test_list=[]\
    --gpu_id 1 \
    --device "cuda:1" \
    --ce_type "ed_image" \
    --grey_scale \
    --seed 0
    
#scripts/Version1.sh

# python main.py --data_dir "data\data_radar" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --test_list=[] --model_idx "test-projection"

python main.py \
    --data_dir "data/radar_test" \
    --image_csv_dir "data/data_sample.csv" \
    --batch 8 \
    --model_idx "test-projection" \
    --test_list []

#--use_multi_gpu \ma


python main.py --data_dir "data/radar_test" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --model_idx "test-projection" --test_list []
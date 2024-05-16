python main.py \
    --data_dir "data/data_seoul_only" \
    --image_csv_dir "data/서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" \
    --batch 8 \
<<<<<<< HEAD
    --model_idx "test-seoul" \
    --test_list []
=======
    --model_idx "test-projection" \
    --test_list []

#--use_multi_gpu \ma


python main.py --data_dir "data/radar_test" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --model_idx "test-projection" --test_list []
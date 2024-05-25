# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/data_sample.csv" \
#     --batch 8 \
#     --test_list []
#     --model_idx "test-projection" \


# python main.py \
#     --data_dir "data/radar_test" \
#     --image_csv_dir "data/data_sample_train.csv" \
#     --epochs 1000 \
#     --patience 50 \
#     --batch 8 \
#     --model_idx "test_debug" \
#     --output_dir "output/test" \
#     --test_list=[]\
#     --gpu_id 1 \
#     --grey_scale \
#     --device "cuda:1" \
#     --loss_type "stamina" \
#     --grey_scale \
#     --seed 0
    

python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/gangwon.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 50 \
    --model_idx "test" \
    --output_dir "output/test/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --loss_type "stamina" \


#scripts/Version1.sh

# python main.py --data_dir "data\data_radar" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --test_list=[] --model_idx "test-projection"

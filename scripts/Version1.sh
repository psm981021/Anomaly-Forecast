python main.py \
    --data_dir "data/data_seoul_only_v2" \
    --image_csv_dir "data/seoul_data_v2.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 40 \
    --model_idx "seoul-V.3.01" \
    --output_dir "output/seoul-V.3.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.0005 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap"\
    --grey_scale




#scripts/Version1.sh

# python main.py --data_dir "data\data_radar" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --test_list=[] --model_idx "test-projection"

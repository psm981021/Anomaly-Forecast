python main.py \
    --data_dir "data/Seoul" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L1.01" \
    --output_dir "output/seoul-L1.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cpu \
    --seed 0 \
    --lr 0.001 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \
    --classification \
    --do_eval

#scripts/Version1.sh

# python main.py --data_dir "data\data_radar" --image_csv_dir "data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv" --batch 8 --test_list=[] --model_idx "test-projection"
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 40 --model_idx "seoul-V.3.01" --output_dir "output/seoul-V.3.01/" --test_list=[] --gpu_id 1 --device cuda:1 --seed 0 --lr 0.0005 --loss_type "stamina" --location "seoul" --regression "gap" --grey_scale
python main.py \
    --data_dir "/workspace/sojeong/Anomaly-Forecast/data/seoul_images_0.55_0.9" \
    --image_csv_dir "/workspace/sojeong/Anomaly-Forecast/data/Seoul_0.55_0.9.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 40 \
    --model_idx "seoul-L3.01" \
    --output_dir "output/seoul-L3.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.01 \
    --loss_type "ed_image" \
    --location "seoul" \
    --regression "gap" \
    --do_eval


# scripts/seoul_ed.sh

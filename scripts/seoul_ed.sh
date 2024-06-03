python main.py \
    --data_dir "data/data_seoul_only_v2" \
    --image_csv_dir "data/seoul_v2.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "seoul-L3.02" \
    --output_dir "output/seoul-L3.02/" \
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

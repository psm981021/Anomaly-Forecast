python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L3.01" \
    --output_dir "output/gangwon-L3.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "ed_image" \
    --location "gangwon" \
    --regression "gap" \
    --pre_train

python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L3.01" \
    --output_dir "output/gangwon-L3.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "ed_image" \
    --location "gangwon" \
    --regression "gap" \


python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L3.01" \
    --output_dir "output/gangwon-L3.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "ed_image" \
    --location "gangwon" \
    --regression "gap" \
    --classification


# scripts/gangwon_ed.sh
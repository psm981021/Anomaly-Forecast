python main.py \
    --data_dir "data/Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L5.01" \
    --output_dir "output/gangwon-L5.01" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --pre_train \
    --balancing 

    python main.py \
    --data_dir "data/Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L5.01" \
    --output_dir "output/gangwon-L5.01" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --balancing \

    python main.py \
    --data_dir "data/Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L5.01" \
    --output_dir "output/gangwon-L5.01" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --balancing \
    --classification

# scripts/gangwon_bal.sh
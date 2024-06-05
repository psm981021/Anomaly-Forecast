python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L4.01" \
    --output_dir "output/gangwon-L4.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --pre_train

python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L4.01" \
    --output_dir "output/gangwon-L4.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \


python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 5000 \
    --patience 40 \
    --model_idx "gangwon-L4.01" \
    --output_dir "output/gangwon-L4.01/" \
    --test_list=[] \
    --gpu_id 0 \
    --device cuda:0 \
    --seed 0 \
    --lr 0.06 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --classification


# scripts/gangwon_ce.sh
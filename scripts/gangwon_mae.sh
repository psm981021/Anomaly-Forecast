python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 4000 \
    --patience 40 \
    --model_idx "gangwon-L2.01" \
    --output_dir "output/gangwon-L2.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "mae_image" \
    --location "gangwon" \
    --regression "gap" \
    --pre_train


python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 4000 \
    --patience 40 \
    --model_idx "gangwon-L2.01" \
    --output_dir "output/gangwon-L2.01/" \
    --test_list=[] \
    --gpu_id 1\
    --device cuda:1\
    --seed 0 \
    --lr 0.02 \
    --loss_type "mae_image" \
    --location "gangwon" \
    --regression "gap" \



# scripts/gangwon_mae.sh
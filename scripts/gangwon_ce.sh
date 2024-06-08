# python main.py \
#     --data_dir "data/data_Gangwon" \
#     --image_csv_dir "data/Gangwon.csv" \
#     --batch 8 \
#     --epochs 3000 \
#     --patience 40 \
#     --model_idx "gangwon-L1.02" \
#     --output_dir "output/gangwon-L1.02/" \
#     --test_list=[] \
#     --gpu_id 1 \
#     --device cuda:1 \
#     --seed 0 \
#     --lr 0.001 \
#     --loss_type "ce_image" \
#     --location "gangwon" \
#     --regression "gap" \
#     --pre_train

python main.py \
    --data_dir "data/data_Gangwon" \
    --image_csv_dir "data/Gangwon.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "gangwon-L1.01" \
    --output_dir "output/gangwon-L1.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.02 \
    --loss_type "ce_image" \
    --location "gangwon" \
    --regression "gap" \
    --classification




# scripts/gangwon_ce.sh
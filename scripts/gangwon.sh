
python main.py \
    --data_dir "data/data_gangwon_only" \
    --image_csv_dir "data/gangwon.csv" \
    --batch 8 \
    --epochs 1000 \
    --patience 50 \
    --model_idx "gangwon-V.4.01" \
    --output_dir "output/gangwon-V.4.01/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.0001 \
    --loss_type "stamina" \
    --location "gangwon" \
    --regression "gap" \
    --grey_scale \
    --do_eval
    #--wandb


# scripts/gangwon.sh


# python main.py --data_dir "data\radar_full" --image_csv_dir "c:\\Users\\PC\\OneDrive\\바탕 화면\\2024-1\\캡스톤\\anomaly\\data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.1.01" --output_dir "output\\gangwon-V.1.01" --test_list=[] --gpu_id 0 --seed 0 --do_eval

# python main.py --data_dir "data\radar_full" --image_csv_dir "c:\\Users\\PC\\OneDrive\\바탕 화면\\2024-1\\캡스톤\\anomaly\\data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "Pretrain-gangwon-V.1.01" --output_dir "output\\Pretrain-gangwon-V.1.01" --test_list=[] --gpu_id 0 --seed 0 --do_eval

# python main.py --data_dir "data\radar_full" --image_csv_dir "c:\\Users\\PC\\OneDrive\\바탕 화면\\2024-1\\캡스톤\\anomaly\\data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "sianet-gangwon-V.1.01" --output_dir "output\\sianet-gangwon-V.1.01" --test_list=[] --gpu_id 0 --seed 0 --sianet

# minjoo computer
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "sianet-gangwon-V.1.01" --output_dir "output\\sianet-gangwon-V.1.01" --test_list=[] --gpu_id 0 --seed 0 --sianet --do_eval
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.3.02" --output_dir "output\\gangwon-V.3.02" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "stamina" 

# regression change
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.3.01" --output_dir "output\\gangwon-V.3.01" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "ed_image"

# inference
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.2.01" --output_dir "output\\gangwon-V.2.01" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "ed_image" --do_eval
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.3.01" --output_dir "output\\gangwon-V.3.01" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "stamina" --do_eval
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\강원_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "gangwon-V.4.01" --output_dir "output\\gangwon-V.4.01" --test_list=[] --gpu_id 0 --seed 0 --loss_type "stamina" --lr 0.0001 --location "gangwon" --pre_train --do_eval
python main.py \
    --data_dir "/dev/shm/data_seoul_only_v2" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "test2" \
    --output_dir "output/test2/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.001 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \
    --pre_train


python main.py \
    --data_dir "/dev/shm/data_seoul_only_v2" \
    --image_csv_dir "data/Seoul.csv" \
    --batch 8 \
    --epochs 3000 \
    --patience 40 \
    --model_idx "test2" \
    --output_dir "output/test2/" \
    --test_list=[] \
    --gpu_id 1 \
    --device cuda:1 \
    --seed 0 \
    --lr 0.02 \
    --loss_type "stamina" \
    --location "seoul" \
    --regression "gap" \


# scripts/seoul_st.sh

#jiwon computer 
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "test" --output_dir "output\\test" --test_list=[] --gpu_id 0 --seed 0 --lr 0.0001 --loss_type "stamina" --location "seoul" --grey_scale --pre_train

# minjoo computer
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "sianet-seoul-V.1.01" --output_dir "output\\sianet-seoul-V.1.01" --test_list=[] --gpu_id 0 --seed 0 --sianet --do_eval 
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "seoul-V.3.01" --output_dir "output\\seoul-V.3.01" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "stamina" --do_eval
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "seoul-V.4.01" --output_dir "output\\seoul-V.4.01" --test_list=[] --gpu_id 0 --seed 0 --grey_scale --loss_type "ed_image" 

# inference
# python main.py --data_dir "data\radar_full" --image_csv_dir "data\\서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv" --batch 8 --epochs 1000 --patience 50 --model_idx "seoul-V.4.01" --output_dir "output\\seoul-V.4.01" --test_list=[] --gpu_id 0 --seed 0 --lr 0.0001 --loss_type "stamina" --location "seoul" --do_eval
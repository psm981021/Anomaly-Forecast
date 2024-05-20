from torch.utils.data import DataLoader
from tqdm import trange
import torch.nn as nn
from datasets import Radar
import argparse
from utils import *
from models import *
from Trainer import *
import time
import wandb

def main():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument("--data_dir", default="data\\radar_test", type=str)
    parser.add_argument("--image_csv_dir", default="data\\22.7_22.9 강수량 평균 0.1 이하 제거.csv", type=str, help="image path, rain intensity, label csv")
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--device",type=str, default="cuda:0")
    parser.add_argument("--seed",type=int, default="42")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--multi_devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument( '--test_list',
        nargs='+',  # This tells argparse to accept one or more arguments for this option
        required=True,  # Make this argument required
        help='A list of values',
    )
    # model args
    parser.add_argument("--model_idx", default="test", type=str, help="model identifier")
    parser.add_argument("--batch", type=int,default=4, help="batch size")

    # train args
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs" )
    parser.add_argument("--log_freq", type=int, default =1, help="number of log frequency" )
    parser.add_argument("--patience",type=int, default="10")

    # learning args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    
    args = parser.parse_args()

    #set seed
    set_seed(args.seed)

    #make path for save
    check_path(args.output_dir)



    # Create instances of Radar class for train, valid, and test datasets

    print("Train Dataset Load")
    train_dataset = Radar(args,csv_path=args.image_csv_dir,flag="Train")

    print("Valid Dataset Load")
    valid_dataset = Radar(args,csv_path=args.image_csv_dir,flag="Valid")
    test_dataset = Radar(args,csv_path=args.image_csv_dir,flag="Test")

    
    
    # Create DataLoader instances for train, valid, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch,drop_last=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devicegpu_id
    print("Using Cuda:", torch.cuda.is_available())

    # if args.use_multi_gpu:
    #     if args.cuda_condition:
    #         args.devices = args.multi_devices.replace(' ', '')
    #         device_ids = args.devices.split(',')
    #         args.device_ids = [int(id_) for id_ in device_ids]
    #         torch.cuda.set_device(args.device_ids[0])
    #         args.gpu = args.device_ids[0]
    #     else:
    #         args.device  = torch.device("cpu")

    if args.use_multi_gpu and torch.cuda.is_available():
        device_ids = list(map(int, args.multi_devices.split(',')))
        args.device_ids = device_ids  # Store device IDs for potential use in DataParallel
        args.device = f"cuda:{device_ids[0]}"  # Set default device to the first GPU
        torch.cuda.set_device(args.device)  # Explicitly set the default device
        print(f"Using multiple GPUs: {device_ids}")
    else:
        print(f"Using single device: {args.device}")
        
    # save model args
    args.str = f"{args.model_idx}-{args.batch}-{args.epochs}"
    args.log_file = os.path.join(args.output_dir,args.str + ".txt" )
    args.dataframe_path = os.path.join(args.output_dir,args.str + ".csv")

    #checkpoint
    checkpoint = args.str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    if os.path.exists(args.checkpoint_path):
        with open(args.log_file, "a") as f:
            f.write("------------------------------ Continue Training ------------------------------ \n")
    
    
    #model
    # n_classes = channel
    model = Fourcaster(n_channels=3,n_classes=3,kernels_per_layer=1, args=args)
    model.to(args.device)

    if args.use_multi_gpu:
        print(args.device_ids)
        model = nn.DataParallel(model, device_ids=args.device_ids)
    

    #trainer
    trainer = FourTrainer(model, train_loader,valid_loader,test_loader, args)

    start_time = time.time()

    # if os.path.exists(args.checkpoint_path):
    #     print("Load pth")
    #     trainer.load(args.checkpoint_path)

    if args.wandb == True:
        wandb.init(project="anomaly_forecast",
                name=f"{args.model_idx}_{args.batch}_{args.epochs}",
                config=args)
        args = wandb.config
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for test!")
        score = trainer.test(args.epochs)
        import IPython; IPython.embed(colors='Linux');exit(1);
        args.test_list.pop(0)
        formatted_data = []
        for record in args.test_list:
            for i in range(len(record[0])):  # Assuming record[0] contains a list of timestamps
                datetime = record[0][i]
                predicted_precipitation = f"{record[1][i].item():.6f}" if record[1].dim() != 0 else f"{record[1].item():.6f}"
                ground_truth = record[2][i].item() if record[2].dim() != 0 else record[2].item()
                formatted_data.append({
                    'datetime': datetime,
                    'predicted precipitation': predicted_precipitation,
                    'ground_truth': ground_truth
        })
        dataframe = pd.DataFrame(formatted_data)
        dataframe.to_csv(args.dataframe_path,index=False)
    else:
        early_stopping = EarlyStopping(args.log_file,args.checkpoint_path, args.patience, verbose=True)

        for epoch in range(args.epochs):
        
            trainer.train(epoch)

            score,_ = trainer.valid(epoch)
            if args.wandb == True:
                wandb.log({"MAE Vailid Loss": score})
            early_stopping(score, trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            

        #test
        print("-----------------Test-----------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        score = trainer.test(args.epochs)

        # save csv file
        try:
            formatted_data = []
            for record in args.test_list:
                for i in range(len(record[0])):  # Assuming record[0] contains a list of timestamps
                    datetime = record[0][i]
                    predicted_precipitation = record[1][i].item() if record[1].dim() != 0 else record[1].item()
                    ground_truth = record[2][i].item() if record[2].dim() != 0 else record[2].item()
                    formatted_data.append({
                        'datetime': datetime,
                        'predicted precipitation': predicted_precipitation,
                        'ground_truth': ground_truth
                    })
            dataframe = pd.DataFrame(formatted_data)
            dataframe.to_csv(args.dataframe_path,index=False)
            import IPython; IPython.embed(colors='Linux');exit(1);
            #dataframe = pd.DataFrame(args.test_list, columns =['datetime', 'predicted precipitation', 'ground_truth'])
        except:
            print("Error Handling csv");
            import IPython; IPython.embed(colors='Linux');exit(1);
            
        
    
        # time check
        end_time = time.time()
        execution_time = end_time - start_time

        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)

        with open(args.log_file, "a") as f:
            f.write(f"To run Epoch:{args.epochs} , It took {hours} hours, {minutes} minutes, {seconds} seconds\n")

if __name__ == "__main__":
    main()

# python main.py --data_dir="data\\radar_test" --image_csv_dir="data\\22.7_22.9 강수량 평균 0.1 이하 제거_set추가.csv"
# python main.py --data_dir="data/radar_test" --image_csv_dir="data/data_sample.csv" --gpu_id=0 --batch=2 --use_multi_gpu --model_idx="test-projection"


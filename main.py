from torch.utils.data import DataLoader
from tqdm import trange
from datasets import Radar
import argparse
from utils import *
from models import *




def main():
    parser = argparse.ArgumentParser()

    # system args
    parser.add_argument("--data_dir", default="data\\radar_test", type=str)
    parser.add_argument("--image_csv_dir", default="data\\22.7_22.9 강수량 평균 0.1 이하 제거.csv", type=str, help="image path, rain intensity, label csv")
    parser.add_argument("--save_path",default="output/", type=str)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--device",type=str, default="cuda:1")
    parser.add_argument("--seed",type=int, default="42")

    # model args
    parser.add_argument("--model_idx", default="test", type=str, help="model identifier")
    parser.add_argument("--batch", type=int, help="batch size")

    # train args
    parser.add_argument("--epochs", type=int, help="number of epochs" )


    # learning args
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    
    
    args = parser.parse_args()

    #set seed
    set_seed(args.seed)

    #make path for save
    check_path(args.save_path)



    # Create instances of Radar class for train, valid, and test datasets

    train_dataset = Radar(args,csv_path=args.image_csv_dir,flag="Train")
    train_dataset.__getitem__(0)
    import IPython; IPython.embed(colors='Linux'); exit(1)
    # valid_dataset = Radar(flag="Valid", csv_path=args.image_csv_dir)
    # test_dataset = Radar(flag="Test", csv_path=args.image_csv_dir)

    
    
    # Create DataLoader instances for train, valid, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=8)
    # valid_loader = DataLoader(valid_dataset, batch_size=8)
    # test_loader = DataLoader(test_dataset, batch_size=8)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    
    # save model args
    args.str = f"{args.model_idx}-{args.batch}-{args.epochs}"
    args.log_file = os.path.join(args.output_dir,args.str + ".txt" )

    #model
    model = Fourcaster(args=args)



if __name__ == "__main__":
    main()
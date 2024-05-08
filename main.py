from torch.utils.data import DataLoader
from tqdm import trange
from datasets import Radar
import argparse
from utils import *





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/radar_test", type=str)
    parser.add_argument("--image_csv_dir", default="data/image_loader.csv", type=str, help="image path, rain intensity, label csv")
    parser.add_argument("--save_path",default="output/", type=str)
    parser.add_argument("--epochs", type=int, help="number of epochs" )
    parser.add_argument("--model_idx", default="test", type=str, help="model idenfier")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, help="batch size")
    parser.add_argument("--device",type=str, default="cuda:1")
    parser.add_argument("--seed",type=int, default="42")
    args = parser.parse_args()

    # Create instances of Radar class for train, valid, and test datasets
    train_dataset = Radar(args,csv_path=args.image_csv_dir,sequence_length=6,flag="Train")
    # valid_dataset = Radar(flag="Valid", csv_path=args.image_csv_dir)
    # test_dataset = Radar(flag="Test", csv_path=args.image_csv_dir)

    
    
    # Create DataLoader instances for train, valid, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=8)
    # import IPython; IPython.embed(colors='Linux');exit(1);
    # valid_loader = DataLoader(valid_dataset, batch_size=1)
    # test_loader = DataLoader(test_dataset, batch_size=1)



    

    #configs
    # args = parser.parse_args()
    # set_seed(args.seed)
    # check_path(args.output_dir)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    # print("Using Cuda:", torch.cuda.is_available())

    

    # main

if __name__ == "__main__":
    main()
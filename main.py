from torch.utils.data import DataLoader
from tqdm import trange
from datasets import Radar
import argparse






def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/radar_test", type=str)
    parser.add_argument("--save_path",default="output/", type=str)
    parser.add_argument("--epochs", type=int, help="number of epochs" )
    parser.add_argument("--model_idx", default="test", type=str, help="model idenfier")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch", type=int, help="batch size")
    parser.add_argument("--device",type=str, default="cuda:1")
    parser.add_argument("--seed",type=int, default="42")



import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() and not args.no_cuda else "cpu")
        torch.cuda.set_device(self.device)

        self.model = model
        self.projection = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),  # 64 input channels, reducing spatial dimensions
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((50, 50))
        )

        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.mae_criterion = torch.nn.L1Loss()
        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch,):
        return self.iteration(epoch, self.valid_dataloader, train=False, test=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False, test=True)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def get_score(self, epoch, pred):

        # pred has to values, cross-entropy loss and mae loss
        ce = pred
        post_fix = {
            "Epoch":epoch,
            "Cross Entropy":"{:.6f}".format(ce),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return  ce, str(post_fix)
    
    
    def save(self, file_name):
        torch.save({
            'epochs': self.args.epochs,
            'model_state_dict': self.model.cpu().state_dict(),
        }, file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    @staticmethod
    def plot_images(image, flag=None):
        
        image = image.cpu().detach().permute(1,2,0).numpy()
        plt.imshow(image)
        if flag == 'R':
            plt.savefig('Real Image')
        else:
            plt.savefig('Generated Image')

class FourTrainer(Trainer):
    def __init__(self,model,train_dataloader, valid_dataloader,test_dataloader, args):
        super(FourTrainer, self).__init__(
            model,train_dataloader, valid_dataloader, test_dataloader,args)
        
    def iteration(self, epoch, dataloader, train=True, test=False):
        
        if train:
            print("Train Fourcaster")            
            self.model.train()
            
            batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
            total_ce, total_mae = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
            
            for i, batch in batch_iter:
                image, label, gap, datetime, class_label = batch

                image_batch = [t.to(self.args.device) for t in image] # 7
                label = label.to(self.args.device) #answer, B
                gap = gap.to(self.args.device) #diff between t-1 t, B
                class_label = class_label.to(self.args.device)
                
                set_ce = 0.0
                precipitation = []       

                for i in range(len(image_batch)-1):
                    
                    # image_batch[i] [B x 3 x R x R]
                    generated_image, regression_logits = self.model(image_batch[i],self.args)
                    
                    # generated_image [B 3 R R ], Regression_logits [B x 1 x 1 x 1]
                    regression_logits = regression_logits.reshape(self.args.batch, -1)
                    precipitation.append(torch.sum(regression_logits, dim=1)) # [B x 1]
                    
                    
                    if self.args.ce_type == 'ce_image':
                        loss_ce =  self.ce_criterion(generated_image.flatten(1), image_batch[i+1].flatten(1))
                    elif self.args.ce_type == 'mse_image':
                        loss_ce = self.mae_criterion(generated_image.flatten(1), image_batch[i+1].flatten(1))
                    else:
                        loss_ce =  self.ce_criterion(generated_image.flatten(1), class_label)
                    
                    set_ce += loss_ce
                
                # set이여서 6으로 나눔
                set_ce /= 6
    
                stack_precipitation = torch.stack(precipitation) # [6 x B]
                predicted_gaps =  stack_precipitation[1:] - stack_precipitation[:-1] # [5 x B]
                total_predict_gap = torch.sum(predicted_gaps, dim=0) # [B]
                
                # Loss_mae
                if self.args.pre_train == False:
                    loss_mae = self.mae_criterion(total_predict_gap, gap)
                            
                # joint Loss
                if self.args.pre_train:
                    joint_loss = set_ce

                else:
                    joint_loss = set_ce + loss_mae
                    total_mae += loss_mae.item()

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                total_ce += set_ce.item()
            # import IPython; IPython.embed(colors='Linux');exit(1);

                # del batch, loss_ce, loss_mae, joint_loss  # After backward pass
                # torch.cuda.empty_cache()
            

            if self.args.wandb == True:
                wandb.log({'Generation Loss (CE)': total_ce / len(batch_iter)}, step=epoch)
                wandb.log({'MAE Train Loss': total_mae / len(batch_iter)}, step=epoch)

            post_fix = {
                "epoch":epoch,
                "CE Loss": "{:.6f}".format(total_ce/len(batch_iter)),
                "MAE Loss":"{:.6f}".format(total_mae/len(batch_iter)),
            }
            if (epoch+1) % self.args.log_freq ==0:
                print(str(post_fix))
            
            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        # end of train


        else:
            #valid and test
            print("Eval Fourcaster")
            self.model.eval()

            with torch.no_grad():
                total_ce = torch.tensor(0.0, device=self.device)
                batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
                for i, batch in batch_iter:
                    image, label, gap, datetime, class_label = batch

                    image_batch = [t.to(self.args.device) for t in image]
                    label = label.to(self.args.device)
                    gap = gap.to(self.args.device)
                    class_label = class_label.to(self.args.device)
                
                    precipitation =[]
                    set_ce =0.0
                    for i in range(len(image_batch)-1):
                    
                        # image_batch[i] [B x 3 x R x R]
                        generated_image, regression_logits = self.model(image_batch[i],self.args)
                        
                        # generated_image [B 3 R R ], Regression_logits [B x 1 x 150 x 150]
                        regression_logits = regression_logits.reshape(self.args.batch, -1)
                        precipitation.append(torch.sum(regression_logits, dim=1)) # [B x 1]
                        
                        #projection_image = self.projection(image_batch[i+1])
                        if self.args.ce_type == 'ce_image':
                            loss_ce =  self.ce_criterion(generated_image.flatten(1), image_batch[i+1].flatten(1))
                        elif self.args.ce_type == 'mse_image':
                            loss_ce = self.mae_criterion(generated_image.flatten(1), image_batch[i+1].flatten(1))
                        else:
                            loss_ce =  self.ce_criterion(generated_image.flatten(1), class_label)

                        set_ce += loss_ce

                    # set이여서 6으로 나눔
                    set_ce /= 6

                    
                    stack_precipitation = torch.stack(precipitation) # [6 x B]
                    predicted_gaps =  stack_precipitation[1:] - stack_precipitation[:-1] # [5 x B]
                    total_predict_gap = torch.sum(predicted_gaps, dim=0) # [B]
                    
                    last_elements = stack_precipitation[-1,:] 

                    # Loss_mae
                    loss_mae = self.mae_criterion(total_predict_gap, gap)

                    total_ce += set_ce.item()

                    if test:
                        self.args.test_list.append([datetime, last_elements, label])
                        
                #     del batch
                # torch.cuda.empty_cache() 
            return self.get_score(epoch, total_ce/len(batch_iter))
            
    



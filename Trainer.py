import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

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
        return self.iteration(epoch, self.valid_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def get_score(self, epoch, pred):

        # pred has to values, cross-entropy loss and mae loss
        mae = pred
        post_fix = {
            "Epoch":epoch,
            "MAE":mae,
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return  mae, str(post_fix)
    
    
    def save(self, file_name):
        torch.save({
            'epochs': self.args.epochs,
            'model_state_dict': self.model.cpu().state_dict(),
        }, file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

class FourTrainer(Trainer):
    def __init__(self,model,train_dataloader, valid_dataloader,test_dataloader, args):
        super(FourTrainer, self).__init__(
            model,train_dataloader, valid_dataloader, test_dataloader,args)
        
    def iteration(self, epoch, dataloader, train=True):
        
        if train:
            print("Train Fourcaster")
            
            self.model.train()
            
            batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
            ce_loss,mae_loss = 0.0, 0.0
            for i, batch in batch_iter:
                image, label, gap = batch

                image_batch = [t.to(self.device) for t in image]
                label = label.to(self.device)
                gap = gap.to(self.device)

                total_ce = 0.0
                precipitation =[]
                for i in range(len(image_batch)-1):
                    
                    generated_image, regression_logits = self.model(image_batch[i],self.args)
                    regression_logits = regression_logits.reshape(self.args.batch, -1)
                    precipitation.append(regression_logits)
                    projection_image = self.projection(image_batch[i+1])
                    
                    loss_ce = self.ce_criterion(generated_image.flatten(1), projection_image.flatten(1))
                    total_ce += loss_ce
        
                total_mae = 0
                for i in range(len(precipitation)-1):
                    # check validity
                    total_mae += torch.sum(precipitation[i+1]-precipitation[i])
                
                # Loss_mae
                loss_mae = self.mae_criterion(total_mae, torch.sum(gap,dim=0))
                # joint Loss
                joint_loss = 0.01 * total_ce + loss_mae

                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                ce_loss += total_ce.item()
                mae_loss += loss_mae.item()

                del batch, loss_ce, loss_mae, joint_loss  # After backward pass
                torch.cuda.empty_cache()

            if self.args.wandb == True:
                wandb.log({'Generation Loss (CE)': ce_loss / len(batch_iter)})
                wandb.log({'MAE Train Loss': mae_loss / len(batch_iter)})

            post_fix = {
                "epoch":epoch,
                "ce_loss": "{:6}".format(ce_loss/len(batch_iter)),
                "mae_loss":"{:6}".format(mae_loss/len(batch_iter))
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

                batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
                for i, batch in batch_iter:
                    image, label, gap = batch

                    image_batch = [t.to(self.device) for t in image]
                    label = label.to(self.device)
                    gap = gap.to(self.device)
                
                    precipitation =[]
                    for i in range(len(image_batch)-1):
                        generated_image, regression_logits = self.model(image_batch[i],self.args)
                        regression_logits = regression_logits.reshape(self.args.batch, -1)
                        precipitation.append(regression_logits)
                    
                    total_mae = 0
                    for i in range(len(precipitation)-1):
                        # check validity
                        total_mae += torch.sum(precipitation[i+1]-precipitation[i])
                    loss_mae = self.mae_criterion(total_mae, torch.sum(gap,dim=0))
                
                    del batch
                torch.cuda.empty_cache() 
            return self.get_score(epoch,loss_mae/len(batch_iter))
            
    



import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from models import RainfallPredictor
from rainnet import *
import numpy as np
from utils import *
from classification_gpu import *
from efficientnet_pytorch import EfficientNet

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = args.device
        # self.device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if self.cuda_condition:
            torch.cuda.set_device(self.args.device)

        self.model = model
        self.projection = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),  # 64 input channels, reducing spatial dimensions
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((50, 50))
        )

        # self.classifier, self.class_criterion, self.class_optimizer = initialize_model(
        #     learning_rate=args.learning_rate,
        #     loss_type="focal",
        #     device=self.device
        # )

        # self.classifier = load_model("cf_models\Seoul_V1_epoch_2.pth", self.device)
        self.classifier = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

        # self.regression_layer = RainfallPredictor().to(self.args.device)
        # self.regression_model = RainNet().to(self.args.device)
        # self.Linear_layer = nn.Linear(self.args.batch, self.args.batch)
        # self.regression_layer =nn.Linear(100, 1)

        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
            self.classifier.cuda()
            #self.regression_model.cuda()
            # self.Linear_layer.cuda()
            # self.regression_layer.cuda()
        
        for param in self.classifier.parameters():
            param.requires_grad = False

        if not self.args.pre_train:
            for param in self.model.parameters():
                param.requires_grad = False

            # Assuming the regression model is meant to be trained
            for param in self.model.regression_layer.parameters():
                param.requires_grad = True
            
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=1e-5, betas=betas, weight_decay=self.args.weight_decay)
        self.reg_optim = Adam(filter(lambda p: p.requires_grad, self.model.regression_layer.parameters()), lr=1e-5, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.mae_criterion = torch.nn.L1Loss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch,):
        return self.iteration(epoch, self.valid_dataloader, train=False, test=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False, test=True)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def correlation_image(self, T,P):
        """
        Element-wise multiplication for tensors
        """
        epsilon = 1e-9
        product = P * T
        numerator = product.sum(dim=0)

        P_squared_sum = (P**2).sum(dim=0)
        T_squared_sum = (T**2).sum(dim=0)

        denominator = torch.sqrt(P_squared_sum * T_squared_sum)

        # Cosine similarity for each pair of images in each set
        cosine_similarity = numerator / (denominator + epsilon)

        return cosine_similarity.mean()
    
    def get_score(self, epoch, pred):

        # pred has to values, cross-entropy loss and mae loss
        ce = pred
        if self.args.pre_train:
            post_fix = {
                "Epoch":epoch,
                "Generation Loss (Eval)":"{:.6f}".format(ce),
            }
        else:
            post_fix = {
                "Epoch":epoch,
                "MAE Loss (Eval)":"{:.6f}".format(ce),
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
        checkpoint = torch.load(file_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

    
    def plot_images(self, image ,epoch, model_idx, datetime, flag=None):
        # image = image.cpu().detach().permute(1,2,0).numpy()
        # image = image.cpu().detach().permute(2,0,1).numpy()
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
            plt.imshow(image)

            model_idx=model_idx.replace('.','-')
            datetime = datetime.replace(':', '-').replace(' ', '_')

            path = os.path.join(self.args.output_dir, str(self.args.pre_train),str(datetime))
            check_path(path)

            if flag == 'R':
                plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch}_Real Image')
            else:
                plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch}_Generated Image')
        else:
            print("Error: Non-tensor input received")

    def plot_images_test(self, image ,epoch, model_idx, i, flag=None):

        
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
            plt.imshow(image)

            model_idx=model_idx.replace('.','-')

            if flag == 'R':
                plt.savefig(f'{self.args.output_dir}/{model_idx}_{i}_{epoch}_Real Image')
            else:
                plt.savefig(f'{self.args.output_dir}/{model_idx}_{i}_{epoch}_Generated Image')
        else:
            print("Error: Non-tensor input received")

class FourTrainer(Trainer):
    def __init__(self,model,train_dataloader, valid_dataloader,test_dataloader, args):
        super(FourTrainer, self).__init__(
            model,train_dataloader, valid_dataloader, test_dataloader,args)
        
    def iteration(self, epoch, dataloader, train=True, test=False):
        
        if train:
            print("Train Fourcaster")            
            self.model.train()
            
            batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
            total_generation_loss, total_mae_loss, total_correlation = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device) , torch.tensor(0.0, device=self.device)
        
            for i, batch in batch_iter:
                # image, label, gap, datetime, class_label = batch
                image, label, gap, datetime = batch

                image_batch = [t.to(self.args.device) for t in image] # 7
                label = label.to(self.args.device) #answer, B
                gap = gap.to(self.args.device) #diff between t-1 t, B
                # #class_label = class_label.to(self.args.device)
                set_generation_loss = 0.0
                correlation_image = 0.0
                total_mae = 0.0
                
                precipitation = []       
                plot_list_seoul = ['2022-07-06 22:00', '2022-07-13 17:00', '2022-07-13 18:00', '2022-07-13 19:00', '2022-09-05 14:00', '2022-07-11 16:00']
                plot_list_gangwon = ['2021-08-01 19:00','2021-01-15 16:00']

                # image batch [B, 7, C, W, H]
                image_batch = torch.stack(image_batch).permute(1,0,2,3,4).contiguous()

                for i in range(len(image_batch)-1):
                    loss_mae = 0.0
                    generation_loss = 0.0
                    # image_batch[i] [B x R x R]
                    
                    generated_image = self.model(image_batch[i],self.args)
                    
                    if self.args.grey_scale:
                        correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1])) #/ self.args.batch
                    else:
                        correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1].mean(dim=1))) #/ self.args.batch
                    
                    if epoch == 0:
                        for j in range(len(datetime)-1):
                            if datetime[j] in plot_list_seoul:
                                
                                self.plot_images(image_batch[-1][j].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')
                    
                    
                    if epoch % 10 == 0 and self.args.pre_train:
                        if self.args.location == "seoul":
                            for j in range(len(datetime)-1):
                                if datetime[j] in plot_list_seoul:
                                    self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')

                        elif self.args.location == "gangwon":
                            if epoch == 0:
                                for j in range(len(datetime)):
                                    if datetime[j] in plot_list_gangwon:
                                        self.plot_images(image_batch[j+1][0].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')
                            elif datetime[j] in plot_list_gangwon:
                                for i in range(len(datetime)):
                                    if datetime[j] in plot_list_gangwon:
                                        self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')


                    # generated_image [B, W, H, C], 
                    # regression_logits = regression_logits.reshape(self.args.batch, -1)
                    precipitation.append(generated_image) # [B x 1]
                    
                    
                    if self.args.loss_type == 'ce_image':
                        
                        generation_loss =  self.ce_criterion(generated_image.mean(dim=-1), image_batch[i+1])

                    elif self.args.loss_type == 'mae_image':

                        generation_loss = self.mae_criterion(generated_image.mean(dim=-1), image_batch[i+1])

                    elif self.args.loss_type == 'ed_image':

                        preds = torch.softmax(generated_image,dim=-1)
                        
                        if self.args.grey_scale == False:
                            label_tensor = torch.arange(100).to(self.device).float()
                            err_r, err_g, err_b = (label_tensor - image_batch[i+1][:,0,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,1,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,2,:,:].unsqueeze(-1)).abs()
                            generation_loss = torch.sum( (preds * err_r) + (preds * err_g) + (preds * err_b) ,dim=-1).mean()

                        else:
                            err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1)).abs()
                            generation_loss = torch.sum((preds * err),dim=-1).mean()

                    elif self.args.loss_type == 'stamina':
                            epsilon = 1e-6
                            
                            if self.args.grey_scale:
                                absolute_error = torch.abs(generated_image - image_batch[i+1].permute(0,2,3,1)) # [B W H C]
                            else:
                                absolute_error = torch.abs(generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1].permute(0,2,3,1)) # [B W H C]

                            event_weight = torch.clamp(image_batch[i] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                            penalty = torch.pow(1 - torch.exp(-absolute_error) + epsilon , 0.5) #  [B W H C]
                            
                            result = absolute_error * event_weight * penalty
                            
                            generation_loss = result.mean()
                            
                    else:
                        generation_loss =  self.ce_criterion(generated_image.flatten(1), class_label)
                    
                    set_generation_loss += generation_loss
                    total_correlation += correlation_image

                # set이여서 6으로 나눔
                # set_generation_loss /= 6
                total_correlation /= 6
               
                
                last_precipitation = precipitation[-1]
                stack_precipitation = torch.stack(precipitation) # [6 , B, 150, 150, 100]
                
                # check
                predicted_gaps =  stack_precipitation[1:] - stack_precipitation[:-1] # [5 ,B, 150, 150, 100]
                total_predict_gap = torch.sum(predicted_gaps, dim=0) # [B, 150, 150, 100] -> [1]
                total_predict_gap=total_predict_gap.permute(0,3,1,2)

                # reg = self.regression_model(image_batch[i+1]).view(self.args.batch) # [B] 
                
                # last_reg = self.regression_model(last_precipitation.permute(0,3,1,2).contiguous()).view(self.args.batch) # [B] 


                if self.args.pre_train == False:
                    
                    # total_predict_gap[:,:,71,86] 관악
                    # total_predict_gap[:,:,58,44] 철원
                    # total_predict_gap[:,:,70:90, 55:85], seoul
                    
                    if self.args.regression == 'gap':

                        crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)

                        import IPython; IPython.embed(colors="Linux");exit(1);
                        predict=inference_jw(self.classifier, crop_predict_gap)

                        reg = abs(self.model.regression_layer(crop_predict_gap)).view(self.args.batch)
                        
                        loss_mae = self.mae_criterion(reg, abs(gap))
                        

                    elif self.args.regression == 'label':
                        last_precipitation = (last_precipitation[:,71,86,:] * 255).clamp(0,255)
                        reg = abs(self.model.regression_layer(last_precipitation)).view(self.args.batch)
                        
                        loss_mae = self.mae_criterion(reg, abs(label))
                    
                    total_mae += loss_mae
            
                #reg = self.regression_model(total_predict_gap).view(self.args.batch) # [B] 
                
                if self.args.pre_train:

                    joint_loss = set_generation_loss #+ loss_mae
                    total_generation_loss += set_generation_loss.item()

                
                elif self.args.pre_train == False: #fine-tuning
                    joint_loss = total_mae
                    total_mae_loss += total_mae.item()

            
                joint_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.args.pre_train:
                    self.optim.step()
                    self.optim.zero_grad()

                elif self.args.pre_train == False: #fine-tuning 
                    self.reg_optim.step()
                    self.reg_optim.zero_grad()

                del batch, generation_loss, loss_mae, joint_loss  # After backward pass
            torch.cuda.empty_cache()
            
            
            if self.args.wandb == True:
                wandb.log({f'Generation Loss {self.args.loss_type} (Train)': total_generation_loss / len(batch_iter)}, step=epoch)
                wandb.log({'Correlation Image (Train)': total_correlation / len(batch_iter)}, step=epoch)
                wandb.log({'MAE Train Loss': total_mae_loss / len(batch_iter)}, step=epoch)

            post_fix = {
                "epoch":epoch,
                f"Geneartion Loss {self.args.loss_type} (Train)": "{:.6f}".format(total_generation_loss/len(batch_iter)),
                "Correlation Image(Train)": "{:.6f}".format(total_correlation/len(batch_iter)),
                "MAE Loss":"{:.6f}".format(total_mae_loss/len(batch_iter)),
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
                total_generation_loss,total_mae_loss = torch.tensor(0.0, device=self.args.device),torch.tensor(0.0, device=self.args.device)
                batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
                for i, batch in batch_iter:

                    # image, label, gap, datetime= batch
                    image, label, gap, datetime = batch # image = [8, 7, 3, 150, 150]
                    
                    image_batch = [t.to(self.args.device) for t in image]
                    label = label.to(self.args.device)
                    gap = gap.to(self.args.device)
                    # class_label = class_label.to(self.args.device)
                
                    precipitation =[]
                    set_generation_loss =0.0
                    correlation_image =0.0
                    total_mae =0.0

                    image_batch = torch.stack(image_batch).permute(1,0,2,3,4).contiguous()

                    for i in range(len(image_batch)-1):
                    
                        # image_batch[i] [B x 3 x R x R]
                        generated_image = self.model(image_batch[i],self.args)
                        # generated_image [B 3 R R ], Regression_logits [B x 1 x 150 x 150]

                        # if self.args.grey_scale:
                        #     correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1])) / self.args.batch
                        # else:
                        #     correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1].mean(dim=1))) / self.args.batch

                        # regression_logits = regression_logits.reshape(self.args.batch, -1)
                        precipitation.append(generated_image) 
                        
                        if self.args.loss_type == 'ce_image':
                            generation_loss =  self.ce_criterion(generated_image.mean(dim=-1), image_batch[i+1])

                        elif self.args.loss_type == 'mae_image':
                            generation_loss = self.mae_criterion(generated_image.mean(dim=-1), image_batch[i+1])

                        elif self.args.loss_type == 'ed_image':
                            preds = torch.softmax(generated_image,dim=-1)

                            if self.args.grey_scale == False:
                                label_tensor = torch.arange(100).to(self.device).float()
                                err_r, err_g, err_b = (label_tensor - image_batch[i+1][:,0,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,1,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,2,:,:].unsqueeze(-1)).abs()
                                generation_loss = torch.sum( (preds * err_r) + (preds * err_g) + (preds * err_b) ,dim=-1).mean()
                            else:
                                err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1)).abs()
                                generation_loss = torch.sum((preds * err),dim=-1).mean()

                        elif self.args.loss_type == 'stamina':
                            
                            epsilon = 1e-6 
                            if self.args.grey_scale:
                                absolute_error = torch.abs(generated_image - image_batch[i+1].permute(0,2,3,1)) # [B W H C]
                            else:
                                absolute_error = torch.abs(generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1].permute(0,2,3,1)) # [B W H C]

                            event_weight = torch.clamp(image_batch[i] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                            penalty = torch.pow(1 - torch.exp(-absolute_error) + epsilon , 0.5) #  [B W H C]
                            
                            result = absolute_error * event_weight * penalty

                            generation_loss = result.mean()
                                
                        else:
                            generation_loss =  self.ce_criterion(generated_image.flatten(1), class_label)
                        
                    set_generation_loss += generation_loss
                    # total_correlation += correlation_image
                    
                    # set이여서 6으로 나눔
                    # set_generation_loss /= 6
                    # correlation_image /= 6

                    # self.plot_images(generated_image[0],self.args.model_idx,datetime[6])
                    last_precipitation = precipitation[-1].permute(0,3,1,2,)
                    stack_precipitation = torch.stack(precipitation) # [6 , B, 150, 150, 100]
                
                    predicted_gaps =  stack_precipitation[1:] - stack_precipitation[:-1] # [5 ,B, 150, 150, 100]
                    total_predict_gap = torch.sum(predicted_gaps, dim=0) # [B, 150, 150, 100] -> [1]
                    
                    total_predict_gap=total_predict_gap.permute(0,3,1,2).contiguous()

                    # reg = self.regression_layer(image_batch[i+1]).view(self.args.batch) # [B] 

                    # reg = self.regression_layer(total_predict_gap).view(self.args.batch) # [B] 

                    # last_reg = self.regression_layer(last_precipitation.permute(0,3,1,2).contiguous()).view(self.args.batch) # [B] 
                    
                       
                    if self.args.pre_train == False:
                        if self.args.regression == 'gap':
                            crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)
                            reg = abs(self.model.regression_layer(crop_predict_gap)).view(self.args.batch)
                            
                            loss_mae = self.mae_criterion(reg, abs(gap))

                        elif self.args.regression == 'label':
                            last_precipitation = (last_precipitation[:,:,71,86] * 255).clamp(0,255)
                            reg = abs(self.model.regression_layer(last_precipitation)).view(self.args.batch)

                            loss_mae = self.mae_criterion(reg, abs(label))
                        total_mae += loss_mae

                    if self.args.pre_train:

                        joint_loss = set_generation_loss #+ loss_mae
                        total_generation_loss += set_generation_loss.item()
                    
                    elif self.args.pre_train == False: #fine-tuning
                        joint_loss = total_mae
                        total_mae_loss += total_mae.item()


                    if test:
                        last_precipitation = (last_precipitation[:,:,71,86] * 255).clamp(0,255)
                        reg = abs(self.regression_layer(last_precipitation)).view(self.args.batch)
                        
                        self.args.test_list.append([datetime, reg, label])
                        
                del batch
                torch.cuda.empty_cache() 

            if self.args.pre_train:
                return self.get_score(epoch, total_generation_loss/len(batch_iter))
            elif self.args.pre_train == False:
                return self.get_score(epoch, total_mae_loss/len(batch_iter))
            
            
    
class SianetTrainer(Trainer):
    def __init__(self,model,train_dataloader, valid_dataloader,test_dataloader, args):
        super(SianetTrainer, self).__init__(
            model,train_dataloader, valid_dataloader, test_dataloader,args)
        
    def iteration(self, epoch, dataloader, train=True, test=False):
        
        if train:
            print("Train Sianet")            
            self.model.train()
            
            batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
            total_l2, total_mae = torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device)
            
            for i, batch in batch_iter:
                # image, label, gap, datetime, class_label = batch 
                image, label, gap, datetime = batch
                
                image_batch = [t.to(self.args.device) for t in image] # 7
                label = label.to(self.args.device) #answer, B
                gap = gap.to(self.args.device) #diff between t-1 t, B
                # class_label = class_label.to(self.args.device)
                
                set_generation_loss = 0.0
                precipitation = []   

                image_batch_tensor=torch.stack(image_batch[:6]) # [6,8,3,150,150]
                target=image_batch[6]

                image_batch_tensor=image_batch_tensor.permute(1,2,0,3,4) # [8,3,6,150,150]
                
                generated_image=self.model(image_batch_tensor)
                
                generated_image=generated_image.expand(-1,3,-1,-1,-1).squeeze(2)


                loss_l2 =  self.l2_criterion(generated_image, target)                

                self.optim.zero_grad()
                loss_l2.backward()
                self.optim.step()

                total_l2 += loss_l2.item()
            

                # del batch, generation_loss, loss_mae, joint_loss  # After backward pass
                # torch.cuda.empty_cache()
            

            if self.args.wandb == True:
                wandb.log({'Generation Loss (CE)': total_l2 / len(batch_iter)}, step=epoch)
                wandb.log({'MAE Train Loss': total_mae / len(batch_iter)}, step=epoch)

            post_fix = {
                "epoch":epoch,
                "Generation Loss (L2)": "{:.6f}".format(total_l2/len(batch_iter)),
                "MAE Loss":"{:.6f}".format(total_mae/len(batch_iter)),
            }
            if (epoch+1) % self.args.log_freq ==0:
                print(str(post_fix))
            
            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        # end of train

### 수정 중 ###
        else:
            #valid and test
            print("Eval Fourcaster")
            self.model.eval()

            with torch.no_grad():
                total_l2 = torch.tensor(0.0, device=self.args.device)
                batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
                for i, batch in batch_iter:
                    image, label, gap, datetime = batch
                    # image, label, gap, datetime, class_label = batch
                    image_batch = [t.to(self.args.device) for t in image]
                    label = label.to(self.args.device)
                    gap = gap.to(self.args.device)
                    # class_label = class_label.to(self.args.device)
                
                    precipitation =[]

                    image_batch_tensor=torch.stack(image_batch[:6]) # [6,8,3,150,150]
                    target=image_batch[6]

                    image_batch_tensor=image_batch_tensor.permute(1,2,0,3,4) # [8,3,6,150,150]

                    generated_image=self.model(image_batch_tensor)
                
                    generated_image=generated_image.expand(-1,3,-1,-1,-1).squeeze(2)

                    # self.plot_images(generated_image[0],self.args.model_idx,datetime[6])
                    
                    loss_l2 =  self.l2_criterion(generated_image, target)                

                    total_l2 += loss_l2.item()




                    # if test:
                    #     self.args.test_list.append([datetime, last_elements, label])
                        
                #     del batch
                # torch.cuda.empty_cache() 
            return self.get_score(epoch, total_l2/len(batch_iter))


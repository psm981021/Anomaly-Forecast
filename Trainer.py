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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        logpt = -nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = -at * (1 - pt) ** self.gamma * logpt
        return F_loss.mean()


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

        # self.classifier = load_model("classification/model/Seoul_b7_epoch_7.pth", self.device, 'b7')
        # self.classifier = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)

        # self.regression_layer = RainfallPredictor().to(self.args.device)
        # self.regression_model = RainNet().to(self.args.device)
        # self.Linear_layer = nn.Linear(self.args.batch, self.args.batch)
        # self.regression_layer =nn.Linear(100, 1)



        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
            # self.classifier.cuda()
            #self.regression_model.cuda()
            # self.Linear_layer.cuda()
            # self.regression_layer.cuda()
        
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

        if not self.args.pre_train:
            for param in self.model.parameters():
                param.requires_grad = False

            # Assuming the regression model is meant to be trained
            for param in self.model.regression_layer.parameters():
                param.requires_grad = True

            for param in self.model.moe.parameters():
                param.requires_grad = True
            
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=1e-5, betas=betas, weight_decay=self.args.weight_decay)
        self.reg_optim = Adam(filter(lambda p: p.requires_grad, self.model.regression_layer.parameters()), lr=1e-5, betas=betas, weight_decay=self.args.weight_decay)
        self.moe_optim = Adam(filter(lambda p: p.requires_grad, self.model.moe.parameters()), lr=1e-5, betas=betas, weight_decay=self.args.weight_decay)
        
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

    def evaluate_model(self, predicted, labels):

        correct = 0
        total = 0
        label_list = []
        pred_list = []

        correct += (predicted == labels).sum().item()
        label_list.extend(labels.cpu().numpy())
        pred_list.extend(predicted.cpu().numpy())

        cm = confusion_matrix(label_list, pred_list, labels=[0, 1, 2])
        accuracy = np.trace(cm) / np.sum(cm)
        precision = precision_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)
        recall = recall_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)
        f1 = f1_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)

        # Average F1 Score
        avg_f1 = np.mean(f1)
    
        # CSI (Critical Success Index) 계산
        csi = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            csi.append(tp / (tp + fn + fp) if (tp + fn + fp) != 0 else 0)
        avg_csi = np.mean(csi)
        

        # POD (Probability of Detection) 계산
        pod = recall  # recall과 동일
        avg_pod = np.mean(pod)

        # FAR (False Alarm Ratio) 계산
        far = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            far.append(fp / (tp + fp) if (tp + fp) != 0 else 0)
        avg_far = np.mean(far)

        return accuracy, avg_f1, avg_csi, avg_pod, avg_far

    
    def plot_images(self, image ,epoch, model_idx, datetime, flag=None, crop =None, test =None):
        # image = image.cpu().detach().permute(1,2,0).numpy()
        # image = image.cpu().detach().permute(2,0,1).numpy()
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
            plt.imshow(image)

            model_idx=model_idx.replace('.','-')
            datetime = datetime.replace('.', '-').replace(' ', '_')

            path = os.path.join(self.args.output_dir, str(self.args.pre_train),str(datetime))
            check_path(path)

            if flag == 'R':

                if crop == 'crop':
                    plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch} crop Real Image')
                else:
                    plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch} Real Image')
            else:
                if crop == 'crop':
                    plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch}  crop Generated Image')
                else:
                    plt.savefig(f'{self.args.output_dir}/{self.args.pre_train}/{datetime}/{model_idx}_{datetime}_{epoch}  Generated Image')
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
            total_generation_loss, total_mae_loss, total_correlation, total_classifier_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device) , torch.tensor(0.0, device=self.device) , torch.tensor(0.0, device=self.device)
        
            for i, batch in batch_iter:
                image, label, gap, datetime, class_label = batch
                # image, label, gap, datetime = batch

                image_batch = [t.to(self.args.device) for t in image] # 7
                label = label.to(self.args.device) #answer, B
                gap = gap.to(self.args.device) #diff between t-1 t, B
                class_label = class_label.to(self.args.device)
                set_generation_loss = 0.0
                correlation_image = 0.0
                total_mae = 0.0
                classifier_loss = 0.0
                
                precipitation = []       
                plot_list_seoul = ['2022-07-06 22:00', '2022-07-13 17:00', '2022-07-13 18:00', '2022-07-13 19:00', '2022-09-05 14:00', '2022-07-11 16:00']
                plot_list_gangwon = ['2021-08-01 19:00','2021-01-15 16:00']

                # image batch [B, 7, C, W, H]
                image_batch = torch.stack(image_batch).permute(1,0,2,3,4).contiguous()

                for i in range(len(image_batch)-1):
                    loss_mae = 0.0
                    generation_loss = 0.0
                    # image_batch[i] [B x R x R]
                    
                    if self.args.balancing:
                        generated_image, crop_generated_image = self.model(image_batch[i],self.args)
                    else:
                        generated_image = self.model(image_batch[i],self.args)
                    
                    if self.args.grey_scale:
                        correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1])) #/ self.args.batch
                    else:
                        correlation_image += torch.abs(self.correlation_image(generated_image.mean(dim=-1), image_batch[i+1].mean(dim=1))) #/ self.args.batch
                    
                    if epoch == 0:
                        if self.args.location == 'seoul':
                            for j in range(len(datetime)):
                                if datetime[j] in plot_list_seoul:
                                    self.plot_images(image_batch[-1][j].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')
                                    self.plot_images(image_batch[-1][j][:,65:95,60:90].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R' ,'crop')
                        
                        elif self.args.location == "gangwon":
                            for j in range(len(datetime)):
                                if datetime[j] in plot_list_gangwon:
                                    self.plot_images(image_batch[-1][j].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')
                                    self.plot_images(image_batch[-1][j][:,30:60,45:75].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R' ,'crop')
                
                    if epoch % 10 == 0 and self.args.pre_train:
                        if self.args.location == "seoul":
                            for j in range(len(datetime)):
                                if datetime[j] in plot_list_seoul:
                                    self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')
                                    if self.args.balancing:
                                        self.plot_images(crop_generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G', 'crop')

                        elif self.args.location == "gangwon":
                            for j in range(len(datetime)):
                                if datetime[j] in plot_list_gangwon:
                                    self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')
                                    if self.args.balancing:
                                        self.plot_images(crop_generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G', 'crop')


                    # generated_image [B, W, H, C], 
                    # regression_logits = regression_logits.reshape(self.args.batch, -1)
                    precipitation.append(generated_image) # [B x 1]
                    
                    
                    if self.args.loss_type == 'ce_image':
                        
                        generation_loss =  self.ce_criterion(generated_image.mean(dim=-1), image_batch[i+1].mean(dim=1))

                    elif self.args.loss_type == 'mae_image':
                        
                        loss_r, loss_g, loss_b = self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,0,:,:].unsqueeze(1)),self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,1,:,:].unsqueeze(1)),self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,2,:,:].unsqueeze(1))
                        generation_loss = (loss_r +  loss_g + loss_b) / 3
                        
                        #generation_loss = self.mae_criterion(generated_image.mean(dim=-1), image_batch[i+1])

                    elif self.args.loss_type == 'ed_image':

                        preds = torch.softmax(generated_image,dim=-1)
                        
                        if self.args.grey_scale == False:
                            label_tensor = torch.arange(100).to(self.device).float()
                            
                            err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1).mean(dim=-1).unsqueeze(-1)).abs()
                            generation_loss = torch.sum((preds * err),dim=-1).mean()
                            #err_r, err_g, err_b = (label_tensor - image_batch[i+1][:,0,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,1,:,:].unsqueeze(-1)).abs(), (label_tensor - image_batch[i+1][:,2,:,:].unsqueeze(-1)).abs()
                            #generation_loss = torch.sum( (preds * err_r) + (preds * err_g) + (preds * err_b) ,dim=-1).mean()

                        else:
                            err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1)).abs()
                            generation_loss = torch.sum((preds * err),dim=-1).mean()

                    elif self.args.loss_type == 'stamina':
                        if self.args.balancing:
                            
                            epsilon = 1e-6

                            absolute_error_full = torch.abs(generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1].permute(0,2,3,1))
                            if self.args.location == 'seoul':
                                absolute_error_crop = torch.abs(crop_generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1][:,:,65:95,60:90].permute(0,2,3,1))
                            elif self.args.location == "gangwon":
                                absolute_error_crop = torch.abs(crop_generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1][:,:,30:60,45:75].permute(0,2,3,1))

                            event_weight_full = torch.clamp(image_batch[i] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                            penalty_full = torch.pow(1 - torch.exp(-absolute_error_full) + epsilon , 0.5) #  [B W H C]

                            if self.args.location == 'seoul':
                                event_weight_crop = torch.clamp(image_batch[i][:,:,65:95,60:90] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                            elif self.args.location == "gangwon":
                                event_weight_crop = torch.clamp(image_batch[i][:,:,30:60,45:75] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                            penalty_crop = torch.pow(1 - torch.exp(-absolute_error_crop) + epsilon , 0.5) #  [B W H C]

                            result_full = absolute_error_full * event_weight_full * penalty_full
                            result_crop = absolute_error_crop * event_weight_crop * penalty_crop

                            generation_loss = result_full.mean() * 0.4 + result_crop.mean() * 0.6

                        else:
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

                    # total_predict_gap[:,:,70:90, 55:86], seoul
                    # total_predict_gap[:,:,30:60, 45:75], gangwon
                    
                    if self.args.regression == 'gap':
                        if self.args.classifier:
                            if self.args.location == "seoul":
                                # [B C ]
                                crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)
                            else: # gangwon
                                crop_predict_gap = (total_predict_gap[:,:,58,44] * 255).clamp(0,255)


                            logits = self.model.classifier(crop_predict_gap) # [B label]
                            logits = logits.float()
                            
                            classifier_loss += self.ce_criterion(logits, class_label)

                            logits = torch.argmax(F.softmax(logits, dim=-1),dim=-1)
                            if 2 in logits:
                                index_of_two = torch.where(logits == 2)[0]
                                for i in index_of_two:
                                    print(datetime[i])
                            reg = torch.zeros(self.args.batch).to(self.args.device)

                            
                            for i, model_index in enumerate(logits):
                                selected_model = self.model.moe[model_index]  
                                if self.args.location == 'seoul':
                                    reg[i] = abs(selected_model(total_predict_gap[:,:,71,86][i]))
                                elif self.args.location == "gangwon":
                                    reg[i] = abs(selected_model(total_predict_gap[:,:,58,44][i])) 

                            loss_mae = self.mae_criterion(abs(reg), abs(gap))

                        
                        if self.args.classification:
                            if self.args.location == "seoul":
                                # [B 100 2 2 ]
                                crop_predict_gap = (total_predict_gap[:,:,70:72,85:87] * 255).clamp(0,255)
                            else: # gangwon
                                crop_predict_gap = (total_predict_gap[:,:,57:59,43:45] * 255).clamp(0,255)


                            reg = torch.zeros(self.args.batch).to(self.args.device)

                            # import IPython; IPython.embed(colors='Linux'); exit(1)
                            for i, model_index in enumerate(class_label): # 라벨값을 직접 주기
                                selected_model = self.model.moe[model_index]  # Select model based on prediction
                                if self.args.location == 'seoul':
                                    reg[i] = abs(selected_model(abs(total_predict_gap[:,:,71,86][i])))
                                elif self.args.location == "gangwon":
                                    reg[i] = abs(selected_model(abs(total_predict_gap[:,:,58,44][i]))) 

                            loss_mae = self.mae_criterion(abs(reg), abs(gap))


                        else:
                            if self.args.location == 'seoul':
                                crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)
                            elif self.args.location == "gangwon":
                                crop_predict_gap = (total_predict_gap[:,:,58,44] * 255).clamp(0,255)
                            reg = abs(self.model.regression_layer(abs(crop_predict_gap))).view(self.args.batch)
                            # import IPython; IPython.embed(colors='Linux'); exit(1)
                            loss_mae = self.mae_criterion(reg, abs(gap))
                        

                    elif self.args.regression == 'label':
                        if self.args.location == 'seoul':
                            last_precipitation = (last_precipitation[:,71,86,:] * 255).clamp(0,255)
                        elif self.args.location == "gangwon":
                            last_precipitation = (last_precipitation[:,58,44,:] * 255).clamp(0,255)

                        reg = abs(self.model.regression_layer(abs(last_precipitation))).view(self.args.batch)
                        loss_mae = self.mae_criterion(reg, abs(label))
                    
                    total_mae += loss_mae
            
                #reg = self.regression_model(total_predict_gap).view(self.args.batch) # [B] 
                
                if self.args.pre_train:

                    joint_loss = set_generation_loss #+ loss_mae
                    total_generation_loss += set_generation_loss.item()

                
                elif self.args.pre_train == False: #fine-tuning
                    joint_loss = total_mae + classifier_loss
                    if self.args.classifier:
                        total_classifier_loss += classifier_loss.item()
                    total_mae_loss += total_mae.item()

            
                joint_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.args.pre_train:
                    self.optim.step()
                    self.optim.zero_grad()

                elif self.args.pre_train == False: #fine-tuning 
                    if self.args.classification: # MoE 
                        self.moe_optim.step()
                        self.moe_optim.zero_grad()
                    else: # Regression Model
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
                "Classifier Loss":"{:.6f}".format(total_classifier_loss/len(batch_iter)),
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

            correct = 0
            total = 0
            label_list = []
            pred_list = []

            with torch.no_grad():
                total_generation_loss,total_mae_loss,total_classifier_loss = torch.tensor(0.0, device=self.args.device), torch.tensor(0.0, device=self.args.device),torch.tensor(0.0, device=self.args.device)
                batch_iter = tqdm(enumerate(dataloader), total= len(dataloader))
                for i, batch in batch_iter:

                    # image, label, gap, datetime= batch
                    image, label, gap, datetime, class_label = batch # image = [8, 7, 3, 150, 150]
                    
                    image_batch = [t.to(self.args.device) for t in image]
                    label = label.to(self.args.device)
                    gap = gap.to(self.args.device)
                    class_label = class_label.to(self.args.device)
                
                    precipitation =[]
                    set_generation_loss =0.0
                    correlation_image =0.0
                    total_mae =0.0
                    classifier_loss = 0.0

                    image_batch = torch.stack(image_batch).permute(1,0,2,3,4).contiguous()
                    
                                    # ['2022-08-08 13:00', '2022-08-11 05:00', '2023-07-04 20:00',
                                    #   '2023-07-14 01:00','2023-07-18 07:00','2023-08-29 13:00','2023-09-17 00:00',
                                    #   '2021-07-03 18:00', '2021-07-03 21:00', '2022-06-23 20:00', '2022-06-23 19:00',
                                    #   '2022-06-23 22:00', '2022-06-24 01:00', '2022-06-30 03:00', '2022-06-30 05:00',
                                    #   '2022-06-30 06:00', '2022-06-30 08:00', '2022-06-30 09:00', '2022-06-30 10:00',
                                    #   '2022-06-30 18:00', '2022-06-30 19:00', '2022-07-13 11:00', '2022-07-13 16:00',
                                    #   '2022-07-13 17:00', '2022-08-08 21:00', '2022-08-08 22:00', '2022-08-08 23:00',
                                    #   '2022-08-09 00:00', '2022-08-09 03:00']
                    
                    test_datetime_seoul = ['2021.7.3 18:00','2021.7.3 21:00'
                                           '2022.6.23 19:00', '2022.6.23 22:00','2022.6.24 1:00' '2022.6.30 05:00','2022.7.13 11:00','2022.7.13 12:00',
                                           '2022.7.13 16:00', '2022.8.8 23:00','2022.8.8 21:00','2022.9.5 14:00','2022.9.5 18:00'
                                             '2023.11.6 04:00','2023.11.6 4:00','2023.4.5 13:00','2023.7.11 16:00', '2023.8.10 18:00']
                    test_datetime_gangwon = []
                    
                    for i in range(len(image_batch)-1):
                    
                        # image_batch[i] [B x 3 x R x R]
                        if self.args.balancing:
                            generated_image, crop_generated_image = self.model(image_batch[i],self.args)
                        else:
                            generated_image = self.model(image_batch[i],self.args)
                        # generated_image [B 3 R R ], Regression_logits [B x 1 x 150 x 150]

                        precipitation.append(generated_image) 
                        if self.args.location == "seoul" and self.args.do_eval:
                            
                            for j in range(len(datetime)):
                                if datetime[j] in test_datetime_seoul:
        
                                    self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')
                                    self.plot_images(image_batch[-1][j].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')

                        elif self.args.location == "gangwon" and self.args.do_eval:
                            for j in range(len(datetime)):
                                if datetime[j] in test_datetime_gangwon:
                                    self.plot_images(generated_image[j].mean(dim=-1),epoch, self.args.model_idx, datetime[j], 'G')
                                    self.plot_images(image_batch[-1][j].permute(1,2,0),epoch, self.args.model_idx, datetime[j], 'R')


                        if self.args.loss_type == 'ce_image':
                            generation_loss =  self.ce_criterion(generated_image.mean(dim=-1), image_batch[i+1].mean(dim=1))

                        elif self.args.loss_type == 'mae_image':
                            # import IPython; IPython.embed(colors='Linux'); exit(1)
                            loss_r, loss_g, loss_b = self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,0,:,:].unsqueeze(1)),self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,1,:,:].unsqueeze(1)),self.mae_criterion(generated_image.permute(0,3,1,2),image_batch[i+1][:,2,:,:].unsqueeze(1))
                            generation_loss = (loss_r +  loss_g + loss_b) / 3

                        elif self.args.loss_type == 'ed_image':
                            preds = torch.softmax(generated_image,dim=-1)

                            if self.args.grey_scale == False:
                                label_tensor = torch.arange(100).to(self.device).float()
                                
                                err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1).mean(dim=-1).unsqueeze(-1)).abs()
                                generation_loss = torch.sum((preds * err),dim=-1).mean()
                            
                            else:
                                err = (torch.arange(100).to(self.device).float() - image_batch[i+1].permute(0,2,3,1)).abs()
                                generation_loss = torch.sum((preds * err),dim=-1).mean()

                        elif self.args.loss_type == 'stamina':
                            
                            if self.args.balancing:
                                
                                epsilon = 1e-6

                                absolute_error_full = torch.abs(generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1].permute(0,2,3,1))
                                if self.args.location == 'seoul':
                                    absolute_error_crop = torch.abs(crop_generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1][:,:,65:95,60:90].permute(0,2,3,1))
                                elif self.args.location == "gangwon":
                                    absolute_error_crop = torch.abs(crop_generated_image.mean(dim=-1).unsqueeze(dim=-1) - image_batch[i+1][:,:,30:60,45:75].permute(0,2,3,1))

                                event_weight_full = torch.clamp(image_batch[i] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                                penalty_full = torch.pow(1 - torch.exp(-absolute_error_full) + epsilon , 0.5) #  [B W H C]

                                if self.args.location == 'seoul':
                                    event_weight_crop = torch.clamp(image_batch[i][:,:,65:95,60:90] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                                elif self.args.location == "gangwon":
                                    event_weight_crop = torch.clamp(image_batch[i][:,:,30:60,45:75] + 1, max=6).permute(0,2,3,1) # [B W H 1]
                                    
                                penalty_crop = torch.pow(1 - torch.exp(-absolute_error_crop) + epsilon , 0.5) #  [B W H C]

                                result_full = absolute_error_full * event_weight_full * penalty_full
                                result_crop = absolute_error_crop * event_weight_crop * penalty_crop

                                generation_loss = result_full.mean() * 0.4 + result_crop.mean() * 0.6

                            else:
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

                            if self.args.classifier:
                                if self.args.location == "seoul":
                                    # [B 100 2 2 ]
                                    crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)
                                else: # gangwon
                                    crop_predict_gap = (total_predict_gap[:,:,58,44] * 255).clamp(0,255)


                                logits = self.model.classifier(crop_predict_gap)
                                logits = logits.float()
                                classifier_loss += self.ce_criterion(logits, class_label)

                                logits = torch.argmax(F.softmax(logits, dim=-1),dim=-1)
                                reg = torch.zeros(self.args.batch).to(self.args.device)

                                correct += (logits == class_label).sum().item()
                                label_list.extend(class_label.cpu().numpy())
                                pred_list.extend(logits.cpu().numpy())

                                cm = confusion_matrix(label_list, pred_list, labels=[0, 1, 2])
                                accuracy = np.trace(cm) / np.sum(cm)
                                precision = precision_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)
                                recall = recall_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)
                                f1 = f1_score(label_list, pred_list, average=None, labels=[0, 1, 2], zero_division=0)

                                # Average F1 Score
                                avg_f1 = np.mean(f1)

                    
                                csi = []
                                for i in range(len(cm)):
                                    tp = cm[i, i]
                                    fn = np.sum(cm[i, :]) - tp
                                    fp = np.sum(cm[:, i]) - tp
                                    csi.append(tp / (tp + fn + fp) if (tp + fn + fp) != 0 else 0)
                                avg_csi = np.mean(csi)

                                
                                # POD (Probability of Detection) 계산
                                pod = recall  # recall과 동일
                                avg_pod = np.mean(pod)


                                # FAR (False Alarm Ratio) 계산
                                far = []
                                for i in range(len(cm)):
                                    tp = cm[i, i]
                                    fp = np.sum(cm[:, i]) - tp
                                    far.append(fp / (tp + fp) if (tp + fp) != 0 else 0)
                                avg_far = np.mean(far)




                                for i, model_index in enumerate(logits):
                                    selected_model = self.model.moe[model_index]  
                                    if self.args.location == 'seoul':
                                        reg[i] = abs(selected_model(total_predict_gap[:,:,71,86][i]))
                                    elif self.args.location == "gangwon":
                                        reg[i] = abs(selected_model(total_predict_gap[:,:,58,44][i])) 

                                loss_mae = self.mae_criterion(abs(reg), abs(gap))
                            
                            if self.args.classification:
                                if self.args.location == "seoul":
                                    # [B 100 2 2 ]
                                    crop_predict_gap = (total_predict_gap[:,:,70:72,85:87] * 255).clamp(0,255)
                                else: # gangwon
                                    crop_predict_gap = (total_predict_gap[:,:,57:59,43:45] * 255).clamp(0,255)

                                reg = torch.zeros(self.args.batch).to(self.args.device)
                                # for i, model_index in enumerate(predict):
                                for i, model_index in enumerate(class_label):
                                    selected_model = self.model.moe[model_index]  # Select model based on prediction

                                    if self.args.location == 'seoul':
                                        reg[i] = abs(selected_model(abs(total_predict_gap[:,:,71,86][i])))
                                        for j in range(len(datetime)):
                                            if '2022.6.23 19:00' in datetime:
                                                import IPython; IPython.embed(colors='Linux');exit(1);
                                    elif self.args.location == "gangwon":
                                        reg[i] = abs(selected_model(abs(total_predict_gap[:,:,58,44][i]))) 

                                loss_mae = self.mae_criterion(abs(reg), abs(gap))


                            else:
                                if self.args.location == 'seoul':
                                    crop_predict_gap = (total_predict_gap[:,:,71,86] * 255).clamp(0,255)
                                elif self.args.location == "gangwon":
                                    crop_predict_gap = (total_predict_gap[:,:,58,44] * 255).clamp(0,255)
                                reg = abs(self.model.regression_layer(abs(crop_predict_gap))).view(self.args.batch)
                                loss_mae = self.mae_criterion(reg, abs(gap))
                        

                        elif self.args.regression == 'label':

                            if self.args.location == 'seoul':
                                last_precipitation = (last_precipitation[:,71,86,:] * 255).clamp(0,255)
                            elif self.args.location == "gangwon":
                                last_precipitation = (last_precipitation[:,58,44,:] * 255).clamp(0,255)

                            reg = abs(self.model.regression_layer(abs(last_precipitation))).view(self.args.batch)

                            loss_mae = self.mae_criterion(reg, abs(label))
                        total_mae += loss_mae

                    if self.args.pre_train:

                        joint_loss = set_generation_loss #+ loss_mae
                        total_generation_loss += set_generation_loss.item()
                    
                    elif self.args.pre_train == False: #fine-tuning
                        joint_loss = total_mae + classifier_loss
                        if self.args.classifier:
                            total_classifier_loss += classifier_loss.item()
                        total_mae_loss += total_mae.item()


                    if test:
                        if self.args.classification:
                            if self.args.location == "seoul":
                                crop_predict_gap = (last_precipitation[:,:,70:72,85:87] * 255).clamp(0,255)
                            else:
                                crop_predict_gap = (last_precipitation[:,:,57:59,43:45] * 255).clamp(0,255)

                            reg = torch.zeros(self.args.batch).to(self.args.device)
                            # for i, model_index in enumerate(predict):
                            
                            for i, model_index in enumerate(class_label):
                                selected_model = self.model.moe[model_index]  # Select model based on prediction
                                if self.args.location == 'seoul':
                                    reg[i] = abs(selected_model(abs(last_precipitation[:,:,71,86][i])))
                                else:
                                    reg[i] = abs(selected_model(abs(last_precipitation[:,:,58,44][i])))

                            # self.args.test_list.append([datetime, reg, label, predict])
                            self.args.test_list.append([datetime, reg, label, class_label])

                        elif self.args.classifier:
                            if self.args.location == "seoul":
                                crop_predict_gap = (last_precipitation[:,:,70:72,85:87] * 255).clamp(0,255)
                            else:
                                crop_predict_gap = (last_precipitation[:,:,57:59,43:45] * 255).clamp(0,255)

                            logits = self.model.classifier(crop_predict_gap)
                            logits = logits.float()
                            classifier_loss += self.ce_criterion(logits, class_label)

                            logits = torch.argmax(F.softmax(logits, dim=-1),dim=-1)
                            reg = torch.zeros(self.args.batch).to(self.args.device)

                            for i, model_index in enumerate(logits):
                                selected_model = self.model.moe[model_index]  
                                if self.args.location == 'seoul':
                                    reg[i] = abs(selected_model(total_predict_gap[:,:,71,86][i]))
                                elif self.args.location == "gangwon":
                                    reg[i] = abs(selected_model(total_predict_gap[:,:,58,44][i])) 
                            self.args.test_list.append([datetime, reg, label, logits])


                        else:
                            if self.args.location == "seoul":
                                last_precipitation = (last_precipitation[:,:,71,86] * 255).clamp(0,255)
                            else:
                                last_precipitation = (last_precipitation[:,:,58,44] * 255).clamp(0,255)
                            
                            reg = abs(self.model.regression_layer(abs(last_precipitation))).view(self.args.batch)
                            self.args.test_list.append([datetime, reg, label])
                        
                        
                del batch
                torch.cuda.empty_cache() 

            if self.args.pre_train:
                return self.get_score(epoch, total_generation_loss/len(batch_iter))
            elif self.args.pre_train == False:
                return self.get_score(epoch, total_mae_loss/len(batch_iter))
            
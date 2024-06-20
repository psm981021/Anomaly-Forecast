   
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


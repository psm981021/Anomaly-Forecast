import torch
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model


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
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def get_score(self, epoch, pred):

        # pred has to values, cross-entropy loss and mae loss
        ce, mae = pred
        post_fix = {
            "Epoch":epoch,
            "MAE":mae,
            "CE-Loss": ce
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [ce, mae], str(post_fix)
    
    def save(self, file_name):
        torch.save({
            'epochs': self.args.epochs,
            'model_state_dict': self.model.cpu().state_dict(),
        }, file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

class FourTrainer(Trainer):
    def __init__(self,model,train_dataloader, valid_dataloader, test_dataloader, args):
        super(FourTrainer, self).__init__(
            model,train_dataloader, valid_dataloader, test_dataloader)
        
    def iteration(self, epoch, train_dataloader, train=True):
        
        if train:
            
            # model eval
            self.model.eval()
            
            batch_iter = tqdm(enumerate(train_dataloader))
            for i, image ,label, gap in batch_iter:
 
                image = image.to(self.device)
                label = label.to(self.device)
                gap = gap.to(self.device)

                # model output
                generated_image,classification_logits, regression_logits = self.model(image,self.args)


                # cross entropy loss (Real image vs Fake image)
                # Loss_ce
                loss_ce = self.ce_criterion(generated_image, image[-1])
                
                # Regression with labels (CNN-regression)
                # Loss_mae
                loss_mae = self.mae_criterion(regression_logits, label)

                #Total Loss = Loss_ce + Loss_mae

                #Total Loss backprop



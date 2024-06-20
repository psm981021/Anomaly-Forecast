import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# Focal Loss
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

# EVL Loss
class EVLLoss(nn.Module):
    def __init__(self, beta_1=0.1, beta_0=0.9, gamma=2.0):
        super(EVLLoss, self).__init__()
        self.beta_1 = beta_1
        self.beta_0 = beta_0
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.clamp(y_pred, min=1e-15, max=1-1e-15)
        y_true = torch.eye(y_pred.size(1))[y_true].to(y_pred.device)
        loss = - self.beta_1 * (1 - y_pred / self.gamma) ** self.gamma * y_true * torch.log(y_pred) \
               - self.beta_0 * (1 - (1 - y_pred) / self.gamma) ** self.gamma * (1 - y_true) * torch.log(1 - y_pred)
        return loss.mean()

def make_image_dataframe(csv_file):
    data = pd.read_csv(csv_file)
    idx_list = []
    label_list = []

    for i in range(len(data)):
        if data.loc[i]['Class_label'] == 2:
            for j in range(4, 8):
                idx_list.append(data.iloc[i][j])
                label_list.append(data.loc[i]['Class_label'])
        else:
            idx_list.append(data.loc[i]['t'])
            label_list.append(data.loc[i]['Class_label'])
                
    df = pd.DataFrame({"Timestamp": idx_list, "Class_label": label_list})
    df.drop_duplicates(keep='first', inplace=True)
    
    return df

# 이미지 데이터셋 클래스
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = make_image_dataframe(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Timestamp']
        img_path = f"{self.img_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['Class_label']
        return image, label

# 데이터 로드
def load_data(csv_file, img_dir, batch_size, transform, test_size=0.1, valid_size=0.1):
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    test_split = int(np.floor(test_size * dataset_size))
    valid_split = int(np.floor(valid_size * dataset_size))
    test_indices = indices[:test_split]
    valid_indices = indices[test_split:test_split + valid_split]
    train_indices = indices[test_split + valid_split:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# 모델 초기화
def initialize_model(learning_rate, loss_type, device, mode):
    name = 'efficientnet-' + mode
    model = EfficientNet.from_pretrained(name, num_classes=3) #분류 라벨 : 0, 1, 2
    
    # 첫 번째 레이어 수정 (3 채널을 받을 수 있도록) - RGB로 돌릴려고
    model._conv_stem = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
    
    if loss_type == 'focal':
        criterion = FocalLoss()
    else:
        criterion = EVLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    criterion.to(device)
    
    return model, criterion, optimizer

def setup_logger(log_path): #로그 기록용 함수
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 파일 핸들러
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# 모델 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, device, logger, mode):
    global best_model_path
    
    best_loss = 10e9
    best_epoch = 0
    patience = 0
    
    for epoch in range(num_epochs): #한 에폭당 거의 2분 걸림
        model.train() #train
        train_loss = 0.0
        #train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}")
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}")

        model.eval() #valid
        valid_loss = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            
                loss = criterion(outputs, labels)
            
                valid_loss += loss.item()
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Valid Loss: {valid_loss / len(val_loader)}")
                
        if valid_loss < best_loss: #renew best model
            best_loss = valid_loss
            best_epoch = epoch
            best_model_path = save_path + f'_{mode}' + '_epoch_' + str(epoch + 1) + '.pth'
            
            logger.info("New Best Valid Loss!")
            save_model(model, best_model_path)
            evaluate_model(model, val_loader, device, logger)
            patience = 0
        else:
            patience += 1
        
        if patience >= 30: #early stopping
            logger.info(f"Early Stopped at epoch {epoch + 1}")
            logger.info(f"Best Loss : {best_loss / len(val_loader)}")
            logger.info(f"Best epoch : {best_epoch}")
            break

# 모델 평가
def evaluate_model(model, data_loader, device, logger, predict = False):
    model.eval()
    correct = 0
    total = 0
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(predicted.cpu().numpy())

    if predict:
        logger.info("--" * 10 + "Test Stage" + "--" * 10)

    # Confusion Matrix를 계산합니다.
    cm = confusion_matrix(label_list, pred_list, labels=[0, 1, 2])
    logger.info("--" * 30)
    logger.info(f"Confusion Matrix: \n{cm}")

    # Accuracy 계산
    accuracy = np.trace(cm) / np.sum(cm)

    # Precision, Recall, F1 Score 계산
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
    
    logger.info(f"Accuracy: {accuracy * 100}%")
    logger.info(f"F1 Score: {avg_f1}")
    logger.info(f"CSI: {avg_csi}")
    logger.info(f"POD: {avg_pod}")
    logger.info(f"FAR: {avg_far}")
    logger.info("--" * 30)

    return accuracy, avg_f1, avg_csi, avg_pod, avg_far

# 모델 저장
def save_model(model, path):
    # 디렉토리가 없으면 생성
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 모델 저장
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
# 모델 로드
def load_model(filepath, device, mode):
    name = 'efficientnet-' + mode
    model = EfficientNet.from_pretrained(name, num_classes=3) #분류 라벨 : 0, 1, 2
    
    # 첫 번째 레이어 수정 (3 채널을 받을 수 있도록)
    model._conv_stem = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

# 이미지 추론
def inference(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

def inference_jw(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    

if __name__ == "__main__":
    csv_dict = {'서울' : 'Seoul', '강원' : 'Gangwon'}
    version = '서울' #data version
    mode = 'b7' #efficientnet version
    csv_file = '/workspace/chanbeen/Anomaly-Forecast/data/' + csv_dict[version] + '.csv' 
    img_dir = '/workspace/chanbeen/Anomaly-Forecast/data/images_classification' 
    log_path = '/workspace/chanbeen/Anomaly-Forecast/classification/log/' + mode + '.log'
    
    
    batch_size = 32
    learning_rate = 0.003
    num_epochs = 500 #500
    loss_type = 'focal'  # 'focal' 또는 'evl'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    
    #import IPython; IPython.embed(colors='Linux');exit(1);
    
    # 서울 crop 150x150
    left = 240  
    top = 120   
    right = 390
    bottom = 270
    
    # 이미지 자를 좌표 - 강원도 150x150
    # left = 300
    # top = 110  
    # right = 450
    # bottom = 260
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.crop((left, top, right, bottom))), # crop
        #transforms.Grayscale(num_output_channels=1), # grayscale
        transforms.ToTensor()
    ])

    logger = setup_logger(log_path)

    train_loader, val_loader, test_loader = load_data(csv_file, img_dir, batch_size, transform)
    model, criterion, optimizer = initialize_model(learning_rate, loss_type, device, mode = mode)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, f'classification/model/{csv_dict[version]}', device, logger, mode)

    loaded_model = load_model(best_model_path, device, mode = mode) #수정 필요
    evaluate_model(loaded_model, test_loader, device, logger = logger, predict=True)

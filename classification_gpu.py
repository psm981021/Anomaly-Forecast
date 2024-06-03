import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from efficientnet_pytorch import EfficientNet

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
        if data.loc[i]['Class_label'] == 0:
            idx_list.append(data.loc[i]['t'])
            label_list.append(data.loc[i]['Class_label'])
        else:
            for j in range(2, 8):
                idx_list.append(data.iloc[i][j])
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
        image = Image.open(img_path).convert("L")
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
def initialize_model(learning_rate, loss_type, device):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    
    # 첫 번째 레이어 수정 (1 채널을 받을 수 있도록) - grayscale로 돌릴려고
    #model._conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    
    if loss_type == 'focal':
        criterion = FocalLoss()
    else:
        criterion = EVLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    criterion.to(device)
    
    return model, criterion, optimizer

# 모델 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path, device):
    best_loss = 10e9
    best_epoch = 0
    patience = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = epoch
            print("New Best Loss!")
            save_model(model, f"{save_path}_epoch_{epoch + 1}.pth")
            print("--" * 30)
            patience = 0
        else:
            patience += 1
        
        if patience >= 20: #early stopping
            print(f"Early Stopped at epoch {epoch + 1}")
            print("Best Loss : ", best_loss)
            print("Best epoch :", best_epoch)
            break
        
        # 10 epoch마다 모델 검증
        if (epoch + 1) % 10 == 0:
            print(f"Check validation at every 10 epochs : {epoch + 1}")
            evaluate_model(model, val_loader, device)
            save_model(model, f"{save_path}_epoch_{epoch + 1}.pth")
            print("--" * 30)

# 모델 평가
def evaluate_model(model, data_loader, device):
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

    accuracy = accuracy_score(label_list, pred_list) * 100
    f1 = f1_score(label_list, pred_list, average='weighted')
    
    tn, fp, fn, tp = confusion_matrix(label_list, pred_list).ravel()
    
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    # TN FP
    # FN TP
    
    print(f"Accuracy: {accuracy}%")
    print(f"F1 Score: {f1}")
    print(f"CSI: {csi}")
    print(f"POD: {pod}")
    print(f"FAR: {far}")
    print(f"Confusion Matrix: \n{confusion_matrix(label_list, pred_list)}")

    return accuracy, f1, csi, pod, far

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
def load_model(filepath, device):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    
    # 첫 번째 레이어 수정 (1 채널을 받을 수 있도록)
    #model._conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

# 모델 테스트
def predict(model, data_loader, device):
    model.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(label_list, pred_list) * 100
    f1 = f1_score(label_list, pred_list, average='weighted')

    tn, fp, fn, tp = confusion_matrix(label_list, pred_list).ravel()
    
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Test Accuracy: {accuracy}%")
    print(f"Test F1 Score: {f1}")
    print(f"Test CSI: {csi}")
    print(f"Test POD: {pod}")
    print(f"Test FAR: {far}")
    print(f"Confusion Matrix: \n{confusion_matrix(label_list, pred_list)}")

    return accuracy, f1, csi, pod, far

# 이미지 추론
def inference(model, image_path, transform, device):
    image = Image.open(image_path).convert("L")
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
        return predicted.item()
    
if __name__ == "__main__":
    csv_dict = {'급격' : 'Seoul_V1', '완만' : 'Seoul_V2'}
    version = '급격'
    csv_file = 'root/data/' + csv_dict[version] + '.csv'
    img_dir = '/root/data/images_classification/' 
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 500
    loss_type = 'focal'  # 'focal' 또는 'evl'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : ", device)
    
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

    train_loader, val_loader, test_loader = load_data(csv_file, img_dir, batch_size, transform)
    model, criterion, optimizer = initialize_model(learning_rate, loss_type, device)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, f'classification_model/{csv_dict[version]}', device)
    evaluate_model(model, val_loader, device)

    loaded_model = load_model('cf_models/classification_model_seoul1.pth', device) #수정 필요
    predict(loaded_model, test_loader, device)

    # # 예측 수행
    # image_path = 'radar_total/202105302300.png'
    # prediction = inference(loaded_model, image_path, transform, device)
    # print(f"Predicted class: {prediction}")

# # inference 확인할 때 코드
# if __name__ == "__main__":
#     csv_file = 'data/서울_2021_2023_강수량 0.1 미만 제거_상위 10% test.csv'
#     img_dir = 'radar_total' 
#     batch_size = 32
#     learning_rate = 0.001
#     num_epochs = 100
#     loss_type = 'focal'  # 'focal' 또는 'evl'
    
#     left = 240  
#     top = 120   
#     right = 390
#     bottom = 270
    
#     transform = transforms.Compose([
#         #transforms.Lambda(lambda x: x.crop((left, top, right, bottom))), # crop
#         transforms.Grayscale(num_output_channels=1), # grayscale
#         transforms.ToTensor()
#     ])    

#     train_loader, val_loader, test_loader = load_data(csv_file, img_dir, batch_size, transform)
#     model, criterion, optimizer = initialize_model(learning_rate, loss_type, device)
#     # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 'cf_models/classification_model_seoul1')
#     # evaluate_model(model, val_loader)
#     # save_model(model, 'cf_models/classification_model_seoul1.pth')

#     loaded_model = load_model('cf_models/classification_model_seoul1.pth', device)
#     # predict(loaded_model, test_loader, device)

#     # 예측 수행
#     image_path = 'radar_total/202105302300.png'
#     prediction = inference(loaded_model, image_path, transform, device)
#     print(f"Predicted class: {prediction}")

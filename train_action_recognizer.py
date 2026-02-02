import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=101):
        super(ActionRecognitionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.resnet(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = self.fc(out[:, -1, :])
        return out

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.UCF101(root='data/ucf101', annotation_path='data/ucf101/ucfTrainTestlist', frames_per_clip=16, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = ActionRecognitionModel(num_classes=101)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'models/action_recognizer.pt')

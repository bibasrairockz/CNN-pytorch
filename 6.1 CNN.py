import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Processor: {device}")
transform= transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_dataset= torchvision.datasets.CIFAR10(root='./data', train= True, transform= transform,
                                            download= True)
test_dataset= torchvision.datasets.CIFAR10(root='./data', train= False, transform= transform,
                                           download= True)
batch_size= 4
# print(f'{train_dataset[0][0].shape}')
n_iters= 50
learning_rate= 0.001
n_psteps= 1000

train_loader= DataLoader(dataset= train_dataset, shuffle= True, batch_size= batch_size)
test_loader= DataLoader(dataset= test_dataset, shuffle= False, batch_size= batch_size)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1= nn.Conv2d(3, 6, 5)
        self.maxp= nn.MaxPool2d(2, 2)
        self.conv2= nn.Conv2d(6, 16, 5)
        self.fc1= nn.Linear(16*5*5, 120)
        self.fc2= nn.Linear(120, 84)
        self.fc3= nn.Linear(84, 10)

    def forward(self, x):
        out= self.maxp(F.relu(self.conv1(x)))
        out= self.maxp(F.relu(self.conv2(out)))
        out= out.view(-1, 16*5*5)
        out= F.relu(self.fc1(out))
        out= F.relu(self.fc2(out))
        out= self.fc3(out)
        return out

model= ConvNet().to(device)

loss= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size= 15, gamma= 0.1)

def train():
    n_steps= len(train_loader)
    for epoch in range(n_iters):
        for i, (data, labels) in enumerate(train_loader):
            data= data.to(device)
            labels= labels.to(device)

            Y_pred= model(data)
            cost= loss(Y_pred, labels)

            optimizer.zero_grad()
            cost.backward()

            optimizer.step()

            if (i+1) % n_psteps == 0:
                print(f"Epoch [{epoch+1}/{n_iters}] | Steps [{i+1}/{n_steps}]: COST= {cost.item():.4f}")

        scheduler.step()
        torch.save(model.state_dict(), "./cnn1.pth")
        break


def accuracy():
    n_samples = n_pred = 0
    n_class_samples = [0 for _ in range(10)]
    n_class_pred = [0 for _ in range(10)]

    model = ConvNet().to(device)
    model.load_state_dict(torch.load('./cnn1.pth'))
    model.eval()

    for (data, labels) in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        Y_pred = model(data)
        _, pred = torch.max(Y_pred, 1)  
        n_pred += (pred == labels).sum().item()
        n_samples += labels.shape[0]

        for i in range(labels.size(0)):
            label = labels[i].item()
            n_class_samples[label] += 1
            if label == pred[i].item():
                n_class_pred[label] += 1

    # Overall accuracy
    acc = (n_pred / n_samples) * 100.0
    print(f"Overall Accuracy = {acc:.2f}%")

    # Accuracy per class
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        if n_class_samples[i] > 0:
            acc_class = 100.0 * (n_class_pred[i] / n_class_samples[i])
        else:
            acc_class = 0.0
        print(f'Accuracy of {classes[i]}: {acc_class:.2f}%')


def infer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvNet().to(device)
    model.load_state_dict(torch.load('./cnn1.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open('frog.jpg')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f'Predicted class index: {classes[predicted.item()]}')
            

if __name__=="__main__":
    # train()
    # accuracy()
    infer()
    pass

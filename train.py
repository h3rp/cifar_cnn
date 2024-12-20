import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Models.CNN import SimpleCNN

# Transformasi
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Dataset dan DataLoader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(trainset))  # 80% untuk training
valid_size = len(trainset) - train_size  # 20% untuk validasi
train_subset, valid_subset = torch.utils.data.random_split(trainset, [train_size, valid_size])

trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_subset, batch_size=32, shuffle=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        
        optimizer.zero_grad()

        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_losses.append(running_train_loss / len(trainloader))

    
    model.eval()
    running_valid_loss = 0.0
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item()

    valid_losses.append(running_valid_loss / len(validloader))

    print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.3f}, Valid Loss: {valid_losses[-1]:.3f}")



torch.save(model.state_dict(), './model.pth')

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig('./loss_plot.png')  
# plt.show()

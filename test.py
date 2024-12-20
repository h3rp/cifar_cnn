import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Models.CNN import SimpleCNN


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('./model.pth'))  
model.eval()


correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Akurasi : {100 * correct / total:.2f}%")


dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    img = images[i].cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5  
    axes[i].imshow(img)
    axes[i].set_title(f"GT: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    axes[i].axis("off")
# plt.show()
plt.savefig('./hasil_prediksi.png')  
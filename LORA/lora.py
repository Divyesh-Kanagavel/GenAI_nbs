# import required libraries
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt
from tqdm import tqdm

# to make random generation deterministic
torch.manual_seed(42)

# MNIST dataset is used to illustrate LORA of neural networks

# transform to be applied to the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# fetch the train dataset from torch.datasets
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# batch load the dataset using DataLoader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
# prepare the test dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# batch load the dataset using DataLoader
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=16, shuffle=True)

# set the device
device = torch.device("mps" if torch.mps.is_available() else "cpu")

print(device)

IMAGE_SIZE = 28
NUM_DIGITS = 10

# constructing a very large neural network for this task of digit classification
# this is to simulate some intermediate weight matrices in large language models
# which are too big for the information they encode for a specific task
class BigNet(nn.Module):
    def __init__(self, hidden_size1 = 500, hidden_size2 = 1000):
        super(BigNet,self).__init__()
        self.linear1 = nn.Linear(IMAGE_SIZE*IMAGE_SIZE, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, NUM_DIGITS)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE*IMAGE_SIZE)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.linear3(x)

net = BigNet().to(device)
print(net)

# write the training loop

def train(train_loader, net, epochs=5, total_iteration_limit=None):
    cross_entropy_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr = 1e-2)
    total_iterations = 0

    for epoch in range(epochs):
        net.train()
        num_iterations = 0
        loss_sum = 0

        data_iterator = tqdm(train_loader, desc = f"Epoch {epoch+1}")
        if total_iteration_limit is not None:
            data_iterator.total = total_iteration_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x,y = data
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            output = net(x.view(-1,IMAGE_SIZE*IMAGE_SIZE))
            loss = cross_entropy_loss(output, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            if total_iteration_limit is not None and total_iterations >= total_iteration_limit:
                return
print("pre-training the model on MNIST data")        
train(train_loader, net, epochs = 1)

original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()

# testing our trained net on test dataset

def test():
    correct = 0
    total = 0

    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = net(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                else:
                    wrong_counts[y[idx]] +=1
                total +=1
    print(f'Accuracy: {round(correct/total, 3)}')
    for i in range(len(wrong_counts)):
        print(f'wrong counts for the digit {i}: {wrong_counts[i]}')
print("testing the model on test set!")
test()

# upon investigation, there were lot of misclassifications for digit 2
# So we will construct lora matrices for the net and fine-tune them to classifiy the digit 2 better

# Print the size of the weights matrices of the network
# Save the count of the total number of parameters
total_parameters_original = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
    print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
print(f'Total number of parameters: {total_parameters_original:,}')

# LORA parameterization

class LORAParameterization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha=1, device="cpu"):
        super(LORAParameterization, self).__init__()
        # A and B are the low rank matrices
        # delW = BA
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        # matrix A is initialized to normal initialization 
        # if both are set to zeros, there is no training! Found it out by trial and error
        nn.init.normal_(self.lora_A, mean=0, std=1)
        # delW = BA is multiplied by scale to prevent larger updates when rank is higher
        # for numerical stability we multiply this with the scale.
        # alpha is an additional tuning variable controlling the updates
        # usually setting alpha to rank gives good numerical stability
        self.scale = alpha / rank
        self.enabled = True
    
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights 
    

# we are going to use parametrize functionality of pytorch 
# to augment of modify a layer of a network without changing its functionality

# simple example to understand!
print("a simple example to understand LORA!!!")
linear = nn.Linear(2,3)
# linear weights before parametrization
print(linear.weight)
# class to scale weights
class ScaleWeight(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self,x):
        return x * self.scale

# apply torch parametrization
parametrize.register_parametrization(linear, "weight", ScaleWeight(2))
# linear weights after parametrization
print(linear.weight)

# parametrizing our BigNet
print("parametrizing our BigNet with LORA!")
def linear_layer_parametrize(layer, device, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LORAParameterization(features_in, features_out,
                                rank, lora_alpha,device)

parametrize.register_parametrization(net.linear1, "weight", linear_layer_parametrize(net.linear1, device))
parametrize.register_parametrization(net.linear2, "weight", linear_layer_parametrize(net.linear2, device))
parametrize.register_parametrization(net.linear3, "weight", linear_layer_parametrize(net.linear3, device))

# print the weights
print("the weights of the network after parametrization are : ", net.linear1.parametrizations["weight"][0])

# function to enable/disable lora addition to our network layers
def enable_disable_lora(enabled=True):
    for layer in [net.linear1, net.linear2, net.linear3]:
        layer.parametrizations["weight"][0].enabled = enabled

#display the number of parameters in our LORA

print("the number of parameters in our LORA = ")
total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    print(
        f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
    )
# The non-LoRA parameters count must match the original network
assert total_parameters_non_lora == total_parameters_original
print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
print(f'Parameters incremment: {parameters_incremment:.3f}%')

# coming to Fine tuning!!

# freeze the non-LORA parameters -> they are not trained during fine tuning

# Freeze the non-Lora parameters
for name, param in net.named_parameters():
    if 'lora' not in name:
        print(f'Freezing non-LoRA parameter {name}')
        param.requires_grad = False

# Load the MNIST dataset again, by keeping only the digit 2
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = mnist_trainset.targets == 2
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)

# Train the network with LoRA only on the digit 2 and only for 100 batches (hoping that it would improve the performance on the digit 9)
train(train_loader, net, epochs=1, total_iteration_limit=100)

# Check that the frozen parameters are still unchanged by the finetuning
assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

enable_disable_lora(enabled=True)

assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original one
assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])

# test with lora enabled
print("test with lora enabled!")
enable_disable_lora(enabled=True)
print(net.named_parameters())
test()

print("test with lora disabled!")
enable_disable_lora(enabled=False)
print(net.named_parameters())
test()














import torch
import torch.nn as nn

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Create dummy input and target tensors (data)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])


# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
   nn.ReLU(),
   nn.Linear(n_h, n_out),
   nn.Sigmoid())

# Construct the loss function
criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Gradient Descent
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   # Compute and print loss
   loss = criterion(y_pred, y)
   print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()

#RNN
# from torch.autograd import Variable
# import torch.nn.functional as F
#
# class SimpleCNN(torch.nn.Module):
#    def __init__(self):
#       super(SimpleCNN, self).__init__()
#       #Input channels = 3, output channels = 18
#       self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
#       self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
#       #4608 input features, 64 output features (see sizing flow below)
#       self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
#       #64 input features, 10 output features for our 10 defined classes
#       self.fc2 = torch.nn.Linear(64, 10)
#
# def forward(self, x):
#    x = F.relu(self.conv1(x))
#    x = self.pool(x)
#    x = x.view(-1, 18 * 16 *16)
#    x = F.relu(self.fc1(x))
#    #Computes the second fully connected layer (activation applied later)
#    #Size changes from (1, 64) to (1, 10)
#    x = self.fc2(x)
#    return(x)
#

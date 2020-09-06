# こりゃはtorchのauto-gradの例

import torch

dtype = torch.float
device = torch.device("cpu")

# device = torch.device('cuda:0') # running on gpu

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input& output data

x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# random initialize the weights& bias

w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

# define learning rate

learning_rate = 1e-6

# train loop

'''
example without autograd
for t in range(500):
    # foward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)

    # loss computing, print
    loss = (y_pred - y).pow(2).sum().item()  #
    if t % 100 == 99:
        print(loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update the weights

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2 
'''

for i in range(500):

    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #print(y_pred)
    #print(y - y_pred)
    # loss computing, print
    loss = (y_pred - y).pow(2).sum()  #
    if i % 100 == 99:
        print(i, loss.item())

    #loss.requires_grad=True 

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
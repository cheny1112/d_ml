import torch
from torch import nn

class TwoLayersNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels):

        super(TwoLayersNet, self).__init__()
        self.linear1 = nn.Linear(input_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, output_channels)

    def forward(self, x):

        h_relu = self.linear1(x).clamp(0)
        y_pred = self.linear2(h_relu)

        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayersNet(D_in, D_out, H)

loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if i % 100 == 99:
        print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


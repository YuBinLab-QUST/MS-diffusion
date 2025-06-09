from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


#加载数据集
dataset = Planetoid(root = "/tmp/Cora", name='Cora')

#定义GAT模型
class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.num_layers = 2
        self.conv1 = GATConv(in_channels=in_channels, out_channels=16, heads=8, dropout=0.6)
        self.conv2 = GATConv(in_channels=16 * 8, out_channels=out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


#创建模型，其中num_features 和 num_classes都是pyg数据集中包含的属性
#model = Net(in_channels=dataset.num_features, out_channels=dataset.num_classes)

#模型训练
#遍历加载器中的每个数据批次，对模型进行训练，对每个数据图批次，计算网络的输出、预测和损失，反向传播来更新权重
#最后将总损失和预测正确率记录并返回
def train(model, loader, optimizer, loss_fn):
    model.train()
    correct = 0
    total_loss = 0
    for _, data in enumerate(tqdm(loader, total=len(loader), desc="Training")):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        pred = out.argmax(dim=1)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
    return total_loss / data.train_mask.sum().item(), correct / data.train_mask.sum().item()

#定义测试函数

def test(model, loader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, total=len(loader), desc="Testing")):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            loss = loss_fn(out[data.test_mask], data.y[data.test_mask])
            total_loss += loss.item()
            correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return total_loss / data.test_mask.sum().item(), correct / data.test_mask.sum().item()



#定义主函数来完成测试

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = Net(in_channels=dataset.num_features, out_channels=dataset.num_classes).to(device)
    data = dataset[0].to(device)
    train_loader = [data]
    test_loader = [data]
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, 1001):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
        test_loss, test_acc = test(model, test_loader, loss_fn)
        print(f"Epoch {epoch:03d}, Train_loss: {train_loss: .4f}, Train_acc: {train_acc:.4f},"
              f"Test Loss: {test_loss: .4f}, Test Acc: {test_acc: .4f}")


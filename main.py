import torch
import os

from torch.utils.tensorboard import SummaryWriter

from generate_data import generate_data, save_data, load_data, test_function_analytics

def main():
    data_dir = './data/' + 'train_data.pth'
    if not os.path.exists(data_dir):
        print('Data does not exist... generating...')
        os.makedirs('./data/')
        x, y = generate_data()
        save_data(x, y, './data/')
    else:
        print('Data already exists... loading...')
        x, y = load_data(data_dir)
        print(x.shape, y.shape)


    class MyMLP(torch.nn.Module):
        def __init__(self):
            super(MyMLP, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(3, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 12),
                torch.nn.ReLU(),
                torch.nn.Linear(12, 3)
            )
            
        def forward(self, x):
            x = self.model(x)
            return x
        
    model = MyMLP()

    writer = SummaryWriter("logs")

    # Define the loss function
    loss_mse = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 5000
    for epoch in range(num_epochs):
        y_pred = model(x)
        loss = loss_mse(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
            writer.add_scalar('Loss/train', loss.item(), epoch)


    # Save the model
    model_dir = './model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + 'model.pth')

    # Load the model
    model = MyMLP()
    model.load_state_dict(torch.load(model_dir + 'model.pth'))
    model.eval()
    #print(model.state_dict())

    # Test the model
    x_test = 2*torch.rand(5, 3)
    y_test = model(x_test)
    y_real = test_function_analytics(x_test)
    print(y_test-y_real)
    test_loss = loss_mse(y_test, y_real)
    print("Test loss: ", test_loss)
    print(torch.mean(torch.abs(y_test-y_real))) 


if __name__ == "__main__":
    main()


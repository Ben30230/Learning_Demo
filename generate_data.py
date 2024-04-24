import torch

def test_function_analytics(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[:,0] = x[:,0] * x[:,1]
    y[:,1] = x[:,1] ** 2
    y[:,2] = x[:,2] ** 3
    return y


def generate_data() -> tuple[torch.Tensor, torch.Tensor]:
    # Generate some data
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_train = 50000
    x = 5*torch.rand(num_train, 3)
    y = test_function_analytics(x)
    return x, y

def save_data(x: torch.Tensor, y: torch.Tensor, path: str) -> None:
    # Save the data
    train_data = {}
    train_data['x'] = x
    train_data['y'] = y
    torch.save(train_data, path + 'train_data.pth')

def load_data(path) -> tuple[torch.Tensor, torch.Tensor]:
    # Load the data
    train_data = torch.load(path)
    x = train_data['x']
    y = train_data['y']
    return x, y
    

if __name__ == '__main__':
    x, y = generate_data()
    print(x.shape, y.shape)
    save_data(x, y, './data/')
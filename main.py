from src.data_loader_and_process import data_loader_and_process
from src.dataset import dataset_prep
from src.model import build_model
from src.train import train_model

data_loader_and_process()
train_data, test_data = dataset_prep()
model = build_model()
history = train_model(train_data, test_data, model)
print(history)
import torch
from torchviz import make_dot
from pathlib import Path

# Visualize the model graph (requires torchviz)
def visualize_model_graph(model, input_shape = [1, 22, 1000], model_name = "model_graph"):     
    input_tensor = torch.randn(*input_shape).to(model.device)  
    save_path =  Path(__file__).resolve().parent / model_name
    dot = make_dot(model(input_tensor), params=dict(model.named_parameters()))
    dot.render(save_path, format="png")

# Displays information about the available GPUs
def show_gpu_info():
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Names:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
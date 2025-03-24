import torch
import json
import numpy as np
import os
import onnxruntime as ort
from tqdm import tqdm, trange
from analyze_args import get_args
import random

state_dim = 150
action_dim = 1
max_action = 20
device = "cuda"
discount = 0.99
tau = 0.005
max_q_backup = False
beta_schedule = 'vp'
# 记得改
T = 10
eta = 1
lr = 3e-5
lr_decay = 'store_true'
num_epochs = 2000
gn = 10.0

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


if __name__ == "__main__":
    onnx_path = '/home/czy/meta/Meta_diffusion_QL/onnx_model/meta-v14-id-gru-eta10-TD7onlyzs-goodin33-3407-1020.onnx' 
    
    # load diffusion policy
    from diffusion.agents.ql_diffusion4v14iql_id_gru_TD7_onlyzs import Diffusion_QL as Agent
    args = get_args()
    agent = Agent(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=device,
                    discount=discount,
                    tau=tau,
                    max_q_backup=max_q_backup,
                    beta_schedule=beta_schedule,
                    n_timesteps=T,
                    eta=eta,
                    lr=lr,
                    lr_decay=lr_decay,
                    lr_maxt=num_epochs,
                    grad_norm=gn,
                    args=args)
    torchBwModel = agent.actor
    torchBwModel.load_state_dict(torch.load('/home/czy/meta/Meta_diffusion_QL/results/meta-v14-id-gru-eta10-TD7-onlyzs|exp1|diffusion-ql|T-10|lr_decay|ms-online|3407/actor_1020.pth'))
    torchBwModel.to("cpu")
    torchBwModel.eval()
    
    # load encoder
    encoder = torch.load('/home/czy/meta/Meta_diffusion_QL/TD7/encoder_1182.0.pt').to("cpu").eval()
    
    # load data
    evaluation_file = '/home/czy/Schaferct/ALLdatasets/emulate/emulate/00000.json'
    with open(evaluation_file, "r") as file:
        call_data = json.load(file)
    observations = np.asarray(call_data['observations'], dtype=np.float32)
    observations = observations.reshape(-1, 150)
    
    # prepare pt model inputs
    observations_tensor = torch.as_tensor(observations).to("cpu")
    with torch.no_grad():
        encoder_zs_tensor = encoder.zs(observations_tensor).to("cpu")
        
    # prepare onnx model inputs
    # export onnx models
    torch.onnx.export(
        torchBwModel,
        (observations_tensor[0:1, :], encoder_zs_tensor[0:1, :]),
        onnx_path,
        opset_version=11,
        input_names=['obs', 'zs'], # the model's input names
        output_names=['action'], # the model's output names
    )
    

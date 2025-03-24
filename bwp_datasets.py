import numpy as np
import pickle
from typing import Dict, Callable, List, Union
import numpy as np
import torch

obs_min = np.array([6933.3335,0.,    0.,    0.,    0.,  693.3333
                            ,0.,    0.,    0.,    0.,    1.,    0.
                            ,0.,    0.,    0.,    1.,    0.,    0.
                            ,0.,    0.,   52.,    0.,    0.,    0.
                            ,0.,   52.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,-2187.5,    -2186.
                            , -2188.,-2188.,-2188.,-2179.5386, -2179.5386, -2179.818
                            , -2179.5,    -2179.5386, -1992.,-1992.,-1992.,-1992.
                            , -1992.,-1992.,-1992.,-1992.,-1992.,-1992.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.
                            ,0.,    0.,    0.,    0.,    0.,    0.])
obs_max = np.array([2.2961187e+08,2.2961187e+08,2.2961187e+08,2.2961187e+08,2.2576613e+08
                            ,2.5140640e+07,1.8429560e+07,2.5140640e+07,2.1878146e+07,2.1878146e+07
                            ,1.7480000e+03,1.7480000e+03,1.7480000e+03,1.7480000e+03,1.7190000e+03
                            ,1.9100000e+03,1.6560000e+03,1.9100000e+03,1.7060000e+03,1.7060000e+03
                            ,1.7220890e+06,1.7220890e+06,1.7220890e+06,1.7220890e+06,1.6932460e+06
                            ,1.8855480e+06,1.3822170e+06,1.8855480e+06,1.6408610e+06,1.6408610e+06
                            ,8.0290000e+03,8.0716841e+03,8.0716841e+03,7.1589414e+03,6.8999048e+03
                            ,8.0290000e+03,8.0716841e+03,7.6990000e+03,7.2922798e+03,7.1413486e+03
                            ,8.0270000e+03,8.0696841e+03,8.0696841e+03,7.1569414e+03,6.8979048e+03
                            ,8.0270000e+03,8.0696841e+03,7.6412500e+03,7.2892798e+03,7.0593486e+03
                            ,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02
                            ,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02,2.0000000e+02
                            ,4.3364285e+02,3.8484000e+02,4.1659259e+02,1.4950000e+02,4.4811111e+02
                            ,4.3364285e+02,6.0400000e+02,8.3088892e+02,6.0400000e+02,4.1924139e+02
                            ,4.3907275e+03,4.0823064e+03,4.3304302e+03,4.3386089e+03,4.3304302e+03
                            ,4.7623037e+03,4.8302510e+03,4.6838955e+03,4.7108174e+03,4.6838955e+03
                            ,3.1400000e+03,3.0880000e+03,2.9200000e+03,3.0880000e+03,2.9860000e+03
                            ,3.1400000e+03,2.8910000e+03,3.1400000e+03,3.1400000e+03,3.0000000e+03
                            ,1.5530000e+03,1.5739735e+03,1.5530000e+03,1.5530000e+03,1.5530000e+03
                            ,1.5530000e+03,1.5530000e+03,1.5305000e+03,1.4965000e+03,1.5530000e+03
                            ,9.8529410e-01,9.8484850e-01,9.9009901e-01,9.8969072e-01,9.8969072e-01
                            ,9.8529410e-01,9.5717883e-01,9.7368419e-01,9.9000001e-01,9.9029124e-01
                            ,2.6875000e+02,3.7900000e+02,3.7900000e+02,1.5600000e+02,1.6000000e+02
                            ,2.6875000e+02,1.9000000e+02,1.6000000e+02,1.9000000e+02,1.6000000e+02
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00
                            ,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00,1.0000000e+00])
normal_vector = np.array(
        [
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1e-1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    

TensorBatch = List[torch.Tensor]

DATASET_DEFAULT_PATH = (
    "/home/czy/Schaferct/v8.pickle"
)
b_in_Mb = 1e6

MAX_ACTION = 20  # Mbps
STATE_DIM = 150
ACTION_DIM = 1

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def adjust_dataset(dataset_dict):
    dataset_dict["actions"] /= b_in_Mb
    dataset_dict["observations"] *= normal_vector
    dataset_dict["next_observations"] *= normal_vector
    return dataset_dict

def adjust_dataset_in_NAORL(dataset_dict):
    dataset_dict["actions"] /= b_in_Mb
    dataset_dict["observations"] = np.clip(((dataset_dict["observations"] - obs_min) / (obs_max - obs_min + 1e-3) * 2.0 - 1.0), -1.0, 1.0)
    dataset_dict["next_observations"] = np.clip(((dataset_dict["next_observations"] - obs_min) / (obs_max - obs_min + 1e-3) * 2.0 - 1.0), -1.0, 1.0)
    return dataset_dict


def GetBwpDataset(pickle_path: str = DATASET_DEFAULT_PATH):
    testdataset_file = open(pickle_path, "rb")
    dataset_dict = pickle.load(testdataset_file)

    # Normalize the actions and states
    dataset_dict = adjust_dataset(dataset_dict)
    print("dataset loaded")
    return dataset_dict




def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        
        # Added for LAP
        self.priority = torch.zeros(buffer_size, device=device)
        self.max_priority = 1

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        print(f"Dataset size: {n_transitions}")
        if n_transitions > self._buffer_size:
            raise ValueError(
                f"Replay buffer is {n_transitions-self._buffer_size} smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        
        # Added for LAP
        self.priority = torch.ones(self._size).to(self._device)

    def LAP_sample(self, batch_size: int) -> TensorBatch:
        csum = torch.cumsum(self.priority[:self._size], 0)
        val = torch.rand(size=(batch_size,), device=self._device)*csum[-1]
        self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        
        states = self._states[self.ind]
        actions = self._actions[self.ind]
        rewards = self._rewards[self.ind]
        next_states = self._next_states[self.ind]
        dones = self._dones[self.ind]
        return [states, actions, rewards, next_states, dones]
    
    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        self.ind = indices
        
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self, data: Dict[str, np.ndarray]):
        # Use this method to add new data into the replay buffer during fine-tuning.
        if self._size != 0:
            ini_idx = self._pointer
        else:
            ini_idx = 0
        n_transitions = data["observations"].shape[0]
        print(f"Dataset size add: {n_transitions}")
        if n_transitions > self._buffer_size:
            raise ValueError(
                f"Replay buffer is {ini_idx + n_transitions - self._buffer_size} smaller than the dataset you are trying to load!"
            )
        self._states[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["observations"])
        self._actions[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["actions"])
        self._rewards[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[ini_idx:ini_idx + n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, ini_idx + n_transitions)
        print(f"Dataset added: {self._pointer}")
        
        # Added for LAP
        self.priority = torch.ones(self._size).to(self._device)

    def update_priority(self, priority):
        priority = priority.to(self._device)
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)
        
    def reset_max_priority(self):
        self.max_priority = float(self.priority[:self._size].max())

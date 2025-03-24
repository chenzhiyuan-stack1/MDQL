import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import json
import math
# import TD7_congestion
import TD7_congestion_good
# import TD7_congestion_select_feature
from bwp_datasets import GetBwpDataset, ReplayBuffer
from norm_vector import NORMAL_VECTOR
import wandb

def maybe_evaluate_and_print(RL_agent, small_evaluation_datasets, start_time):
	print("---------------------------------------")
	print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
	# ======= test each dataset ========    
	every_call_mse = []
	every_call_accuracy = []
	every_call_over = []
	for call_data in tqdm(small_evaluation_datasets, desc="process the test dataset"):
		observations = np.asarray(call_data["observations"], dtype=np.float32)
		true_capacity = np.asarray(call_data["true_capacity"], dtype=np.float32)
		actions = np.asarray(call_data["bandwidth_predictions"], dtype=np.float32)
		observations = observations * NORMAL_VECTOR
		actions = actions / 1e6
		model_predictions = []
		for t in range(observations.shape[0]):
			obss_ = torch.tensor(observations[t:t+1,:].reshape(1,-1), device="cuda:0", dtype=torch.float32)
			with torch.no_grad():
				# action = policy_fn(obss_).cpu().numpy()
				fixed_zs = RL_agent.fixed_encoder.zs(obss_)
				# print(obss_.shape, fixed_zs.shape)
				action = RL_agent.actor(obss_, fixed_zs).cpu().numpy()
				action = action / 1e6
			bw_prediction = np.squeeze(action)
			model_predictions.append(bw_prediction)
		# mse and accuracy of this call
		model_predictions = np.asarray(model_predictions, dtype=np.float32)
		true_capacity = true_capacity / 1e6
		# model_predictions = model_predictions / 1e6
		call_mse = []
		call_accuracy = []
		call_over = []
		for true_bw, pre_bw in zip(true_capacity, model_predictions):
			if np.isnan(true_bw) or np.isnan(pre_bw):
				continue
			else:
				mse_ = (true_bw - pre_bw) ** 2
				call_mse.append(mse_)
				accuracy_ = max(0, 1 - abs(pre_bw - true_bw) / true_bw)
				call_accuracy.append(accuracy_)
				over = max(0,(pre_bw - true_bw) / true_bw)
				call_over.append(over)
		call_mse = np.asarray(call_mse, dtype=np.float32)
		every_call_mse.append(np.mean(call_mse))
		call_accuracy = np.asarray(call_accuracy, dtype=np.float32)
		every_call_accuracy.append(np.mean(call_accuracy))
		call_over = np.asarray(call_over, dtype=np.float32)
		every_call_over.append(np.mean(call_over))
	every_call_mse = np.asarray(every_call_mse, dtype=np.float32)
	every_call_accuracy = np.asarray(every_call_accuracy, dtype=np.float32)
	every_call_over = np.asarray(every_call_over, dtype=np.float32)
	return np.mean(every_call_mse), np.mean(every_call_accuracy), np.mean(every_call_over)
			
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# RL
	parser.add_argument("--env", default="bwp", type=str)
	parser.add_argument("--seed", default=0, type=int)
	# Evaluation
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--max_timesteps", default=1e7, type=int)
	# File
	parser.add_argument('--file_name', default=None)
	args = parser.parse_args()

	if args.file_name is None:
		args.file_name = f"TD7_{args.env}_{args.seed}_{int(time.time())}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	exp_dir = f"./results/{args.file_name}"
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	log_file_name = os.path.join(exp_dir, "log.txt")

	wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
	run = wandb.init(
        project="TD7", name=f"{args.env}-{wandb_id}"
    )
	wandb.config.update(args)
	args.run = run

	print("---------------------------------------")
	print(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = 150
	action_dim = 1
	max_action = 20 # Mbps

	# load data
    # ======= get the datasets ========
	# dataset_path = "/home/czy/Schaferct/v8.pickle"
	dataset_path = "/home/czy/Schaferct/mstrain-id-123.pickle"
	dataset = GetBwpDataset(dataset_path)
    # ======= Create Dataset =======
	replay_buffer = ReplayBuffer(
        150,
        1,
        # 6_538_000,
        20_000_000,
    )
	replay_buffer.load_dataset(dataset)
	del dataset
	add_data_path = "/home/czy/Schaferct/mstrain-id-345.pickle"
	add_dataset = GetBwpDataset(add_data_path)
	replay_buffer.add_transition(add_dataset)
	del add_dataset

	# test dataset
	# ======= get test datasets ========
	small_evaluation_datasets = []
	eval_data_path = "/home/czy/Schaferct/ALLdatasets/emulate"
	policy_dir_names = os.listdir(eval_data_path)   
	# ======= choose 20 test json files ========     
	every_epoch_test_loss_list = []
	for p_t in policy_dir_names:
		policy_type_dir = os.path.join(eval_data_path, p_t)
		for e_f_name in os.listdir(policy_type_dir)[:20]:
			e_f_path = os.path.join(policy_type_dir, e_f_name)
			with open(e_f_path, "r") as file:
				call_data = json.load(file)
			small_evaluation_datasets.append(call_data)
 
	# create agent
	# RL_agent = TD7_congestion.IQL_Agent(state_dim, action_dim, replay_buffer)
	RL_agent = TD7_congestion_good.IQL_Agent(state_dim, action_dim, replay_buffer)
	# RL_agent = TD7_congestion_select_feature.IQL_Agent(state_dim, action_dim, replay_buffer)

	for t in range(int(args.max_timesteps+1)):
		start_time = time.time()
		RL_agent.train(args)
		if t % args.eval_freq == 0:
			current_epoch = t // args.eval_freq
			mse, acc, over = maybe_evaluate_and_print(RL_agent, small_evaluation_datasets, start_time)
			# logging evaluation results
			args.run.log({"mse": mse, "acc": acc, "over": over}, step=t+1)
			with open(log_file_name, "a") as file:
				file.write(f"current epoch: {current_epoch}, mse: {mse}, acc: {acc}, over: {over}\n")
			# save model checkpoints
			actor_name = "actor_" + str(current_epoch)
			q_name = "q_critic_" + str(current_epoch)
			v_name = "v_critic_" + str(current_epoch)
			encoder_name = "encoder_" + str(current_epoch)
			actor_dir = os.path.join(exp_dir, actor_name + ".pt")
			q_dir = os.path.join(exp_dir, q_name + ".pt")
			v_dir = os.path.join(exp_dir, v_name + ".pt")
			encoder_dir = os.path.join(exp_dir, encoder_name + ".pt")
			torch.save(RL_agent.actor, actor_dir)
			torch.save(RL_agent.q_critic, q_dir)
			torch.save(RL_agent.v_critic, v_dir)
			torch.save(RL_agent.encoder, encoder_dir)

	args.run.finish()
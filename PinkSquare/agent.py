import torch
import random
import numpy as np
from collections import deque
from red_tide_the_game import *
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001


class Agent:
    target_entity: OrganismJelly

    def __init__(self, target_entity):
        self.target_entity = target_entity

        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12, 64, 17)
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.start_random = True

        # Predict actions
        all_move_actions = [0, 1, 2, 3, 4]
        all_reach_point_actions = list(range(0, 12))
        all_reach_act_actions = [1]
        self.all_possible_actions = []
        for m in all_move_actions:
            self.all_possible_actions.append((m, 0, 0))
        for rp in all_reach_point_actions:
            for ra in all_reach_act_actions:
                self.all_possible_actions.append((0, rp, ra))

    def load_model(self):
        try:
            checkpoint = torch.load('model/model.pth')
            print("Saved Model Found")
            self.model.load_state_dict(checkpoint)
            self.start_random = False
        except FileNotFoundError:
            print("No Models Saved")
            pass

    def get_state(self, game):
        sense_data = self.target_entity.get_sense_data()
        max_health = self.target_entity.max_health

        state = [
                # self.target_entity.curr_health / max_health
                 ]

        for entity_code in sense_data:
            state.append(entity_code)

        """
        min_vals = np.array(
            [0] * len(state))  # Minimum values for each feature (all zeros)
        max_vals = np.array(
            [max_health] + [2] * len(sense_data))  # Maximum values for each feature
        range_vals = max_vals - min_vals  # Range for each feature
        state_norm = (np.array(
            state) - min_vals) / range_vals  # Normalized input data

        
        state = state_norm.tolist()"""
        # print(state)

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action_index(self, state) -> int:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games

        if random.randint(0, 200) < self.epsilon and self.start_random:
            move = random.randint(0, len(self.all_possible_actions)-1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return int(move)

    def index_to_num_lst(self, index):
        num_list = [0] * len(self.all_possible_actions)
        num_list[index] = 1
        return num_list

    def index_to_game_action(self, index):
        return self.all_possible_actions[index]


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0


    game = WorldSimulation()
    game.set_game()

    agent = Agent(game.user.entity_in_control)

    round_count = 0
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        i = agent.get_action_index(state_old)
        game_action = agent.index_to_game_action(i)
        num_list = agent.index_to_num_lst(i)

        # perform move and get new state
        reward, done, score = game.play_step(game_action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, num_list, reward, state_new, done)

        # remember
        agent.remember(state_old, num_list, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.set_game()
            agent.target_entity = game.user.entity_in_control
            agent.n_games += 1
            agent.train_long_memory()

            if score > record or round_count > 200:
                record = score
                agent.model.save()
                print(">> Saved")
                round_count = 0
            else:
                round_count += 1

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
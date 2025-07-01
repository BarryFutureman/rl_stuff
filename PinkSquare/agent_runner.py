from agent import *
from red_tide_the_game import *


def run():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = WorldSimulation()
    game.set_game()

    agent = Agent(game.user.entity_in_control)
    agent.load_model()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        i = agent.get_action_index(state_old)
        game_action = agent.index_to_game_action(i)

        # perform move and get new state
        reward, done, score = game.play_step(game_action)

        if done:
            # train long memory, plot result
            game.set_game()
            agent.target_entity = game.user.entity_in_control
            agent.n_games += 1

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = WorldSimulation()
    game.set_game()

    agent = Agent(game.user.entity_in_control)
    agent.load_model()
    # agent.start_random = True

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

            if score > record or round_count > 100:
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


def train_fresh():
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


if __name__ == '__main__':
    train()

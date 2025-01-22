import time
import gymnasium
from gymnasium.wrappers import RecordVideo

from bomberman_rl import ScoreRewardWrapper, RestrictedKeysWrapper, FlattenWrapper, TimePenaltyRewardWrapper, FixedLengthOpponentsInfo

from argparsing import parse
# from bomberman_rl.envs.agent_code.rule_based_agent.agent import Agent
# from learning_agent.agent import Agent
from our_agent.agent import Agent
from our_agent.utils import TrainingLogger
from our_agent.q_learning import Model

class DummyAgent:
    def setup(self):
        pass

    def setup_training(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        return None
    
    def game_events_occurred(self, *args, **kwargs):
        pass

    def end_of_round(self, *args, **kwargs):
        pass

def loop(env, agent, args, n_episodes=20000):
    # Create logger with path relative to our_agent directory
    logger = TrainingLogger(
        save_dir='scripts/our_agent/training_logs',
        fresh=(args.weights == "fresh"),
        agent=agent,
        scenario=args.scenario
    ) if args.train else None
    
    if args.train:
        print(f"\nStarting training for {n_episodes} episodes...")
        print(f"Logs and plots will be saved to: scripts/our_agent/training_logs")
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    for episode in range(n_episodes):
        state, info = env.reset()
        terminated, truncated, quit = False, False, False
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        
        while not (terminated or truncated):
            if args.user_play:
                action, quit = env.unwrapped.get_user_action()
                while action is None and not quit:
                    time.sleep(0.1)  # wait for user action or quit
                    action, quit = env.unwrapped.get_user_action()
            else:
                action, quit = agent.act(state, train=args.train), env.unwrapped.get_user_quit()

            if quit:
                env.close()
                return None
                
            new_state, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if args.train:
                # Get the shaped reward from our agent
                shaped_reward = agent._shape_reward(info["events"])
                episode_reward += shaped_reward  # Use our shaped reward instead
                
                loss = agent.game_events_occurred(state, action, new_state, info["events"])
                if loss:
                    episode_loss += loss
            state = new_state

        if args.train:
            agent.end_of_round()
            # Log episode stats
            logger.log_episode(
                episode=episode,
                epsilon=agent.q_learning.get_epsilon(),
                loss=episode_loss/episode_length if episode_length > 0 else 0,
                reward=episode_reward,
                episode_length=episode_length
            )
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Loss: {episode_loss/episode_length if episode_length > 0 else 0:6.3f} | "
                      f"Epsilon: {agent.q_learning.get_epsilon():.3f} | "
                      f"Length: {episode_length}")

    if not args.no_gui:
        quit = env.unwrapped.get_user_quit()
        while not quit:
            time.sleep(0.5) # wait for quit
            quit = env.unwrapped.get_user_quit()

    env.close()
    print("Training complete")
    print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"That took {time.time() - start_time:.2f} seconds")

def provideAgent(passive: bool, weights: str = None, use_double_dqn: bool = False):
    if passive:
        return DummyAgent()
    else:
        agent = Agent(use_double_dqn=use_double_dqn)
        if weights == "fresh":
            # Use the appropriate model based on use_double_dqn flag
            if use_double_dqn:
                from our_agent.double_q_learning import Model as DoubleModel
                agent.q_learning = DoubleModel(load=False)
            else:
                from our_agent.q_learning import Model
                agent.q_learning = Model(load=False)
        elif weights:  # if weights is a timestamp
            agent.q_learning.weights_suffix = weights  # Store the weights to load later
        return agent

def main(argv=None):
    args = parse(argv)
    env = gymnasium.make("bomberman_rl/bomberman-v0", args=args)
    env = FixedLengthOpponentsInfo(env, 3)

    # Notice that you can not use wrappers in the tournament!
    # However, you might wanna use this example interface to kickstart your experiments
    # env = ScoreRewardWrapper(env)
    # env = TimePenaltyRewardWrapper(env, penalty=.1)
    #env = RestrictedKeysWrapper(env, keys=["self_pos"])
    #env = FlattenWrapper(env)
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=args.match_name)

    agent = provideAgent(passive=args.passive, weights=args.weights, use_double_dqn=args.use_double_dqn)
    if agent is None and not args.passive and not args.user_play:
        raise AssertionError("Either provide an agent or run in passive mode by providing the command line argument --passive")
    if args.train:
        agent.setup_training()

    loop(env, agent, args)


if __name__ == "__main__":
    main()
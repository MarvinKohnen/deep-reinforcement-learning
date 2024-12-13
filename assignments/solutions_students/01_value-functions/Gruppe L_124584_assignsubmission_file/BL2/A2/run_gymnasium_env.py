import gymnasium
import gymnasium_env

render_mode = "human"
size = 5
custom_transitions = {
    (0,1): ([4,1], 10),
    (0,3): ([2,3], 5)
}
env = gymnasium.make('gymnasium_env/GridWorld-v0', 
                     render_mode = render_mode, 
                     size = size, 
                     custom_transitions = custom_transitions)
steps = []
observation, info = env.reset()
steps.append(observation)
num_steps = 10
#Da keine abbruchbedingung genannt wurde, muss zuvor definiert werden, wie viele steps eine Episode umfassen soll
for i in range(num_steps):
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    steps.append(observation)
print(steps)
env.close()
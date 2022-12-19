"""A script that exercises the basic Avalon functionality to ensure the install was successful."""
from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.common.log_utils import configure_local_logger

if __name__ == "__main__":
    configure_local_logger()

    env_params = GodotEnvironmentParams(
        resolution=96,
        training_protocol=TrainingProtocolChoice.SINGLE_TASK_EAT,
        initial_difficulty=0,
    )
    env = AvalonEnv(env_params)
    env.reset()

    def random_env_step():
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
        return obs

    observations = [random_env_step() for _ in range(50)]

    print("avalon seems to be working properly!")  # script

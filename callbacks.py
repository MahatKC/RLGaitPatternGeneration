from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> bool:
        x_robot_value = self.locals['infos'][0]['x_robot_value']
        self.logger.record("x_robot_value", x_robot_value)

        return True

    def _on_step(self) -> bool:
        return True
    

# Gait Pattern Generation for Humanoid Robots using Reinforcement Learning

One of the biggest challenges in the field of humanoid robots is walking because it involves two stages: gait pattern generation to trajectory following control. Most of the classical approaches use dynamic models, such as pendulums, to generate curves that represent the robot's center of mass movements along with controllers to follow these trajectories. However, these approaches require parameter tuning for both stages. Our approach aims to surpass this issue for the first stage, by using a Reinforcement Learning agent to learn trajectories that produce human-like walk and, at the same time, are feasible for the controllers. We used the SAC algorithm with a Multilayer Perceptron as the agent and designed our Gymnasium environment with the NAO robot. We also used the PyBullet simulator to train our model and visualize our results. We realized that the robot successfully learned how to walk, but the model can still be improved to generate a perfectly straight-line trajectory and be generalizable to other humanoid robots.

---
## Installation

Install the libraries in the `requirements.txt` file by running:

```
pip install -r requirements.txt
```

Make sure to check the installation instructions for QiBullet for specific instructions on how to install the robot meshes: https://github.com/softbankrobotics-research/qibullet

Then, install the Gymnasium environment by running:

```
pip install -e humanoid-envs
```

---

Disclaimer: The files inside the folder "controller_functions" were taken from the 2022 BSc Thesis “Controle de caminhada para robôs humanoides kidsize” by Dimitria Silveria and have not seen significant changes in this project (beyond slight changes to integrate it with the RL agent).

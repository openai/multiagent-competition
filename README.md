**Status:** Archive (code is provided as-is, no updates expected)

# Competitive Multi-Agent Environments

This repository contains the environments for the paper [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748)

## Dependencies
Use `pip install -r requirements.txt` to install dependencies. If you haven't used MuJoCo before, please refer to the [installation guide](https://github.com/openai/mujoco-py).
The code has been tested with the following dependencies:
* Python version 3.6
* [OpenAI GYM](https://github.com/openai/gym) version 0.9.1 with MuJoCo 1.31 support (use [mujoco-py version 0.5.7](https://github.com/openai/mujoco-py/tree/0.5))
* [Tensorflow](https://www.tensorflow.org/versions/r1.1/install/) version 1.1.0
* [Numpy](https://scipy.org/install.html) version 1.12.1

## Installing Package
After installing all dependencies, make sure gym works with support for MuJoCo environments.
Next install `gym-compete` package as:
```bash
cd gym-compete
pip install -e .
```
Check install is successful by coming out of the directory and trying `import gym_compete` in python console. Some users might require a `sudo pip install`.

## Trying the environments
Agent policies are provided for the various environments in folder `agent-zoo`. To see a demo of all the environments do:
```bash
bash demo_tasks.sh all
```
To instead try a single environment use:
```bash
bash demo_tasks.sh <task>
```
where `<task>` is one of: `run-to-goal-humans`, `run-to-goal-ants`, `you-shall-not-pass`, `sumo-ants`, `sumo-humans` and `kick-and-defend`

#!/bin/bash

task=$1
max_episodes=3
correct=0

if [ $task == "run-to-goal-ants" ] || [ $task == "all" ]
then
  echo "Task 1: Run-to-Goal (Ants)"
  python main.py --env run-to-goal-ants --param-paths agent-zoo/run-to-goal/ants/agent1_parameters-v1.pkl agent-zoo/run-to-goal/ants/agent2_parameters-v1.pkl --max-episodes $max_episodes
  correct=1
fi
if [ $task == "run-to-goal-humans" ] || [ $task == "all" ]
then
  echo "Task 1: Run-to-Goal (Humans)"
  python main.py --env run-to-goal-humans --param-paths agent-zoo/run-to-goal/humans/agent1_parameters-v1.pkl agent-zoo/run-to-goal/humans/agent2_parameters-v1.pkl --max-episodes $max_episodes
  correct=1
fi
if [ $task == "you-shall-not-pass" ] || [ $task == "all" ]
then
  echo "Task 2: You-Shall-Not-Pass"
  python main.py --env you-shall-not-pass --param-paths agent-zoo/you-shall-not-pass/agent1_parameters-v1pkl agent-zoo/you-shall-not-pass/agent2_parameters-v1.pkl --max-episodes $max_episodes
fi
if [ $task == "sumo-ants" ] || [ $task == "all" ]
then
  echo "Task 3: Sumo (Ants)"
  python main.py --env sumo-ants --param-paths agent-zoo/sumo/ants/agent_parameters-v2.pkl agent-zoo/sumo/ants/agent_parameters-v2.pkl --max-episodes $max_episodes
  correct=1
fi
if [ $task == "sumo-humans" ] || [ $task == "all" ]
then
  echo "Task 3: Sumo (Humans)"
  python main.py --env sumo-humans --param-paths agent-zoo/sumo/humans/agent_parameters-v1.pkl agent-zoo/sumo/humans/agent_parameters-v1.pkl --max-episodes $max_episodes
  correct=1
fi
if [ $task == "kick-and-defend" ] || [ $task == "all" ]
then
  echo "Task 4: Kick-and-Defend"
  python main.py --env kick-and-defend --param-paths agent-zoo/kick-and-defend/kicker/agent1_parameters-v1.pkl agent-zoo/kick-and-defend/defender/agent2_parameters-v1.pkl --max-episodes $max_episodes
  correct=1
fi
if [ $correct == 0 ]
then
  echo "Usage: bash demo_tasks.sh <task>"
  echo "where <task> is all to demo all tasks or one of: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend"
fi

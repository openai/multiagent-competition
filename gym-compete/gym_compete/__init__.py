from gym.envs.registration import register
import os

register(
    id='run-to-goal-ants-v0',
    entry_point='gym_compete.new_envs:MultiAgentEnv',
    kwargs={'agent_names': ['ant', 'ant'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__),
                "new_envs", "assets",
                "world_body.ant_body.ant_body.xml"
            ),
            'init_pos': [(-1, 0, 0.75), (1, 0, 0.75)]
            },
)

register(
    id='run-to-goal-humans-v0',
    entry_point='gym_compete.new_envs:MultiAgentEnv',
    kwargs={'agent_names': ['humanoid', 'humanoid'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body.humanoid_body.humanoid_body.xml"
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)]
            },
)

register(
    id='you-shall-not-pass-humans-v0',
    entry_point='gym_compete.new_envs:HumansBlockingEnv',
    kwargs={'agent_names': ['humanoid_blocker', 'humanoid'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body.humanoid_body.humanoid_body.xml"
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)],
            'max_episode_steps': 500,
            },
)

register(
    id='sumo-humans-v0',
    entry_point='gym_compete.new_envs:SumoEnv',
    kwargs={'agent_names': ['humanoid_fighter', 'humanoid_fighter'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body_arena.humanoid_body.humanoid_body.xml"
            ),
            'init_pos': [(-1, 0, 1.4), (1, 0, 1.4)],
            'max_episode_steps': 500,
            'min_radius': 1.5,
            'max_radius': 3.5
            },
)

register(
    id='sumo-ants-v0',
    entry_point='gym_compete.new_envs:SumoEnv',
    kwargs={'agent_names': ['ant_fighter', 'ant_fighter'],
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body_arena.ant_body.ant_body.xml"
            ),
            'world_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets", 'world_body_arena.xml'
            ),
            'init_pos': [(-1, 0, 2.5), (1, 0, 2.5)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5
            },
)

# register(
#     id='HumanAntArena-v0',
#     entry_point='gym_compete.new_envs:HumansKnockoutEnv',
#     kwargs={'agent_names': ['ant_fighter', 'humanoid_fighter'],
#             'scene_xml_path': os.path.join(
#                 os.path.dirname(__file__), "new_envs",
#                 "assets",
#                 "world_body_arena.ant_body.human_body.xml"
#             ),
#             'world_xml_path': os.path.join(
#                 os.path.dirname(__file__), "new_envs",
#                 "assets", 'world_body_arena.xml'
#             ),
#             'init_pos': [(-1, 0, 2.5), (1, 0, 2.5)],
#             'max_episode_steps': 500,
#             'min_radius': 2,
#             'max_radius': 3.5
#             },
# )

register(
    id='kick-and-defend-v0',
    entry_point='gym_compete.new_envs:KickAndDefend',
    kwargs={'agent_names': ['humanoid_kicker', 'humanoid_goalkeeper'], # ['humanoid_goalkeeper', 'humanoid_kicker']
            'scene_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets",
                "world_body_football.humanoid_body.humanoid_body.xml"
            ),
            'world_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets", 'world_body_football.xml'
            ),
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'max_episode_steps': 500,
            },
)

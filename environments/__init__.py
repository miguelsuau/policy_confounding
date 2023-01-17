from gym.envs.registration import register

register(
    id='tmaze-v0',
    entry_point='environments.tmaze:Tmaze',
)
register(
    id='ontrack-v0',
    entry_point='environments.ontrack:OnTrack',
)
register(
    id='keydoor-v0',
    entry_point='environments.key_door:KeyDoor',
    max_episode_steps=100
)
register(
    id='cartpole-v0',
    entry_point='environments.cartpole:CartPoleEnv',
)
register(
    id='pendulum-v0',
    entry_point='environments.pendulum:PendulumEnv',
)
from gym.envs.registration import register

register(
    id='tmaze-v0',
    entry_point='environments.tmaze:Tmaze',
)
register(
    id='keydoor-v0',
    entry_point='environments.keydoor:KeyDoor',
)
register(
    id='diversion-v0',
    entry_point='environments.diversion:Diversion',
)
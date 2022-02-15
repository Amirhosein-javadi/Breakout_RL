from gym.envs.registration import register

register(
    id='shootndodge-v0',
    entry_point='gym_shootndodge.envs:ShootNDodgeEnv',
)

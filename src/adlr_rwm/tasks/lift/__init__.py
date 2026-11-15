import gymnasium as gym

gym.register(
    id="ADLR-Lift-Cube-Franka-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.recorded_lift_env_cfg:RecordedLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg"
        },
    
)

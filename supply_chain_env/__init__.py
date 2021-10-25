from gym.envs.registration import register

register(
    id="SupplyChainTournament-v0",
    entry_point="supply_chain_env.envs:SupplyChainBotTournament",
)

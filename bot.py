"""Supply Chain Game entry point.

The game consists of four agents interacting with each other as the diagram shows:

                     orders                orders                   orders                 orders
demand             --------->             --------->              --------->              --------->
------> (RETAILER)            (WHOLESALER)           (DISTRIBUTOR)          (MANUFACTURER)
                   <---------             <---------              <---------              <---------
                    shipments             shipments                shipments               shipments

The agents form a supply chain, i.e. each agent can send a request to its neighbour and ask for a delivery of a
certain amount of goods. The neighbour does a shipment but also orders delivery from the next entity in the chain
in case of getting out of stock. The game unrolls in a turn-based fashion: all agents take decisions about the number
of items to order simultaneously, then a next turn starts.

A retailer is the first agent in the chain. It gets demand from customers and should keep them fulfilled ordering more
items from agents next in the chain.

A manufacturer is the last agent in the chain. It "orders" items from an infinite supply and ships them down the chain.

The problem is that the agents don't know the current numbers of the stock level of their partners. Also, the
order/shipment exchange doesn't happen instantaneously but involves two turns of lead time. (Except Manufacturer that
refills its supply with lead time of one turn). The same lead time is true for delivery of a previously ordered amount.
For example, if the Retailer orders X amount on n-th turn, this information reaches the Wholesaler in two "days", i.e.
on the (n+2)-th turn. Therefore, non-optimal orderings could result in stock-outs or too many items hold, and
both conditions incur costs.

Your goal is to implement a strategy for each of the four agents in such a way, that the costs are minimal after
20 game turns. It means that you should try to escape both shortages AND holding too many items in stock. Your strategy
shouldn't use any other information except stored in a dictionary that is given to `get_action()` method. Also, the
agents are not allowed to communicate their stock levels or any other internal information to each other.

In this file, you'll find a dummy implementation for each agent that orders a random amount of items each turn. If you
run this script as-is, you'll see that the costs at the end of the game are very high. Try to come up with a better
solution!
"""
import os
from argparse import ArgumentParser
from typing import Optional, List

import numpy as np
from scipy import stats

from supply_chain_env.envs.env import SupplyChainBotTournament
from supply_chain_env.leaderboard import post_score_to_api, write_result_to_file

class BaseVendor():
    def __init__(self):
        self.orders = []
        self.stock = []
    def get_action(self, step_state: dict) -> int:
        # Save Order history
        if step_state["next_incoming_order"] > 0:
            self.orders.append(step_state["next_incoming_order"])
        
        # Median Filter
        num_order = 5
        median_orders = np.median(self.orders)
        if len(self.orders) > num_order:
            median_orders = np.median(self.orders[-num_order:])

            # detect linear trend
            slope, intercept, _, _, _ = stats.linregress(list(range(len(self.orders))), self.orders)

            next_order = np.median([intercept + slope*len(self.orders), median_orders, self.orders[-1]]) + slope

        else:
            next_order = self.orders[-1] + 2

        return int(max(0, next_order))  # provide your implementation here


class Retailer(BaseVendor):
    def __init__(self):
        super().__init__()


class Wholesaler(BaseVendor):
    def __init__(self):
        super().__init__()


class Distributor(BaseVendor):
    def __init__(self):
        super().__init__()


class Manufacturer(BaseVendor):
    def __init__(self):
        super().__init__()
    def get_action(self, step_state: dict) -> int:
        # infinite supply, let's try to minimize the storage cost.
        return step_state["next_incoming_order"]


# -------------------------------------------------------------
# Game setup and utils. DO NOT MODIFY ANYTHING BELOW THIS LINE!
# -------------------------------------------------------------


def create_agents() -> List:
    """Creates a list of agents acting in the environment.

    Note that the order of agents is important here. It is always considered by the environment that the first
    agent is Retailer, the second one is Wholesaler, etc.
    """
    return [Retailer(), Wholesaler(), Distributor(), Manufacturer()]



def run() -> float:
    """Runs supply chain simulation.

    When a solution is submitted to the leaderboard, the simulation is executed several times with different (fixed)
    random seeds. The best result is used as the final score.
    """
    seeds = get_seeds()
    if seeds is None:
        # single run
        return run_game(create_agents(), verbose=True)
    total_costs = [
        run_game(create_agents(), verbose=False, seed=seed)
        for seed in seeds
    ]
    return sum(total_costs)/len(total_costs)


def run_game(
    agents: List,
    environment: str = "classical",
    verbose: bool = False,
    seed: Optional[int] = None
) -> float:
    """Performs one simulation run."""

    env = SupplyChainBotTournament(env_type=environment, seed=seed)
    state = env.reset()
    while not env.done:
        if verbose:
            env.render()
        actions = [a.get_action(state[i]) for i, a in enumerate(agents)]
        state, rewards, done, _ = env.step(actions)
    total_cost = sum(agent_state["cum_cost"] for agent_state in state)
    return total_cost


def get_seeds() -> Optional[List[int]]:
    """Returns a list of predefined random seeds when preparing a submission for the leaderboard.

    Should return None when executed locally.
    """
    string = os.environ.get("LEADERBOARD_SEEDS_STRING")
    if not string:
        return None
    try:
        seeds = [int(x) for x in string.split(",")]
        print(f"Found {len(seeds)} seeds, will do {len(seeds)} runs and average")
        return seeds
    except TypeError:
        raise RuntimeError(f"wrong seed format: {string}; should be a comma-separated list")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--user", default=None)
    return parser.parse_args()


def main(args):
    score = run()

    if args.submit:
        # get total costs and post results to leaderboard api
        write_result_to_file(score=score, filename="result.txt")
        post_score_to_api(score=score, user=args.user)


if __name__ == "__main__":
    main(parse_args())

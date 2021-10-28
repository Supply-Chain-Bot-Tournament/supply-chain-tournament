import itertools
import random
from collections import deque

import cloudpickle
import gym
import numpy as np
from gym import error
from gym.utils import seeding


def add_noise_to_init(init, noise):
    """
    Add noise to initial values.
    :type init: iterable, list or (list of lists)
    :type noise: np.array, 1-dimensional
    :rtype with_noise: np.array or (list of np.arrays)
    """
    # TODO add continuous variant
    is_init_array = all([isinstance(x, (float, int)) for x in init])

    if is_init_array:  # init is a list
        with_noise = (np.array(init) + noise).astype(int).tolist()
    else:  # init is a lists of lists
        with_noise = []
        c = 0
        for row in init:
            noise_row = np.array(row) + noise[c : (c + len(row))]
            noise_row = noise_row.astype(int).tolist()
            c += len(noise_row)
            with_noise.append(noise_row)

    return with_noise


def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len


class SupplyChainBotTournament(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env_type: str,
        seed=None,
    ):
        super().__init__()
        self.orders = (
            []
        )  # this is the decision each agent makes - how much do I need and should order?
        self.inbound_shipments = (
            []
        )  # shipments inbound at each agent (for example, for the agent Distributor, how much the Manufacturer has shipped)
        self.next_incoming_orders = []  # demand for each agent
        self.stocks = []  # current inventory level for each agent
        self.holding_cost = None
        self.stockout_cost = None
        self.cum_holding_cost = None
        self.cum_stockout_cost = None
        self.end_customer_demand = None  # end customer's demand, i.e., the customers buying goods at the retailer
        self.score_weight = (
            None  # a list of 2 lists, each of which has `n_agents` elements
        )
        self.turn = None
        self.done = True
        self.n_states_concatenated = (
            3  # number of recent turns for which the state is persisted
        )
        self.prev_states = None
        self.np_random = None

        self.n_agents = 4  # keeping 4 as default corresponding to brewery, distributor, wholesaler, retailer
        self.env_type = env_type
        if self.env_type not in ["classical", "uniform_0_2", "normal_10_4"]:
            raise NotImplementedError(
                "env_type must be in ['classical', 'uniform_0_2', 'normal_10_4']"
            )

        self.n_turns = 20
        self.add_noise_initialization = True
        self.seed(seed)

        # TODO calculate state shape
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

    def _save(self):
        """
        serialize environment to a pickle string
        :rtype: string
        """
        canned = cloudpickle.dumps(self)
        return canned

    def _load(self, pickle_string):
        """
        deserialize environment from a pickle string
        """
        self.__dict__.update(cloudpickle.loads(pickle_string).__dict__)

    def _get_observations(self):
        # these observations are in the order starting with the retailer and
        # subsequently going upstream, for example, for `n_agents` as 4, the
        # sequence is: retailer, wholesaler, distribution and manufacturer
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            observations[i] = {
                "current_stock": self.stocks[i],
                "turn": self.turn,
                "cum_cost": self.cum_holding_cost[i] + self.cum_stockout_cost[i],
                "inbound_shipments": list(self.inbound_shipments[i]),
                "orders": list(self.orders[i]),
                "next_incoming_order": self.next_incoming_orders[i],
            }
        return observations

    def _get_rewards(self):
        return -(self.holding_cost + self.stockout_cost)

    def _get_demand(self):
        return self.end_customer_demand[self.turn]

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False

        if self.env_type == "classical":
            temp_orders = [[4, 4]] * (self.n_agents - 1) + [
                [4]
            ]  # order lead time of 2 for all agents except the manufacturer
            temp_inbound_shipments = [
                [4, 4]
            ] * self.n_agents  # shipment lead time of 2 for all agents
            self.next_incoming_orders = [
                4
            ] * self.n_agents  # pending order delivery for each agent
            self.stocks = [12] * self.n_agents  # initial inventory level for each agent

            if self.add_noise_initialization:
                # noise is uniform [-2,2]
                orders_noise = (
                    np.random.choice(np.arange(5), size=get_init_len(temp_orders)) - 2
                )
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                inbound_shipments_noise = (
                    np.random.choice(
                        np.arange(5), size=get_init_len(temp_inbound_shipments)
                    )
                    - 2
                )
                temp_inbound_shipments = add_noise_to_init(
                    temp_inbound_shipments, inbound_shipments_noise
                )

                last_incoming_orders_noise = (
                    np.random.choice(
                        np.arange(5), size=get_init_len(self.next_incoming_orders)
                    )
                    - 2
                )
                self.next_incoming_orders = add_noise_to_init(
                    self.next_incoming_orders, last_incoming_orders_noise
                )

                stocks_noise = (
                    np.random.choice(np.arange(13), size=get_init_len(self.stocks)) - 6
                )
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            self.end_customer_demand = [4] * 4 + [8] * (self.n_turns - 4)
            self.score_weight = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.env_type == "uniform_0_2":
            temp_orders = [[1, 1]] * (self.n_agents - 1) + [[1]]
            temp_inbound_shipments = [[1, 1]] * self.n_agents
            self.next_incoming_orders = [1] * self.n_agents
            self.stocks = [4] * self.n_agents

            if self.add_noise_initialization:
                # noise is uniform [-1,1]
                orders_noise = (
                    np.random.choice(np.arange(3), size=get_init_len(temp_orders)) - 1
                )
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                inbound_shipments_noise = (
                    np.random.choice(
                        np.arange(3), size=get_init_len(temp_inbound_shipments)
                    )
                    - 1
                )
                temp_inbound_shipments = add_noise_to_init(
                    temp_inbound_shipments, inbound_shipments_noise
                )

                last_incoming_orders_noise = (
                    np.random.choice(
                        np.arange(3), size=get_init_len(self.next_incoming_orders)
                    )
                    - 1
                )
                self.next_incoming_orders = add_noise_to_init(
                    self.next_incoming_orders, last_incoming_orders_noise
                )

                stocks_noise = (
                    np.random.choice(np.arange(5), size=get_init_len(self.stocks)) - 2
                )
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            # uniform [0, 2]
            self.end_customer_demand = self.np_random.uniform(
                low=0, high=3, size=self.n_turns
            ).astype(np.int)
            self.score_weight = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.env_type == "normal_10_4":
            temp_orders = [[10, 10]] * (self.n_agents - 1) + [[10]]
            temp_inbound_shipments = [[10, 10]] * self.n_agents
            self.next_incoming_orders = [10] * self.n_agents
            self.stocks = [40] * self.n_agents

            if self.add_noise_initialization:
                # noise is uniform [-1,1]
                orders_noise = np.random.normal(
                    loc=0, scale=5, size=get_init_len(temp_orders)
                )
                orders_noise = np.clip(
                    orders_noise, -10, 10
                )  # clip to prevent negative orders
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                inbound_shipments_noise = np.random.normal(
                    loc=0, scale=5, size=get_init_len(temp_inbound_shipments)
                )
                inbound_shipments_noise = np.clip(
                    inbound_shipments_noise, -10, 10
                )  # clip to prevent negative inbound shipments
                temp_inbound_shipments = add_noise_to_init(
                    temp_inbound_shipments, inbound_shipments_noise
                )

                last_incoming_orders_noise = np.random.normal(
                    loc=0, scale=5, size=get_init_len(self.next_incoming_orders)
                )
                last_incoming_orders_noise = np.clip(
                    last_incoming_orders_noise, -10, 10
                )
                self.next_incoming_orders = add_noise_to_init(
                    self.next_incoming_orders, last_incoming_orders_noise
                )

                stocks_noise = np.random.normal(
                    loc=0, scale=4, size=get_init_len(self.stocks)
                )
                stocks_noise = np.clip(stocks_noise, -10, 10)
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            self.end_customer_demand = self.np_random.normal(
                loc=10, scale=4, size=self.n_turns
            )
            self.end_customer_demand = np.clip(self.turns, 0, 1000).astype(np.int)
            # dqn paper page 24
            self.score_weight = [
                [1.0, 0.75, 0.5, 0.25] * self.n_agents,
                [10.0] + [0.0] * (self.n_agents - 1),
            ]

        else:
            raise NotImplementedError(
                f"Environment type {self.env_type} is not implemented yet."
            )

        # initialize other variables
        self.cum_holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.orders = [deque(x) for x in temp_orders]
        self.inbound_shipments = [deque(x) for x in temp_inbound_shipments]
        self.turn = 0

        temp_obs = [None] * self.n_agents
        for i in range(self.n_agents):
            temp_obs[i] = {
                "current_stock": self.stocks[i],
                "turn": self.turn,
                "cum_cost": self.cum_holding_cost[i] + self.cum_stockout_cost[i],
                "inbound_shipments": list(self.inbound_shipments[i]),
                "orders": list(self.orders[i])[::-1],
                "next_incoming_order": self.next_incoming_orders[i],
            }
        prev_state = temp_obs
        self.prev_states = deque([prev_state] * (self.n_states_concatenated - 1))
        return self._get_observations()

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} is not implemented yet")
        print("\n" + "=" * 50)
        print("Turn #: ", self.turn + 1)
        print(
            "Stocks Levels (at the end of the turn): ",
            ", ".join([str(x) for x in self.stocks]),
        )
        print("Orders Placed: ", [list(x) for x in self.orders])
        print("Shipments Inbound: ", [list(x) for x in self.inbound_shipments])
        print("Next Incoming Orders: ", self.next_incoming_orders)
        print("Cumulative holding cost: ", self.cum_holding_cost)
        print("Cumulative stockout cost: ", self.cum_stockout_cost)
        print("Last holding cost: ", self.holding_cost)
        print("Last stockout cost: ", self.stockout_cost)

    def step(self, action: list):
        # sanity checks
        if self.done:
            raise error.ResetNeeded(
                "Environment is finished, please run env.reset() before taking actions"
            )
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(
                f"Length of action array must be same as n_agents({self.n_agents})"
            )
        if any(np.array(action) < 0):
            raise error.InvalidAction(
                f"You can't order negative amount. You agents actions are: {action}"
            )

        # concatenate previous states, self.prev_states in an queue of previous states
        self.prev_states.popleft()
        self.prev_states.append(self._get_observations())
        # make incoming step
        demand = self._get_demand()
        orders_inc = [order.popleft() for order in self.orders]
        self.next_incoming_orders = [demand] + orders_inc[
            :-1
        ]  # what's the demand for each agent
        ship_inc = [shipment.popleft() for shipment in self.inbound_shipments]
        # calculate inbound shipments respecting orders and stock levels
        for i in range(
            self.n_agents - 1
        ):  # manufacturer is assumed to have no constraints
            max_possible_shipment = (
                max(0, self.stocks[i + 1]) + ship_inc[i + 1]
            )  # stock + incoming shipment
            order = orders_inc[i] + max(
                0, -self.stocks[i + 1]
            )  # incoming order + stockout (backorder)
            shipment = min(order, max_possible_shipment)
            self.inbound_shipments[i].append(shipment)
        self.inbound_shipments[-1].append(orders_inc[-1])
        # update stocks
        self.stocks = [(stock + inc) for stock, inc in zip(self.stocks, ship_inc)]
        for i in range(1, self.n_agents):
            self.stocks[i] -= orders_inc[i - 1]
        self.stocks[0] -= demand  # for the retailer
        # update orders
        for i in range(self.n_agents):
            self.orders[i].append(action[i])
        self.next_incoming_orders = [self._get_demand()] + [
            x[0] for x in self.orders[:-1]
        ]

        # calculate costs
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        for i in range(self.n_agents):
            if self.stocks[i] >= 0:
                self.holding_cost[i] = (
                    max(0, self.stocks[i]) * self.score_weight[0][i]
                )  # only applicable when stocks > 0
            else:
                self.stockout_cost[i] = (
                    -min(0, self.stocks[i]) * self.score_weight[1][i]
                )  # only applicable when stocks < 0
        self.cum_holding_cost += self.holding_cost
        self.cum_stockout_cost += self.stockout_cost
        # calculate reward
        rewards = self._get_rewards()

        # check if done
        if self.turn == self.n_turns - 1:
            # print(
            #     f"\nTotal cost is: EUR {sum(self.cum_holding_cost + self.cum_stockout_cost)}"
            # )
            self.done = True
        else:
            self.turn += 1
        state = self._get_observations()
        # todo flatten observation dict
        return state, rewards, self.done, {}

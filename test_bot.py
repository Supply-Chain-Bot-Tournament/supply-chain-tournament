from bot import Retailer, Wholesaler, Distributor, Manufacturer, create_agents, run_game

N_AGENTS = 4
N_TURNS = 20


def test_number_of_agents():
    assert len(create_agents()) == N_AGENTS


def test_each_agent_has_its_own_class():
    retailer, wholesaler, distributor, manufacturer = create_agents()

    assert isinstance(retailer, Retailer)
    assert isinstance(wholesaler, Wholesaler)
    assert isinstance(distributor, Distributor)
    assert isinstance(manufacturer, Manufacturer)


def test_random_agents_take_actions_during_the_game(monkeypatch):
    n_calls = 0

    def get_action(_, state: dict):
        nonlocal n_calls
        n_calls += 1
        return 4

    monkeypatch.setattr("bot.Retailer.get_action", get_action)
    monkeypatch.setattr("bot.Wholesaler.get_action", get_action)
    monkeypatch.setattr("bot.Distributor.get_action", get_action)
    monkeypatch.setattr("bot.Manufacturer.get_action", get_action)

    _ = run_game(create_agents())

    assert n_calls == N_AGENTS * N_TURNS

from my_types import ColorCount, Key, Color, Zone, n_colors, n_zones
from typing import NamedTuple
from jax import random, jit, lax, numpy as np
from functools import partial


class Game(NamedTuple):
    state: ColorCount  # how many colors are in each zone
    finished_deck_once: bool  # whether or not we've cycled through all the cards


def init_game(key: Key) -> Game:
    """
    Sets up a game.
    """
    # make new game with a fresh deck
    game_state = np.zeros((n_zones, n_colors))
    game_state = game_state.at[Zone.DECK].set(np.tile(18, n_colors))
    game = Game(game_state, False)
    # put two cards in each cup
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(game, Zone.P1_CUP, 2, key1)
    game = move_from_deck(game, Zone.P2_CUP, 2, key2)
    # place two cards in each mountain
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(game, Zone.M1_MOUNTAIN, 2, key1)
    game = move_from_deck(game, Zone.M2_MOUNTAIN, 2, key2)
    # each player draws 6 cards
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(game, Zone.P1_HAND, 6, key1)
    game = move_from_deck(game, Zone.P2_HAND, 6, key2)
    return game


@jit
def move_discard_to_deck(game: Game) -> Game:
    """
    Move discard pile into deck.
    """
    game_state = game.state
    game_state = game_state.at[Zone.DECK].set(game_state[Zone.DISCARD])
    game_state = game_state.at[Zone.DISCARD].set(np.zeros(n_colors))
    game = Game(game_state, True)
    return game


@jit
def move_one_card_from_deck(zone: Zone, game: Game, key: Key) -> tuple[Game, None]:
    """
    Move one card from the deck to another zone.
    """
    game = lax.cond(np.sum(game.state[Zone.DECK]) == 0,
                    lambda _: move_discard_to_deck(game),
                    lambda _: game,
                    None)
    game_state = game.state
    color = random.choice(key, n_colors, p=game.state[Zone.DECK])
    game_state = game_state.at[Zone.DECK, color].add(-1)
    game_state = game_state.at[zone, color].add(1)
    return Game(game_state, game.finished_deck_once), None


def move_from_deck(game: Game, zone: Zone, n_cards: int, key: Key) -> Game:
    """
    Move cards from the deck into another zone.
    """
    keys = random.split(key, n_cards)
    game, _ = lax.scan(partial(move_one_card_from_deck, zone), game, keys)
    return game


@jit
def move_from_zone(game: Game, zone_1: Zone, zone_2: Zone, color: Color, n_cards: int) -> Game:
    """
    Move cards from one zone into another zone.
    Does not check for legality of action.
    """
    game_state = game.state
    game_state = game_state.at[zone_1, color].add(-n_cards)
    game_state = game_state.at[zone_2, color].add(n_cards)
    return Game(game_state, game.finished_deck_once)


@jit
def move_all_from_zone(game: Game, zone_1: Zone, zone_2: Zone) -> Game:
    """
    Move all cards from one zone into another zone.
    """
    game_state = game.state
    game_state = game_state.at[zone_2].set(game_state[zone_1] + game_state[zone_2])
    game_state = game_state.at[zone_1].set(np.zeros(n_colors))
    return Game(game_state, game.finished_deck_once)

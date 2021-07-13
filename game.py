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
    game_state = np.zeros((n_zones, n_colors), dtype=np.int32)
    game_state = game_state.at[Zone.DECK].set(np.tile(18, n_colors))
    game = Game(game_state, False)
    # put two cards in each cup
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(2, game, Zone.P1_CUP, key1)
    game = move_from_deck(2, game, Zone.P2_CUP, key2)
    # place two cards in each mountain
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(2, game, Zone.M1_MOUNTAIN, key1)
    game = move_from_deck(2, game, Zone.M2_MOUNTAIN, key2)
    # each player draws 6 cards
    key, key1, key2 = random.split(key, 3)
    game = move_from_deck(6, game, Zone.P1_HAND, key1)
    game = move_from_deck(6, game, Zone.P2_HAND, key2)
    return game


# @jit
def move_discard_to_deck(game: Game) -> Game:
    """
    Move discard pile into deck.
    """
    game_state = game.state
    game_state = game_state.at[Zone.DECK].set(game_state[Zone.DISCARD])
    game_state = game_state.at[Zone.DISCARD].set(np.zeros(n_colors))
    game = Game(game_state, True)
    return game


def move_discard_then_move_from_deck(game: Game, zone: Zone, n_cards: int, key: Key) -> Game:
    """
    Getting around recursion problems in Jax.
    """
    import pdb; pdb.set_trace()
    game = move_discard_to_deck(game)
    color = random.choice(key, n_colors, shape=[n_cards], p=game.state[Zone.DECK])
    game_state = game.state
    game_state = game_state.at[Zone.DECK, color].add(-n_cards)
    game_state = game_state.at[zone, color].add(n_cards)
    game = Game(game_state, game.finished_deck_once)
    return game


# @partial(jit, static_argnums=[0])
def move_from_deck(n_cards: int, game: Game, zone: Zone, key: Key) -> Game:
    """
    Move cards from the deck into another zone.
    """
    n_cards_to_move = np.minimum(n_cards, np.sum(game.state[Zone.DECK]))
    n_cards_remaining_to_move = n_cards - n_cards_to_move
    key1, key2 = random.split(key, 2)
    color = random.choice(key1, n_colors, shape=[n_cards_to_move], p=game.state[Zone.DECK])
    game_state = game.state
    game_state = game_state.at[Zone.DECK, color].add(-n_cards_to_move)
    game_state = game_state.at[zone, color].add(n_cards_to_move)
    game = Game(game_state, game.finished_deck_once)
    game = lax.cond(n_cards_remaining_to_move > 0,
                    lambda _: move_discard_then_move_from_deck(game, zone, n_cards_remaining_to_move, key2),
                    lambda _: game,
                    None)
    return game


# @jit
def move_from_zone(game: Game, zone_1: Zone, zone_2: Zone, color: Color, n_cards: int) -> Game:
    """
    Move cards from one zone into another zone.
    Does not check for legality of action.
    """
    game_state = game.state
    game_state = game_state.at[zone_1, color].add(-n_cards)
    game_state = game_state.at[zone_2, color].add(n_cards)
    return Game(game_state, game.finished_deck_once)


# @jit
def move_all_from_zone(game: Game, zone_1: Zone, zone_2: Zone) -> Game:
    """
    Move all cards from one zone into another zone.
    """
    game_state = game.state
    game_state = game_state.at[zone_2].set(game_state[zone_1] + game_state[zone_2])
    game_state = game_state.at[zone_1].set(np.zeros(n_colors))
    return Game(game_state, game.finished_deck_once)

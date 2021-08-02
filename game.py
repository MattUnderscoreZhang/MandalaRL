from my_types import ColorCount, DeckOrder, Color, Zone, Key, N_COLORS, N_ZONES, N_CARDS
from typing import NamedTuple
from jax import numpy as np
from jax import jit, random
from jax.lax import map, while_loop, cond
from jax.random import PRNGKey
from functools import partial


class Game(NamedTuple):
    color_counts: ColorCount  # how many colors are in each zone
    deck_order: DeckOrder
    deck_index: int
    finished_deck_once: bool  # whether or not we've cycled through all the cards


def init_game() -> Game:
    """
    Sets up a game.
    """
    seed = 0
    key = PRNGKey(seed)
    # make new game with a fresh deck
    color_counts = np.zeros((N_ZONES, N_COLORS))
    color_counts = color_counts.at[Zone.DECK].set(np.tile(18, N_COLORS))
    key, key1 = random.split(key, 2)
    deck_order = shuffle(color_counts[Zone.DECK], key1)
    game = Game(color_counts, deck_order, N_CARDS, False)
    # put two cards in each cup
    key, key1, key2 = random.split(key, 3)
    game = move_cards_from_deck(game, Zone.P1_CUP, 2, key1)
    game = move_cards_from_deck(game, Zone.P2_CUP, 2, key2)
    # # place two cards in each mountain
    # game = move_cards_from_deck(game, Zone.M1_MOUNTAIN, 2)
    # game = move_cards_from_deck(game, Zone.M2_MOUNTAIN, 2)
    # # each player draws 6 cards
    # game = move_cards_from_deck(game, Zone.P1_HAND, 6)
    # game = move_cards_from_deck(game, Zone.P2_HAND, 6)
    # return game


@jit
def shuffle(color_counts: ColorCount, key: Key) -> DeckOrder:
    deck_color_indices = np.array([np.sum(color_counts[:i]) for i in range(N_COLORS + 1)])
    deck_order = np.stack([np.sum(map(lambda j: i >= j, deck_color_indices)) - 1 for i in range(N_CARDS)])
    deck_order = random.permutation(key, deck_order)  # 6 indicates out of range
    return deck_order


# @jit
def move_cards_from_deck(game: Game, zone: Zone, n_cards: int, key: Key) -> Game:
    """
    Move cards from the deck into another zone.
    """
    n_cards_to_move = np.minimum(n_cards, np.sum(game.color_counts[Zone.DECK]))

    def move_card_from_deck(loop_data):
        i, game, key = loop_data
        color_counts = game.color_counts
        color = game.deck_order[game.deck_index]
        color_counts = color_counts.at[Zone.DECK, color].add(-1)
        color_counts = color_counts.at[zone, color].add(1)
        game = Game(color_counts, game.deck_order, game.deck_index, game.finished_deck_once)
        def true_fn(true_loop_data):
            game, key = true_loop_data
            key, key1 = random.split(key, 2)
            game = move_discard_to_deck(game, key1)
            return game, key
        false_fn = lambda game: Game(game.color_counts, game.deck_order, game.deck_index - 1, game.finished_deck_once)
        game, key = cond(game.deck_index == 0, true_fn, false_fn, (game, key))
        return [i + 1, game, key]

    # i = np.constant(0)
    while_condition = lambda loop_data: loop_data[0] < n_cards_to_move
    i = 0
    i, game, key = while_loop(while_condition, move_card_from_deck, [i, game, key])
    import pdb; pdb.set_trace()
    return game


@jit
def move_discard_to_deck(game: Game, key: Key) -> Game:
    """
    Move discard pile into deck.
    """
    color_counts = game.color_counts
    assert(np.sum(color_counts[Zone.DECK]) == 0)
    color_counts = color_counts.at[Zone.DECK].update(color_counts[Zone.DISCARD])
    color_counts = color_counts.at[Zone.DISCARD].update(np.tile(0, N_COLORS))
    deck_order = shuffle(color_counts[Zone.DECK], key)
    deck_index = np.sum(color_counts[Zone.DISCARD]) - 1
    return Game(color_counts, deck_order, deck_index, True)












@jit
def move_from_zone(game: Game, zone_1: Zone, zone_2: Zone, color: Color, n_cards: int) -> Game:
    """
    Move cards from one zone into another zone.
    Does not check for legality of action.
    """
    color_counts = game.color_counts
    color_counts = color_counts.at[zone_1, color].add(-n_cards)
    color_counts = color_counts.at[zone_2, color].add(n_cards)
    return Game(color_counts, game.finished_deck_once)


@jit
def move_all_from_zone(game: Game, zone_1: Zone, zone_2: Zone) -> Game:
    """
    Move all cards from one zone into another zone.
    """
    color_counts = game.color_counts
    color_counts = color_counts.at[zone_2].set(color_counts[zone_1] + color_counts[zone_2])
    color_counts = color_counts.at[zone_1].set(np.zeros([N_COLORS]))
    return Game(color_counts, game.finished_deck_once)

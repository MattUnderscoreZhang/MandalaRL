from my_types import ColorCount, CardOrdering, Key, Color, Zone
from typing import NamedTuple
from jax import random, jit, numpy as np
from jax.ops import index, index_add, index_update


class Game(NamedTuple):
    state: ColorCount  # how many colors are in each zone
    deck_ordering: CardOrdering  # ordered list of cards
    finished_deck_once: bool  # whether or not we've cycled through all the cards


def init_game(key: Key) -> Game:
    """
    Sets up a game.
    """
    game_state = np.zeros((len(Zone), len(Color)))
    game_state = index_update(game_state, index[Zone.DECK], np.array([18] * len(Color)))
    # put two cards in each cup
    # place two cards in each mountain
    deck_ordering = shuffle_deck(game_state[Zone.DECK], key)
    game = Game(game_state, deck_ordering, False)
    key, key1, key2 = random.split(key, 3)
    game = place_from_deck(game, Zone.P1_HAND, 6, key1)
    game = place_from_deck(game, Zone.P2_HAND, 6, key2)
    return game


def shuffle_deck(cards: ColorCount, key: Key) -> CardOrdering:
    """
    Takes a ColorCount array and returns a randomized array of colors.
    """
    deck_ordering = [[color] * int(cards[color]) for color in Color]
    deck_ordering = np.array([i for j in deck_ordering for i in j])
    deck_ordering = random.shuffle(key, deck_ordering)
    return deck_ordering


def shuffle_discard_into_deck(game: Game, key: Key) -> Game:
    """
    Shuffle discard pile into deck.
    """
    game_state = game.state
    game_state = index_update(game_state, index[Zone.DECK], game_state[Zone.DISCARD])
    game_state = index_update(game_state, index[Zone.DISCARD], np.zeros(len(Color)))
    deck_ordering = shuffle_deck(game_state[Zone.DECK], key)
    game = Game(game_state, deck_ordering, True)
    return game


def place_from_deck(game: Game, zone: Zone, n_cards: int, key: Key) -> Game:
    """
    Move cards from the deck into another zone.
    """
    game_state = game.state
    deck_ordering = game.deck_ordering
    for _ in range(n_cards):
        if len(deck_ordering) == 0:
            key, key1 = random.split(key, 2)
            game = Game(game_state, deck_ordering, game.finished_deck_once)
            game = shuffle_discard_into_deck(game, key1)
            game_state = game.state
            deck_ordering = game.deck_ordering
        color = deck_ordering[0]
        game_state = index_add(game_state, index[zone, color], 1)
        game_state = index_add(game_state, index[Zone.DECK, color], -1)
        deck_ordering = deck_ordering[1:]
    return Game(game_state, deck_ordering, game.finished_deck_once)

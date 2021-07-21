from my_types import ColorCount, DeckOrder, Key, Color, Zone, N_COLORS, N_ZONES
from typing import NamedTuple
import tensorflow as tf
import numpy as np
from functools import partial


class Game(NamedTuple):
    state: ColorCount  # how many colors are in each zone
    deck_order: DeckOrder
    finished_deck_once: bool  # whether or not we've cycled through all the cards


def init_game(seed: tf.Tensor) -> Game:
    """
    Sets up a game.
    """
    # make new game with a fresh deck
    game_state = tf.zeros([N_ZONES, N_COLORS], dtype=tf.dtypes.int32)
    game_state = tf.tensor_scatter_nd_update(game_state, [[Zone.DECK]], [tf.tile([18], [N_COLORS])])
    deck_order = shuffle(game_state[Zone.DECK])
    game = Game(game_state, deck_order, False)
    # put two cards in each cup
    # game = move_from_deck(2, game, Zone.P1_CUP, seed)
    # game = move_from_deck(2, game, Zone.P2_CUP, seed)
    # # place two cards in each mountain
    # game = move_from_deck(2, game, Zone.M1_MOUNTAIN, seed)
    # game = move_from_deck(2, game, Zone.M2_MOUNTAIN, seed)
    # # each player draws 6 cards
    # game = move_from_deck(6, game, Zone.P1_HAND, seed)
    # game = move_from_deck(6, game, Zone.P2_HAND, seed)
    return game


def shuffle(deck_colors: ColorCount) -> DeckOrder:
    card_colors = np.array([np.tile([i], n_cards) for i, n_cards in enumerate(deck_colors)])
    card_colors = tf.reshape(card_colors, [1, -1])
    card_colors = tf.cast(card_colors, dtype="int64")
    n_cards = card_colors.shape[1]
    deck_indices, _, _ = tf.random.fixed_unigram_candidate_sampler(
        true_classes=card_colors,
        num_true=n_cards,
        num_sampled=n_cards,
        unique=True,
        range_max=n_cards,
        unigrams=[1]*n_cards
        )
    deck_order = tf.gather(card_colors[0], deck_indices)
    return deck_order


@tf.function
def move_discard_to_deck(game: Game) -> Game:
    """
    Move discard pile into deck.
    """
    game_state = game.state
    game_state = game_state.at[Zone.DECK].set(game_state[Zone.DISCARD])
    game_state = game_state.at[Zone.DISCARD].set(tf.zeros([N_COLORS]))
    game = Game(game_state, True)
    return game


@tf.function
def move_discard_then_move_from_deck(game: Game, zone: Zone, n_cards: int, seed: tf.Tensor) -> Game:
    """
    Getting around recursion problems in Jax.
    """
    game = move_discard_to_deck(game)
    color = tf.random.fixed_unigram_candidate_sampler(seed, N_COLORS, shape=[n_cards], p=game.state[Zone.DECK])
    game_state = game.state
    game_state = game_state.at[Zone.DECK, color].add(-n_cards)
    game_state = game_state.at[zone, color].add(n_cards)
    game = Game(game_state, game.finished_deck_once)
    return game


@tf.function
def move_card_from_deck(i, seed, game):
    color = tf.random.fixed_unigram_candidate_sampler(seed, N_COLORS, shape=[1], p=game.state[Zone.DECK])
    game_state = game.state
    game_state = game_state.at[Zone.DECK, color].add(-1)
    game_state = game_state.at[zone, color].add(1)
    game = Game(game_state, game.finished_deck_once)
    return (seed, game)


# @partial(tf.function, static_argnums=[0])
@tf.function
def move_from_deck(n_cards: int, game: Game, zone: Zone, seed: tf.Tensor) -> Game:
    """
    Move cards from the deck into another zone.
    """
    n_cards_to_move = np.minimum(n_cards, np.sum(game.state[Zone.DECK]))
    lax.fori_loop(0, n_cards_to_move, move_card_from_deck, [seed, game])
    game = lax.cond(n_cards > n_cards_to_move,
                    lambda _: move_discard_then_move_from_deck(game, zone, n_cards - n_cards_to_move, seed),
                    lambda _: game,
                    None)
    return game


@tf.function
def move_from_zone(game: Game, zone_1: Zone, zone_2: Zone, color: Color, n_cards: int) -> Game:
    """
    Move cards from one zone into another zone.
    Does not check for legality of action.
    """
    game_state = game.state
    game_state = game_state.at[zone_1, color].add(-n_cards)
    game_state = game_state.at[zone_2, color].add(n_cards)
    return Game(game_state, game.finished_deck_once)


@tf.function
def move_all_from_zone(game: Game, zone_1: Zone, zone_2: Zone) -> Game:
    """
    Move all cards from one zone into another zone.
    """
    game_state = game.state
    game_state = game_state.at[zone_2].set(game_state[zone_1] + game_state[zone_2])
    game_state = game_state.at[zone_1].set(tf.zeros([N_COLORS]))
    return Game(game_state, game.finished_deck_once)

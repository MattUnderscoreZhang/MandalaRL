from my_types import ColorCount, DeckOrder, Color, Zone, N_COLORS, N_ZONES
from typing import NamedTuple
import tensorflow as tf
import numpy as np
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
    # make new game with a fresh deck
    color_counts = tf.zeros([N_ZONES, N_COLORS], dtype=tf.dtypes.int32)
    color_counts = tf.tensor_scatter_nd_update(color_counts, [[Zone.DECK]], [tf.tile([18], [N_COLORS])])
    deck_order = shuffle(color_counts[Zone.DECK])
    deck_index = len(deck_order) - 1
    game = Game(color_counts, deck_order, deck_index, False)
    # put two cards in each cup
    game = move_cards_from_deck(game, Zone.P1_CUP, 2)
    game = move_cards_from_deck(game, Zone.P2_CUP, 2)
    # place two cards in each mountain
    game = move_cards_from_deck(game, Zone.M1_MOUNTAIN, 2)
    game = move_cards_from_deck(game, Zone.M2_MOUNTAIN, 2)
    # each player draws 6 cards
    game = move_cards_from_deck(game, Zone.P1_HAND, 6)
    game = move_cards_from_deck(game, Zone.P2_HAND, 6)
    return game


@tf.function
def shuffle(deck_colors: ColorCount) -> DeckOrder:
    """
    Given a set of counts for the six colors, return a shuffled list with that many entries of each color.
    """
    deck_colors_i = tf.expand_dims(tf.range(tf.shape(deck_colors)[0], dtype=tf.int32), 1)
    deck_order, _ = tf.map_fn(lambda i: (tf.tile(i[0], [i[1]]), i[0]), (deck_colors_i, deck_colors), dtype=(tf.int32, tf.int32))
    deck_order = tf.reshape(deck_order, [1, -1])[0]
    deck_order = tf.cast(deck_order, dtype=tf.int64)
    deck_order = tf.random.shuffle(deck_order)
    return deck_order


# @tf.function
def move_cards_from_deck(game: Game, zone: Zone, n_cards: int) -> Game:
    """
    Move cards from the deck into another zone.
    """
    n_cards_to_move = tf.minimum(n_cards, tf.math.reduce_sum(game.color_counts[Zone.DECK]))
    i = tf.constant(0)
    while_condition = lambda i, _: tf.less(i, n_cards_to_move)

    def move_card_from_deck(i, game):
        color = game.deck_order[game.deck_index]
        def true_fn() -> Game:  # no cards left - shuffle discard pile into deck
            return move_discard_to_deck(game)
        def false_fn() -> Game:  # move the deck index
            deck_index = game.deck_index - 1
            return Game(game.color_counts, game.deck_order, deck_index, game.finished_deck_once)
        game = tf.cond(game.deck_index==0, true_fn=true_fn, false_fn=false_fn)
        color_counts = game.color_counts
        color_counts = tf.tensor_scatter_nd_add(color_counts, [[Zone.DECK, color]], [-1])
        color_counts = tf.tensor_scatter_nd_add(color_counts, [[zone, color]], [1])
        game = Game(color_counts, game.deck_order, game.deck_index, game.finished_deck_once)
        return [tf.add(i, 1), game]

    i, game = tf.while_loop(while_condition, move_card_from_deck, [i, game])
    return game


@tf.function
def move_discard_to_deck(game: Game) -> Game:
    """
    Move discard pile into deck.
    """
    color_counts = game.color_counts
    color_counts = tf.tensor_scatter_nd_update(color_counts, [[Zone.DECK]], [color_counts[Zone.DISCARD]])
    color_counts = tf.tensor_scatter_nd_update(color_counts, [[Zone.DISCARD]], [tf.tile([0], [N_COLORS])])
    deck_order = shuffle(game.color_counts[Zone.DECK])
    deck_index = tf.reduce_sum(game.color_counts[Zone.DISCARD]) - 1
    return Game(color_counts, deck_order, deck_index, True)












@tf.function
def move_discard_then_move_from_deck(game: Game, zone: Zone, n_cards: int) -> Game:
    """
    Getting around recursion problems in Jax.
    """
    game = move_discard_to_deck(game)
    color = tf.random.fixed_unigram_candidate_sampler(N_COLORS, shape=[n_cards], p=game.color_counts[Zone.DECK])
    color_counts = game.color_counts
    color_counts = color_counts.at[Zone.DECK, color].add(-n_cards)
    color_counts = color_counts.at[zone, color].add(n_cards)
    game = Game(color_counts, game.finished_deck_once)
    return game


@tf.function
def move_from_zone(game: Game, zone_1: Zone, zone_2: Zone, color: Color, n_cards: int) -> Game:
    """
    Move cards from one zone into another zone.
    Does not check for legality of action.
    """
    color_counts = game.color_counts
    color_counts = color_counts.at[zone_1, color].add(-n_cards)
    color_counts = color_counts.at[zone_2, color].add(n_cards)
    return Game(color_counts, game.finished_deck_once)


@tf.function
def move_all_from_zone(game: Game, zone_1: Zone, zone_2: Zone) -> Game:
    """
    Move all cards from one zone into another zone.
    """
    color_counts = game.color_counts
    color_counts = color_counts.at[zone_2].set(color_counts[zone_1] + color_counts[zone_2])
    color_counts = color_counts.at[zone_1].set(tf.zeros([N_COLORS]))
    return Game(color_counts, game.finished_deck_once)

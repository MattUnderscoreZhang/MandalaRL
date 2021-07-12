from my_types import Color, Zone
import unittest
import game
from jax import random, numpy as np
from jax.random import PRNGKey
from jax.ops import index, index_update


class GameTests(unittest.TestCase):
    def setUp(self):
        seed = 0
        self.key = PRNGKey(seed)
        self.key, key1 = random.split(self.key, 2)
        self.my_game = game.init_game(key1)


    def test_enums(self):
        assert(len(Color) == 6)
        assert(len(Zone) == 14)


    def test_shuffle(self):
        self.key, key1 = random.split(self.key, 2)
        deck_ordering = game.shuffle_deck(self.my_game.state[Zone.DECK], key1)
        color_counts = np.sum(self.my_game.state, axis=0)
        assert(len(deck_ordering) == 96)
        assert(all(color_counts == np.array([18] * 6)))


    def test_shuffle_discard_into_deck(self):
        self.key, key1 = random.split(self.key, 2)
        game_state = self.my_game.state
        game_state = index_update(game_state, index[Zone.DISCARD], game_state[Zone.DECK])
        game_state = index_update(game_state, index[Zone.DECK], np.zeros(len(Color)))
        test_game = game.Game(game_state, np.array([]), self.my_game.finished_deck_once)
        test_game = game.shuffle_discard_into_deck(test_game, key1)
        color_counts = np.sum(test_game.state, axis=0)
        assert(len(test_game.deck_ordering) == 96)
        assert(all(color_counts == np.array([18] * 6)))


    def test_draw(self):
        self.key, key1, key2 = random.split(self.key, 3)
        test_game = self.my_game
        test_game = game.place_from_deck(test_game, Zone.P1_HAND, 10, key1)
        test_game = game.place_from_deck(test_game, Zone.P2_HAND, 5, key2)
        state = test_game.state
        deck_ordering = test_game.deck_ordering
        assert(len(deck_ordering) == 81)
        assert(np.sum(state[Zone.DECK]) == 81)
        assert(np.sum(state[Zone.P1_HAND]) == 16)
        assert(np.sum(state[Zone.P2_HAND]) == 11)


    def test_draw_empty_deck(self):
        # place all but 30 cards from the deck into the discard pile
        colors, counts = np.unique(self.my_game.deck_ordering, return_counts=True)
        original_color_counts = [i[1] for i in sorted(zip(colors, counts))]
        remaining_deck_ordering = self.my_game.deck_ordering[-30:]
        colors, counts = np.unique(remaining_deck_ordering, return_counts=True)
        deck_color_counts = [i[1] for i in sorted(zip(colors, counts))]
        discard_color_counts = [i-j for i, j in zip(original_color_counts, deck_color_counts)]
        game_state = self.my_game.state
        game_state = index_update(game_state, index[Zone.DECK, :], np.array(deck_color_counts))
        game_state = index_update(game_state, index[Zone.DISCARD, :], np.array(discard_color_counts))
        test_game = game.Game(game_state, remaining_deck_ordering, self.my_game.finished_deck_once)
        # have each player draw 20 cards
        self.key, key1, key2 = random.split(self.key, 3)
        test_game = game.place_from_deck(test_game, Zone.P1_HAND, 20, key1)
        test_game = game.place_from_deck(test_game, Zone.P2_HAND, 20, key2)
        # check states
        state = test_game.state
        deck_ordering = test_game.deck_ordering
        assert(len(deck_ordering) == 56)
        assert(np.sum(state[Zone.DECK]) == 56)
        assert(np.sum(state[Zone.DISCARD]) == 0)
        assert(np.sum(state[Zone.P1_HAND]) == 26)
        assert(np.sum(state[Zone.P2_HAND]) == 26)
        assert(np.sum(state) == 108)

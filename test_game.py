from my_types import Color, Zone, n_colors
import unittest
import game
from jax import random, numpy as np
from jax.random import PRNGKey 
from jax.ops import index

class GameTests(unittest.TestCase):
    def setUp(self):
        seed = 0
        self.key = PRNGKey(seed)
        self.key, key1 = random.split(self.key, 2)
        self.my_game = game.init_game(key1)


    def assert_state_validity(self, state):
        color_counts = np.sum(state, axis=0)
        assert(np.sum(state) == 108)
        assert(all(color_counts == np.tile(18, 6)))


    def test_enums(self):
        assert(len(Color) == 6)
        assert(len(Zone) == 14)


    def test_move_discard_into_deck(self):
        state = self.my_game.state
        state = state.at[index[Zone.DISCARD]].set(state[Zone.DECK])
        state = state.at[index[Zone.DECK]].set(np.zeros(n_colors))
        test_game = game.Game(state, self.my_game.finished_deck_once)
        test_game = game.move_discard_to_deck(test_game)
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 88)
        self.assert_state_validity(state)


    def test_draw(self):
        self.key, key1, key2 = random.split(self.key, 3)
        test_game = self.my_game
        test_game = game.move_from_deck(test_game, Zone.P1_HAND, 10, key1)
        test_game = game.move_from_deck(test_game, Zone.P2_HAND, 5, key2)
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 73)
        assert(np.sum(state[Zone.P1_HAND]) == 16)
        assert(np.sum(state[Zone.P2_HAND]) == 11)
        self.assert_state_validity(state)


    def test_draw_empty_deck(self):
        # place all but 30 cards from the deck into the discard pile
        self.key, key1 = random.split(self.key, 2)
        test_game = game.move_from_deck(self.my_game, Zone.DISCARD, 58, key1)
        # have each player draw 20 cards
        self.key, key1, key2 = random.split(self.key, 3)
        test_game = game.move_from_deck(test_game, Zone.P1_HAND, 20, key1)
        test_game = game.move_from_deck(test_game, Zone.P2_HAND, 20, key2)
        # check states
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 48)
        assert(np.sum(state[Zone.DISCARD]) == 0)
        assert(np.sum(state[Zone.P1_HAND]) == 26)
        assert(np.sum(state[Zone.P2_HAND]) == 26)
        self.assert_state_validity(state)


    def test_move_cards_in_cup(self):
        # place 20 more cards into each cup
        self.key, key1, key2 = random.split(self.key, 3)
        test_game = self.my_game
        test_game = game.move_from_deck(test_game, Zone.P1_CUP, 20, key1)
        test_game = game.move_from_deck(test_game, Zone.P2_CUP, 20, key2)
        # check states
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 48)
        assert(np.sum(state[Zone.DISCARD]) == 0)
        assert(np.sum(state[Zone.P1_HAND]) == 6)
        assert(np.sum(state[Zone.P2_HAND]) == 6)
        assert(np.sum(state[Zone.P1_CUP]) == 22)
        assert(np.sum(state[Zone.P2_CUP]) == 22)
        self.assert_state_validity(state)


    def test_move_cards_from_zone(self):
        # place cards from hand onto Mandala
        test_game = self.my_game
        p1_colors, p1_counts = np.unique(test_game.state[Zone.P1_HAND], return_counts=True)
        test_game = game.move_from_zone(test_game, Zone.P1_HAND, Zone.M1_MOUNTAIN, int(p1_colors[0]), int(p1_counts[0]))
        p2_colors, p2_counts = np.unique(test_game.state[Zone.P2_HAND], return_counts=True)
        test_game = game.move_from_zone(test_game, Zone.P2_HAND, Zone.M2_MOUNTAIN, int(p2_colors[0]), int(p2_counts[0]))
        # check states
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 88)
        assert(np.sum(state[Zone.DISCARD]) == 0)
        assert(np.sum(state[Zone.P1_HAND]) == 6-p1_counts[0])
        assert(np.sum(state[Zone.P2_HAND]) == 6-p2_counts[0])
        assert(np.sum(state[Zone.P1_CUP]) == 2)
        assert(np.sum(state[Zone.P2_CUP]) == 2)
        assert(np.sum(state[Zone.M1_MOUNTAIN]) == 2+p1_counts[0])
        assert(np.sum(state[Zone.M2_MOUNTAIN]) == 2+p2_counts[0])
        self.assert_state_validity(state)


    def test_move_all_cards_from_zone(self):
        # place all cards from hand onto Mandala
        test_game = self.my_game
        test_game = game.move_all_from_zone(test_game, Zone.P1_HAND, Zone.M1_MOUNTAIN)
        test_game = game.move_all_from_zone(test_game, Zone.P2_HAND, Zone.M2_MOUNTAIN)
        # check states
        state = test_game.state
        assert(np.sum(state[Zone.DECK]) == 88)
        assert(np.sum(state[Zone.DISCARD]) == 0)
        assert(np.sum(state[Zone.P1_HAND]) == 0)
        assert(np.sum(state[Zone.P2_HAND]) == 0)
        assert(np.sum(state[Zone.P1_CUP]) == 2)
        assert(np.sum(state[Zone.P2_CUP]) == 2)
        assert(np.sum(state[Zone.M1_MOUNTAIN]) == 8)
        assert(np.sum(state[Zone.M2_MOUNTAIN]) == 8)
        self.assert_state_validity(state)

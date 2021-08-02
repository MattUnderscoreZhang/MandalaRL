from my_types import Color, Zone, N_COLORS
import unittest
import game
from jax import numpy as np
from jax.random import PRNGKey

class GameTests(unittest.TestCase):
    def setUp(self):
        self.my_game = game.init_game()


    def assert_color_counts_validity(self, color_counts):
        color_counts = np.reduce_sum(color_counts, axis=0)
        assert(np.reduce_sum(color_counts) == 108)
        assert(all(color_counts == np.tile([18], [6])))


    def test_enums(self):
        assert(len(Color) == 6)
        assert(len(Zone) == 14)


    def test_draw(self):
        test_game = self.my_game
        test_game = game.move_cards_from_deck(test_game, Zone.P1_HAND, 10)
        test_game = game.move_cards_from_deck(test_game, Zone.P2_HAND, 5)
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 73)
        assert(np.reduce_sum(color_counts[Zone.P1_HAND]) == 16)
        assert(np.reduce_sum(color_counts[Zone.P2_HAND]) == 11)


    def test_move_discard_into_deck(self):
        test_game = self.my_game
        color_counts = test_game.color_counts
        color_counts = np.tensor_scatter_nd_update(color_counts, [[Zone.DISCARD]], [color_counts[Zone.DECK]])
        color_counts = np.tensor_scatter_nd_update(color_counts, [[Zone.DECK]], [np.tile([0], [N_COLORS])])
        test_game = game.Game(color_counts, test_game.deck_order, test_game.deck_index, test_game.finished_deck_once)
        test_game = game.move_discard_to_deck(test_game)
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 88)
        self.assert_color_counts_validity(color_counts)


    def test_move_cards_from_deck_to_cups(self):
        # place 20 more cards into each cup
        test_game = self.my_game
        test_game = game.move_cards_from_deck(test_game, Zone.P1_CUP, 20)
        test_game = game.move_cards_from_deck(test_game, Zone.P2_CUP, 20)
        # check color_counts
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 48)
        assert(np.reduce_sum(color_counts[Zone.DISCARD]) == 0)
        assert(np.reduce_sum(color_counts[Zone.P1_HAND]) == 6)
        assert(np.reduce_sum(color_counts[Zone.P2_HAND]) == 6)
        assert(np.reduce_sum(color_counts[Zone.P1_CUP]) == 22)
        assert(np.reduce_sum(color_counts[Zone.P2_CUP]) == 22)
        self.assert_color_counts_validity(color_counts)







    def test_draw_more_than_deck(self):
        test_game = self.my_game
        test_game = game.move_cards_from_deck(test_game, Zone.DISCARD, 20)
        test_game = game.move_cards_from_deck(test_game, Zone.P2_HAND, 80)
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 8)
        assert(np.reduce_sum(color_counts[Zone.P1_HAND]) == 6)
        assert(np.reduce_sum(color_counts[Zone.P2_HAND]) == 68)
        self.assert_color_counts_validity(color_counts)


    def test_move_cards_from_zone(self):
        # place cards from hand onto Mandala
        test_game = self.my_game
        p1_colors, p1_counts = np.unique(test_game.color_counts[Zone.P1_HAND], return_counts=True)
        test_game = game.move_from_zone(test_game, Zone.P1_HAND, Zone.M1_MOUNTAIN, int(p1_colors[0]), int(p1_counts[0]))
        p2_colors, p2_counts = np.unique(test_game.color_counts[Zone.P2_HAND], return_counts=True)
        test_game = game.move_from_zone(test_game, Zone.P2_HAND, Zone.M2_MOUNTAIN, int(p2_colors[0]), int(p2_counts[0]))
        # check color_counts
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 88)
        assert(np.reduce_sum(color_counts[Zone.DISCARD]) == 0)
        assert(np.reduce_sum(color_counts[Zone.P1_HAND]) == 6-p1_counts[0])
        assert(np.reduce_sum(color_counts[Zone.P2_HAND]) == 6-p2_counts[0])
        assert(np.reduce_sum(color_counts[Zone.P1_CUP]) == 2)
        assert(np.reduce_sum(color_counts[Zone.P2_CUP]) == 2)
        assert(np.reduce_sum(color_counts[Zone.M1_MOUNTAIN]) == 2+p1_counts[0])
        assert(np.reduce_sum(color_counts[Zone.M2_MOUNTAIN]) == 2+p2_counts[0])
        self.assert_color_counts_validity(color_counts)


    def test_move_all_cards_from_zone(self):
        # place all cards from hand onto Mandala
        test_game = self.my_game
        test_game = game.move_all_from_zone(test_game, Zone.P1_HAND, Zone.M1_MOUNTAIN)
        test_game = game.move_all_from_zone(test_game, Zone.P2_HAND, Zone.M2_MOUNTAIN)
        # check color_counts
        color_counts = test_game.color_counts
        assert(np.reduce_sum(color_counts[Zone.DECK]) == 88)
        assert(np.reduce_sum(color_counts[Zone.DISCARD]) == 0)
        assert(np.reduce_sum(color_counts[Zone.P1_HAND]) == 0)
        assert(np.reduce_sum(color_counts[Zone.P2_HAND]) == 0)
        assert(np.reduce_sum(color_counts[Zone.P1_CUP]) == 2)
        assert(np.reduce_sum(color_counts[Zone.P2_CUP]) == 2)
        assert(np.reduce_sum(color_counts[Zone.M1_MOUNTAIN]) == 8)
        assert(np.reduce_sum(color_counts[Zone.M2_MOUNTAIN]) == 8)
        self.assert_color_counts_validity(color_counts)

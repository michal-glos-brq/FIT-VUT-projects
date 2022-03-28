import logging
import pickle
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack
import sys
import traceback
from .utils import GameData, GameSearch
import logging
import time

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.data = GameData(player_name, board, players_order, max_transfers)
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        # Need to search the state tree
        self.search = True

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        self.board = board
        # If search requested, perform it
        if self.search:
            self.do_search(nb_transfers_this_turn, time_left)
            # If there are no possible moves, end turn
            if self.node[0].best_sons == []:
                return EndTurnCommand()
            # If moves found, perform it before searching the tree again
            self.search = False
        return self.turn(nb_transfers_this_turn)

    def do_search(self, transfers, time_left):
        # If searching in the middle of turn (transfers are not max_allowed)
        if transfers:
            self.search_tree = GameSearch(self.board, self.data, transfers=self.data.max_transfers-transfers)
        else:
            self.search_tree = GameSearch(self.board, self.data)
        # Create game state tree
        self.search_tree.create_game_tree(time_left)
        self.node = [self.search_tree]
        self.attack = 0

    def turn(self, transfers):
        # If previous turn was attack, check if we won, choose the corresponding next move
        if self.attack:
            self.node = self.node[1].best_sons if self.board.get_area(self.attack).get_owner_name() == self.player_name else self.node[0].best_sons
        else:
            self.node = self.node[0].best_sons
        # This should be always True, but just in case something unpredictable happens ...
        if self.node:
            # if, after this turn, there will be no next turns, requiere search
            if not all(node.best_sons for node in self.node):
                self.search = True
            # Get the move, decide whether it's transfer or attack and perform it
            move = self.node[0].move_applied
            if len(move) == 3:
                self.attack = move[1]
                return BattleCommand(int(move[0]), int(move[1]))
            else:
                self.attack = 0
                return TransferCommand(int(move[0]), int(move[1]))
        # If there is no move to be done, end the turn
        return EndTurnCommand()
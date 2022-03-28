import logging
import pickle
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack

class TransferManager:
    def __init__(self, board, my_name):
        self.board = board
        self.my_name = my_name
        self.update(board)

    def update(self, board):
        self.board = board
        self.forward_graph = self._create_forward_graph()

    def _create_forward_graph(self):
        areas = self.board.get_player_areas(self.my_name)
        forward_graph = {}
        for border_area in self.board.get_player_border(self.my_name):
            forward_graph[border_area.get_name()] = {
                'area': border_area,
                'moves': [],
                'border_distance': 0
            }

        for depth in range(len(areas)):
            searched = [search for search in forward_graph.values() if search['border_distance'] == depth]
            for area in searched:
                discovered = [a for a in area['area'].get_adjacent_areas_names() if self.board.get_area(a).get_owner_name() == self.my_name]
                # Handle existing
                for discovered_area in discovered:
                    if discovered_area not in forward_graph:
                        forward_graph[discovered_area] = {
                            'area': self.board.get_area(discovered_area),
                            'moves': [area['area']],
                            'border_distance': area['border_distance'] + 1
                        }
                    elif area['border_distance'] + 1 == forward_graph[discovered_area]['border_distance']:
                        forward_graph[discovered_area]['moves'].append(area['area'])
            if len(forward_graph) == len(areas):
                break
        return forward_graph

    def _transfer_gain(self, src, dst):
        if dst['area'].get_dice == 8 or src['area'].get_dice() == 1:
            return -1
        return min(src['area'].get_dice() - 1, 8 - dst['area'] .get_dice()) - dst['border_distance']

    def get_transfer(self):
        thinkable_transfers = [area for area in self.forward_graph.values() if area['border_distance'] > 0]
        transfers = []
        for transfer in thinkable_transfers:
            for destination in transfer['moves']:
                transfers.append([transfer['area'].get_name(), destination.get_name(), self._transfer_gain(transfer, self.forward_graph[destination.get_name()])])
        return sorted(transfers, reverse=True, key=lambda x: x[2])[0] if transfers else []

class Attack:
    def __init__(self, src, dst, board):
        self.src = src
        self.dst = dst
        self.dice_advantage = src.get_dice() - dst.get_dice()
        self.success_p = probability_of_successful_attack(board, src.get_name(), dst.get_name())

    def attack(self):
        return BattleCommand(self.src.get_name(), self.dst.get_name())

class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.my_name = player_name
        self.board = board
        self.max_transfers = max_transfers
        self.players_order = players_order
        self.transfer_manager = TransferManager(board, player_name)
        self.logger = logging.getLogger('AI')


    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):

        #import pdb; pdb.set_trace()
        if nb_transfers_this_turn < 6:
            self.transfer_manager.update(board)
            tr = self.transfer_manager.get_transfer()
            if tr:
                src, dst, _ = tr
                return TransferCommand(src, dst)

        # Attack
        # Get attack possibilities, sort them according to success rate
        attacks = [Attack(src, dst, board) for src, dst in possible_attacks(board, self.my_name)]
        attacks = [attack for attack in sorted(attacks, key=lambda attack: attack.success_p, reverse=True) if attack.dice_advantage >= 0]

        if attacks:
            return attacks[0].attack()
        return EndTurnCommand()

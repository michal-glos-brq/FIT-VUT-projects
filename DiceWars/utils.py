from time import time
from pprint import pprint
from numpy import array, concatenate, apply_along_axis, zeros, eye, append, argmin, int8, logical_or, argsort
from torch import load, tensor
from .estimator import Estimator

############### Constants and caching ###############
# This is kind of redundant, but indexing this dictionary is faster then calling probability_of_successful_attack method
ATTACK_PROBS = {
        1: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
        2: {1: 0.83796296, 2: 0.44367284, 3: 0.15200617, 4: 0.03587963, 5: 0.00610497, 6: 0.00076625, 7: 0.00007095, 8: 0.00000473},
        3: {1: 0.97299383, 2: 0.77854938, 3: 0.45357510, 4: 0.19170096, 5: 0.06071269, 6: 0.01487860, 7: 0.00288998, 8: 0.00045192},
        4: {1: 0.99729938, 2: 0.93923611, 3: 0.74283050, 4: 0.45952825, 5: 0.22044235, 6: 0.08342284, 7: 0.02544975, 8: 0.00637948},
        5: {1: 0.99984997, 2: 0.98794010, 3: 0.90934714, 4: 0.71807842, 5: 0.46365360, 6: 0.24244910, 7: 0.10362599, 8: 0.03674187},
        6: {1: 0.99999643, 2: 0.99821685, 3: 0.97529981, 4: 0.88395347, 5: 0.69961639, 6: 0.46673060, 7: 0.25998382, 8: 0.12150697},
        7: {1: 1.00000000, 2: 0.99980134, 3: 0.99466336, 4: 0.96153588, 5: 0.86237652, 6: 0.68516499, 7: 0.46913917, 8: 0.27437553},
        8: {1: 1.00000000, 2: 0.99998345, 3: 0.99906917, 4: 0.98953404, 5: 0.94773146, 6: 0.84387382, 7: 0.67345564, 8: 0.47109073},
    }

# Expanded states: A set with tuples in format tuple(tuple((tuple(area_dice, area_owner) for each area in board)), player_on_turn)
EXPANDED_STATES = set()
DICE_LIMIT = 8


###########################################################################

class GameData:
    '''
    This class holds some 'global' data for all states
        self.my_name:       Integer assigned to us as ID of player
        self.actual_player: Integer identifiyng player for whom GameStates are being expanded
        self.max_transfers: Maximal amount of transfers allowed for each turn
        self.estimator:     GCN for estimating win probability for game state
        self.g_mat:         Matrix representing board as graph (NnodesxNnodes, 1 where connestion, 0 where no connections)
        self.players_order: A list with 'actions' needed to search the game state
            This is a list of integers, ordered as game turns go. self.my_name is first and last, because we want to search the
            states beggining with ours turn and ending with ours second turn. 0 is there to represent the dice giveaway between each round
    '''
    def __init__(self, player_name, board, players_order, max_transfers):
        '''Initiate the GameData class, see it's docstring for explanation'''
        self.my_name, self.actual_player, self.max_transfers = player_name, player_name, max_transfers
        my_turn = players_order.index(player_name)
        self.players_order = players_order[my_turn:] + [0] + players_order[:my_turn] + [player_name]
        self.g_mat_gen(board)
        self.calculate_distances(board)
        self.load_estimator()

    def g_mat_gen(self, board):
        '''Create g_mat matrix'''
        g_mat = zeros((34,34), dtype=int8)
        points = [[area.name, n] for area in board.areas.values() for n in area.get_adjacent_areas_names() if n]
        points = array(points, dtype=int8) - 1
        for point in points:
            g_mat[point[0], point[1]] = 1
        g_mat += eye(34, dtype=int8)
        self.g_mat = g_mat.reshape((1,34,34))

    def calculate_distances(self, board):
        g_mat = zeros((34,34), dtype=int8)
        for area in board.areas.values():
            for n in area.get_adjacent_areas_names():
                g_mat[int(area.name)-1, n-1] = 1
        for i in range(34):
            for x in range(34):
                for y in range(34):
                    if g_mat[x, y] != 0:
                        for yy in range(34):
                            if g_mat[y, yy] != 0:
                                if g_mat[x,yy] > g_mat[y,yy] + g_mat[x,y] or g_mat[x,yy] == 0:
                                    g_mat[x,yy] = g_mat[y,yy] + g_mat[x,y]
                                    g_mat[yy,x] = g_mat[y,yy] + g_mat[x,y]
        for i in range(34):
            g_mat[i,i] = 34
        self.distances = g_mat
        self.max_dist = self.distances.max()

    def load_estimator(self, path='dicewars/ai/xbedna74/estimator'):
        '''Load the estimator neural network'''
        ##### TODO: Comment line 80 and uncomment line 81 when downloaded new model
        #self.estimator = Estimator(gcn=[4,8], linear=[272,16], bias=False, lin_bias=False)
        self.estimator = Estimator(gcn=[4,12,20,28], linear=[1088, 256], bias=False, lin_bias=False)
        self.estimator.load_state_dict(load(path))
        self.estimator.eval()



class GameSearch:
    '''
    Class representing the state of game as node
    self.father:            Father state
    self.move_applied:      Move applied to father state to get into this state
    self.data:              Game data instance. For this reason, we have to expand all states for each player before letting the other player play
    self.new_states_t:      New states generated from actual state with a transfer (list of GameSearch instances)
    self.new_states_a:      New states generated from actual state with a attacks (list of tuples. Tuple includes states: (failed_attack, succesfull attack))
    self.best_sons:         Son states with highest heuristics
    self.h:                 Heuristics of this state
    self._h:                Heuristics of move_applied (combined heuristics for attacks)
        (direct son when transfer applied, combination of heuristics of succesfull and failed attacks when attack applied)
    self.p:                 Probability of this state happening when self.move_applied applied to father state
    Important structures
    self.board - a dict of dicts representing areas (area id used as key),
        area dict schema: {
            'id': areas id,
            'neighbours': This is dictionary with player ids as keys. For each player id, a value (set) is assigned.
                The set contains neigbouring areas belonging to that player.
            'dice': amount of dice present on area,
            'owner': owner of the area
        }
    How things are represented:
    attack:     tuple((src_id, dst_id, succ_probability))
    transfer:   tuple((src_id, dst_id))
    when move applied is (-1, -1), it's root sate or it's state generated from let_next_player method
    '''

    def __init__(self, game_board, game_data, father=None, constructed_board=None, move_applied=(-1, -1), lost=False, transfers=None, h=None):
        '''
        Initialize new instance of game state
        
        Arguments:
            game_board:         This is only passed for root state. It's instance of game board from actual dicewars (not our internal representation)
            game_data:          Instance of GameData
            father:             Father node of initiated GameState
            constructed_board:  For each GameNode (excluded root node), this is passed as the board sate
            move_applied:       Move applied to father state to get this instance
            lost:               If this state was generated from a attack, this indicates this is the next state, when we lost the battle
            transfers:          Transfers left
            h:                  Heuristics
        '''
        # Initialize with some essential data
        self.data, self.move_applied, self.lost, self.father, self.player = game_data, move_applied, lost, father, game_data.actual_player
        self.new_states_a, self.new_states_t, self.h, self.best_sons, self.brother, self.next_player_state = [], [], h, [], None,  None
        self.transfers = transfers if not transfers is None else game_data.max_transfers
        # If dictionary of this state already constructed with father state and move applied in one of methods: {from_attack, from_transfer, let_next_player}
        # do not construct it again from class provided by original dicewars
        if not constructed_board:
            constructed_board = {}
            # Go through each area of board and create an internal representation of GameState
            for k, area in game_board.areas.items():
                # index with integers...
                k = int(k)
                neighbours = [ game_board.get_area(_area) for _area in area.get_adjacent_areas_names()]
                area_data = {
                    'id': k,
                    # Placeholder for neighbours
                    'neighbours': {_id: set() for _id in self.data.players_order},
                    'dice': area.get_dice(),
                    'owner': area.get_owner_name()
                }
                # Find neighbouring areas of each player (empty set if actual area does not neighbour with a player)
                [area_data['neighbours'][_area.get_owner_name()].add(_area.get_name()) for _area in neighbours]
                constructed_board[k] = area_data
        self.board = constructed_board
        # Remember my nodes, useful for later actions
        self.my_nodes = [(n['id'], n['dice']) for n in self.board.values() if n['owner'] == game_data.actual_player]
        # Those are not real areas IDs, those are indexes into distance matrix from self.data.distances
        self.border_nodes = array([node[0]-1 for node in self.my_nodes if set().union(*constructed_board[node[0]]['neighbours'].values()) - constructed_board[node[0]]['neighbours'][self.player]], dtype=int8)
        # Calculate probability (for transfers or let_next_player 1, else probablility of this state happening whe attacked)
        self.p = 1 if len(move_applied) == 2 else (1 - move_applied[2] if lost else move_applied[2])


    def let_next_player(self):
        '''New state just to let the other player move'''
        # This does not have to be cehcked (Is two same states should happen, they would be eliminated in previous hash-checking)
        hash_str = (tuple((a['dice'], a['owner']) for a in self.board.values()), self.data.actual_player)
        new_state = GameSearch(None, self.data, father=self, constructed_board=self.board, transfers=self.data.max_transfers)
        # But still, save it's hash for further control in self.get_good_leafs
        new_state.hash = hash_str
        new_state.apply_heuristics()
        self.next_player_state = new_state
        return new_state


    def from_transfer(self, transfer, EXPANDED_STATES):
        '''Create new state from father state and transfer. TIME CRITICAL. Use debug arg to show the board changes.'''
        # This is probably as fast as it could get
        # Expand the move, avoid indexing with local variables
        board = self.board
        # Cut it
        transfer = transfer[:2]
        src_id, dst_id = transfer
        # Copy the board and copy two affected areas - shallow copy is fine, changing only values directly accessed with key dice
        new_board = board.copy()
        src_area, dst_area = board[src_id].copy(), board[dst_id].copy()
        src_dice, dst_dice = src_area['dice'], dst_area['dice']
        src_area['dice'] = max(1, src_dice - (DICE_LIMIT - dst_dice))
        dst_area['dice'] = min(8, src_dice + dst_dice - 1)
        new_board[src_id], new_board[dst_id] = src_area, dst_area
        # look if this state was not already discovered
        hash_str = (tuple((a['dice'], a['owner']) for a in new_board.values()), self.data.actual_player)
        if hash_str not in EXPANDED_STATES:
            EXPANDED_STATES.add(hash_str)
            new_state = GameSearch(None, self.data, father=self, constructed_board=new_board, move_applied=transfer, transfers=self.transfers-1)
            new_state.hash = hash_str
            self.new_states_t.append(new_state)


    def from_attacks(self, attack, EXPANDED_STATES):
        '''Create new states from father and attack (success and fail) TIME CRITICAL. Use debug arg to show the board changes.'''
        # Expand the move and get some variables to avoid indexing
        board = self.board
        src_id, dst_id, _ = attack
        src_area, dst_area = board[src_id], board[dst_id]
        dst_owner, src_owner = dst_area['owner'], self.data.actual_player
        src_dice, dst_dice = src_area['dice'], dst_area['dice']
        src_area_fail, dst_area_fail = src_area.copy(), dst_area.copy()
        new_board_fail = board.copy()

        # Fail needs only shallow copy of src and dst area dict, because we change integers directly accessed by key (dice)
        src_area_fail['dice'], dst_area_fail['dice'] = 1, max(1, dst_dice - (0 if src_dice < 4 else (1 if dst_dice != 8 else 2)))
        new_board_fail[src_id], new_board_fail[dst_id] = src_area_fail, dst_area_fail
        # If failed state does not already exist, consider it for further expansion

        # Succesfull attack. First avoud some indexing ...
        src_area_succ, dst_area_succ, new_board_succ = src_area.copy(), dst_area.copy(), board.copy()
        # Success tho needs semi-deep copy of two areas and their neighbours. This gets a little tricky here
        # First, the destination tile. The neighbours stay the same, the owner and dice would change, hence shallow copy of area dict
        # Conquer the destination tile, become owner and transfer all dice except one
        dst_area_succ['owner'] = src_owner
        dst_area_succ['dice'] = src_dice - 1
        # Source tile stays with 1 dice only
        src_area_succ['dice'] = 1
        # Fix neighbours
        src_area_succ['neighbours'] = src_area['neighbours'].copy()
        neighbours = src_area_succ['neighbours']
        neighbours[src_owner], neighbours[dst_owner] = neighbours[src_owner].copy(), neighbours[dst_owner].copy()
        neighbours[src_owner].add(dst_id)
        neighbours[dst_owner].remove(dst_id)
        # Third, neighbours of dst tile -> area shallow copy, neighs shallow copy, sets with tiles owned by src and dst owners deepcopy
        # For each neighbour of dts tile, change it's owner of neighbour dst_id to new owner (src_area already fixed)
        for neigh in set().union(*board[dst_id]['neighbours'].values()) - {src_id}:
            # Copying process ...
            new_board_succ[neigh] = board[neigh].copy()
            new_board_succ[neigh]['neighbours'] = board[neigh]['neighbours'].copy()
            neighbours = new_board_succ[neigh]['neighbours']
            neighbours[src_owner] = board[neigh]['neighbours'][src_owner].copy()
            neighbours[dst_owner] = board[neigh]['neighbours'][dst_owner].copy()
            # Now, we can actually manipulate the neighbours
            neighbours[src_owner].add(dst_id)
            neighbours[dst_owner].remove(dst_id)
        # Now store the new areas into board
        new_board_succ[dst_id], new_board_succ[src_id] = dst_area_succ, src_area_succ
        hash_str_fail = (tuple((a['dice'], a['owner']) for a in new_board_fail.values()), self.data.actual_player)
        hash_str_succ = (tuple((a['dice'], a['owner']) for a in new_board_succ.values()), self.data.actual_player)
        if hash_str_fail not in EXPANDED_STATES and hash_str_succ not in EXPANDED_STATES:
            EXPANDED_STATES.add(hash_str_fail)
            EXPANDED_STATES.add(hash_str_succ)
            sate_fail = GameSearch(None, self.data, father=self, constructed_board=new_board_fail,
                                                        move_applied=attack, transfers=self.transfers, lost=True)
            state_succ = GameSearch(None, self.data, father=self, constructed_board=new_board_succ,
                                                        move_applied=attack, transfers=self.transfers)
            sate_fail.brother, state_succ.brother = state_succ, sate_fail
            sate_fail.hash, state_succ.hash = hash_str_fail, hash_str_succ
            self.new_states_a.append([sate_fail, state_succ])


    def get_good_attacks(self, board):
        '''Return preffered attacks. TIME CRITICAL METHOD'''
        # Get all possile attacks
        attacks = []
        for src in [src for src in self.my_nodes if src[1] > 1]:
            src_id, src_dice = src
            for dst_id in (set().union(*board[src_id]['neighbours'].values())-board[src_id]['neighbours'][self.data.actual_player]):
                dst_dice = board[dst_id]['dice']
                if src_dice > dst_dice or src_dice == 8:
                    attacks.append((src_id, dst_id, ATTACK_PROBS[src_dice][dst_dice]))
        # Expand attack only from areas (all its' possible attacks) with highest probability of succesfull attack
        if attacks:
            highest_probability = max([att[2] for att in attacks])
            best_attacks_srcs = set(att[0] for att in attacks if att[2] == highest_probability)
            attacks = [attack for attack in attacks if attack[0] in best_attacks_srcs]
        return attacks


    def get_good_transfers(self, board):
        '''Return preffered transfers. TIME CRITICAL METHOD'''
        transfers = []
        if self.transfers > 0:
            for area in self.my_nodes:
                tmp_neighs = self.board[area[0]]['neighbours'][self.data.actual_player]
                if area[1] > 1 and tmp_neighs:
                    for neigh in tmp_neighs:
                        neigh_dice = self.board[neigh]['dice']
                        if neigh_dice < DICE_LIMIT:
                            transfers.append([area[0], neigh])
        if transfers and self.border_nodes.size:
            # Get distances from all nodes on the border
            dists_from_borders = self.data.distances[:,self.border_nodes]
            transfers = array(transfers, dtype=int8)
            # Get transfer source and destination as IDs
            src_ids, dst_ids = (transfers).transpose() - 1
            # Which border node is closest to sources?
            min_src_from_border_areas = argmin(dists_from_borders[src_ids], axis=1)
            min_dst_from_border_areas = argmin(dists_from_borders[dst_ids], axis=1)
            # Get distances from closest src border node to dst etc...
            dst_distances1 = dists_from_borders[dst_ids, min_src_from_border_areas]
            dst_distances2 = dists_from_borders[dst_ids, min_dst_from_border_areas]
            src_distances1 = dists_from_borders[src_ids, min_src_from_border_areas]
            src_distances2 = dists_from_borders[src_ids, min_dst_from_border_areas]
            distance_gain1 = src_distances1 - dst_distances1
            distance_gain2 = src_distances2 - dst_distances2
            # Choose transfers with distance gain
            candidate_transfers_ids = logical_or(
                logical_or(distance_gain1 == 1, distance_gain1 == -33), logical_or(distance_gain2 == 1, distance_gain2 == -33)
            )
            return transfers[candidate_transfers_ids]
        return transfers


    def prune_sons_GCN(self):
        '''Get best attack and transfer'''
        # Get the inputs for estimator from expanded move from this state
        states = self.new_states_t + [s for ss in self.new_states_a for s in ss]
        if states:
            boards = [s.board for s in states]
            owners = zeros((len(boards),34,4), dtype=int8)
            for i in range(1,35):
                for j, board in enumerate(boards):
                    owners[j][i-1][board[i]['owner']-1] = 1
            dice = array([[board[i]['dice'] for i in range(1,35)] for board in boards], dtype=int8).reshape((-1,34,1)) * owners
            X = self.apply_GCN(dice, entries=len(boards))
            # save the heuristics in sons objects
            for i, son in enumerate(states):
                son._h = float(X[i][self.data.actual_player - 1])
            # Avg attack sons heuristics (to know heuristics of the move)
            for state_a in self.new_states_a:
                s1, s2 = state_a
                h = s1._h * s1.p + s2._h * s2.p
                s1._h, s2._h = h, h
            # Is attack better then transfer or vice versa
            a, t = 0, 0
            if self.new_states_t:
                self.new_states_t = [sorted(self.new_states_t, key=lambda x: x._h, reverse=True)[0]]
                t = self.new_states_t[0]._h
            if self.new_states_a:
                self.new_states_a = [sorted(self.new_states_a, key=lambda x: x[0]._h, reverse=True)[0]]
                a = self.new_states_a[0][0]._h


    def apply_GCN(self, dice, entries=1):
        '''Apply GCN estimator to predict the probability of winning from this state'''
        return self.data.estimator(tensor(dice).float(), tensor(self.data.g_mat).float().repeat((entries,1,1)))


    def apply_heuristics(self):
        '''Apply heuristics just on single state (For root node and let_next_player)'''
        # Construct the input to be fed into estimator
        owners = zeros((34,4), dtype=int8)
        for i in range(1,35):
            owners[i-1][self.board[i]['owner']-1] = 1
        dice = array([self.board[i]['dice'] for i in range(1,35)], dtype=int8).reshape((-1,34,1)) * owners
        X = self.apply_GCN(dice)
        # save the heuristics
        self._h = float(X[0][self.data.actual_player - 1])

    def get_good_leafs(self, nodes, player):
        # Always choose 4 best paths (for performance reasons)
        n_paths_to_examine = 4
        n_paths_added_to_examine = 0 # How many subtrees loaded?
        leafs_to_examine = []
        subtree_nodes = set()   # Nodes already put to examination (avoid those in leafs_to_examine)
        interest_leafs = sorted(nodes, reverse=True, key=lambda x: x._h) # Sort all leafs from best to worse
        for leaf in interest_leafs:
            # If leaf not already in examined subtree
            if not leaf.hash in subtree_nodes:
                # Add all it's father nodes, brother nodes and fathers brothers sons
                node = leaf
                leafs_to_examine.append(leaf)
                # Search only for players subtrees
                while node and node.player == player:
                    subtree_nodes.add(node.hash) # add actual node as examined node
                    if node.brother:    # If examined node has brothers, add it's sons as examined nodes and it's leafs for further examination
                        if not node.brother.hash in subtree_nodes:
                            to_add = [node.brother]
                            i = 0
                            while i < len(to_add):
                                n = to_add[i]
                                subtree_nodes.add(n.hash)
                                if n.best_sons:
                                    to_add += n.best_sons
                                else:
                                    leafs_to_examine.append(n)
                                i += 1
                    # Move up the tree
                    node = node.father
                n_paths_added_to_examine += 1
                if n_paths_added_to_examine == n_paths_to_examine:
                    break
        # Now, let the other player move from examined nodes
        nodes = array([s.let_next_player() for s in leafs_to_examine])
        [node.apply_heuristics() for node in nodes]
        return nodes


    def create_game_tree(self, time_left):
        '''Search through state tree. Take time into account'''
        # First, time measurement
        EXPANDED_STATES = set()
        time_left = min(1.1, 0.1*time_left)
        time_start = time()

        # There will be stored the whole game tree
        tree = []

        # Hash the state to recognize it, apply heuristics
        hash_str = (tuple((a['dice'], a['owner']) for a in self.board.values()), self.data.actual_player)
        EXPANDED_STATES.add(hash_str)
        self.hash = hash_str
        self.apply_heuristics()
        
        # Start the search
        to_expand = array([self])
        # There will be stored same moves searched in my first turn to be further examined
        moves_to_examine = []
        # BFS (sorta)
        for player in self.data.players_order:
            # Choose player for whom the tree is being created
            self.data.actual_player = player
            if self.data.actual_player:
                i = 0
                while i < to_expand.size:
                    # It's faster to save it once and do not index after
                    node = to_expand[i]
                    # Expand all new states
                    [GameSearch.from_attacks(node, move, EXPANDED_STATES) for move in node.get_good_attacks(node.board)]
                    [GameSearch.from_transfer(node, move, EXPANDED_STATES) for move in node.get_good_transfers(node.board)]
                    node.prune_sons_GCN()
                    # Save new expanded states fo further expansion
                    to_expand = concatenate((to_expand, [s for ss in node.new_states_a for s in ss], node.new_states_t))
                    i += 1
                    # If not enough time left, cancel the search
                    if time() - time_start > time_left:
                        # Add new nodes into tree
                        tree = append(tree, to_expand)
                        break
                # Add new nodes into tree
                tree = append(tree, to_expand)
                # Update player ID in order to "let it play"
                p_pos = self.data.players_order.index(player)
                self.data.actual_player = self.data.players_order[p_pos+1] if self.data.players_order[p_pos+1] else self.data.players_order[p_pos+2]
                # Get some good subtrees for further examination
                # Those subtrees would be further expanded
                to_expand = self.get_good_leafs(to_expand, self.data.actual_player)
        # After the whole search, choose the pseudo-optimal path determined by estimator neural network
        # Search the tree from leafs (there is the most precise heuristics, I suppose ...)
        for node in tree[::-1]:
            if node.next_player_state:
                # sub heurisitcs
                node._h -= node.next_player_state._h
            elif node.new_states_a:
                if node.new_states_t:
                    if node.new_states_t[0]._h > node.new_states_a[0][0]._h:
                        node._h = node.new_states_t[0]._h
                        node.best_sons = node.new_states_t
                        continue
                node._h = node.new_states_a[0][0]._h
                node.best_sons = node.new_states_a[0]
                # Choose best son, avg the heuristics
            elif node.new_states_t:
                node._h = node.new_states_t[0]._h
                node.best_sons = node.new_states_t
        # Set me as actual player in order not to confuse self.data
        self.data.actual_player = self.data.players_order[0]
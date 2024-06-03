from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    # ### START CODE HERE ###
    def minimax(state, depth, agentIndex):
      """
      Recursive function for minimax algorithm.
      """
      numAgents = state.getNumAgents()

      # check isEnd
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      successorStates = [state.generateSuccessor(agentIndex, action) for action in legalActions]

      # pacman (maximizer)
      if agentIndex == 0:
        return max(minimax(successor, depth - 1, (agentIndex + 1) % numAgents) for successor in successorStates)

      # ghost (minimizer)
      else:
        nextAgentIndex = (agentIndex + 1) % numAgents

        # If all ghosts have moved, increment the depth
        if nextAgentIndex == 0:
          depth -= 1

        return min(minimax(successor, depth, nextAgentIndex) for successor in successorStates)

    legalActions = gameState.getLegalActions(0)  # pacman's legal actions
    bestAction = Directions.STOP
    bestScore = float('-inf')

    for action in legalActions:
      successorState = gameState.generateSuccessor(0, action)
      score = minimax(successorState, self.depth, 1)

      if score > bestScore:
        bestScore = score
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    # ### START CODE HERE ###
    def alpha_beta(state, depth, alpha, beta, agentIndex):
      """
      Recursive function for minimax algorithm with alpha-beta pruning.
      """
      numAgents = state.getNumAgents()

      # check isEnd
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)

      if agentIndex == 0:  # pacman (maximizer)
        v = float('-inf')
        for action in legalActions:
          successorState = state.generateSuccessor(agentIndex, action)
          v = max(v, alpha_beta(successorState, depth - 1, alpha, beta, (agentIndex + 1) % numAgents))
          if v > beta:
            return v  # prune
          alpha = max(alpha, v)
        return v

      else:  # ghost (minimizers)
        v = float('inf')
        nextAgentIndex = (agentIndex + 1) % numAgents

        if nextAgentIndex == 0:
          depth -= 1

        for action in legalActions:
          successorState = state.generateSuccessor(agentIndex, action)
          v = min(v, alpha_beta(successorState, depth, alpha, beta, nextAgentIndex))
          if v < alpha:
            return v  # prune
          beta = min(beta, v)
        return v

    legalActions = gameState.getLegalActions(0)
    bestAction = Directions.STOP
    alpha = float('-inf')
    beta = float('inf')

    for action in legalActions:
      successorState = gameState.generateSuccessor(0, action)
      score = alpha_beta(successorState, self.depth, alpha, beta, 1)

      if score > alpha:
        alpha = score
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # ### START CODE HERE ###
    def expectimax(state, depth, agentIndex):
      """
      Recursive function for expectimax algorithm.
      """
      numAgents = state.getNumAgents()

      # check isEnd
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)

      if agentIndex == 0:  # pacman (maximizer)
        return max(
          expectimax(state.generateSuccessor(agentIndex, action), depth - 1, (agentIndex + 1) % numAgents) for action in
          legalActions)

      else:  # ghost (expectation over uniform random choice)
        nextAgentIndex = (agentIndex + 1) % numAgents

        if nextAgentIndex == 0:
          depth -= 1

        probability = 1.0 / len(legalActions)
        return sum(
          expectimax(state.generateSuccessor(agentIndex, action), depth, nextAgentIndex) * probability for action in
          legalActions)

    legalActions = gameState.getLegalActions(0)
    bestAction = Directions.STOP

    bestScore = max(expectimax(gameState.generateSuccessor(0, action), self.depth, 1) for action in legalActions)

    for action in legalActions:
      successorState = gameState.generateSuccessor(0, action)
      score = expectimax(successorState, self.depth, 1)
      if score == bestScore:
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

  """
  # ### START CODE HERE ###
  from util import manhattanDistance
  from multiAgentsSolution import staffEvaluationFunction, DistanceCalculator

  evaluationScore = staffEvaluationFunction(currentGameState, DistanceCalculator)

  return evaluationScore
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction

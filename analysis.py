# analysis.py
# -----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

######################
# ANALYSIS QUESTIONS #
######################

# Change these default values to obtain the specified policies through
# value iteration.

def question2():
  answerDiscount = 0.9
  answerNoise = 0
  # if noise is 0, then the action right will not be affected by the other action out of bridge.
  return answerDiscount, answerNoise

def question3a():
  answerDiscount = 0.2
  answerNoise = 0
  answerLivingReward = 0.0
  # noise = 0, make the action deterministic, 
  # disctount = 0.2, mak
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3b():
  # not solved
  answerDiscount = 0.3
  answerNoise = 0.2
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3c():
  answerDiscount = 0.9
  answerNoise = 0
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3d():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = 0.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question3e():
  answerDiscount = 0.9
  answerNoise = 0.2
  answerLivingReward = 10.0
  return answerDiscount, answerNoise, answerLivingReward
  # If not possible, return 'NOT POSSIBLE'

def question6():
  answerEpsilon = None
  answerLearningRate = None
  return 'NOT POSSIBLE'
  # return answerEpsilon, answerLearningRate
  # If not possible, return 'NOT POSSIBLE'
  
if __name__ == '__main__':
  print('Answers to analysis questions:')
  import analysis
  for q in [q for q in dir(analysis) if q.startswith('question')]:
    response = getattr(analysis, q)()
    print('  Question %s:\t%s' % (q, str(response)))

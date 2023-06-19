import numpy as np
import os
import gym

# a = []



# for i in range(10):
#     a.append(i)
    
    
# np.savez('./test.npz', test=a)    


# score_board = np.load('./score/dqn_score.npz', allow_pickle=True)
# score_board = list(score_board['score_board'])
env = gym.make('BreakoutDeterministic-v4')
score_board = np.load('./score/ddqn_score.npz', allow_pickle=True)
score_board = list(score_board['score_board'])

print(len(score_board))
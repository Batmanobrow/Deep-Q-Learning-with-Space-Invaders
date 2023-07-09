import gym
import random
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

#Criando Ambiente:
env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

def build_model(height, width, channels, actions):
    model = Sequential()
    #Camadas convolucionais:
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    #cria camada convolucional 2D com 32 filtros de tamanho 8x8 strides 4x4 -se movem 4 pixels na altura e 4 pixels na largura a cada passo)
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    #Camadas densas
    #usadas para mapear os recursos extraídos pelas camadas convolucionais em ações ou 
    #valores Q em problemas de aprendizagem por reforço.
    model.add(Dense(512, activation='relu')) #camada com 512 neuronios
    model.add(Dense(256, activation='relu')) #camada densa com 256 
    model.add(Dense(actions, activation='linear')) #Cria a ultima camada com numero de neurônos iguais a de açãoes posssiveis no jogo, 6
    return model

    #Função para criar o agente
def build_agent(model, actions):
    #politica de exploração 
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=1000)
    #memoria   armazenar as transições (estado, ação, recompensa, próximo estado) 
    #para posteriormente utilizar no treinamento do agente.
    memory = SequentialMemory(limit=1000, window_length=3)
    #Criando o agente de aprendizado por reforço
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    enable_dueling_network=True, dueling_type='avg',  ##rede dueling separa a estimativa valor do estado e valor da ação
                    nb_actions=actions, nb_steps_warmup=1000 ##joga 10000 vezes com ações aleatórias antes de aprender
                    )
    return dqn

model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions) ### criando o agente
dqn.compile(Adam(lr=1e-4))

#Inicia o treinamento do agente
#dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)
#passa o ambiente env, treina 100000 vezes, visualize falso pois não copila aqui,
#verbose=2 para ter mais detalhamento 
##pontuacao=[]
##dqn.load_weights('Pesos/10kep/dqn_weights.h5f')
##scores = dqn.test(env, nb_episodes=100, visualize=True)
##print(np.mean(scores.history['episode_reward']))
##pontuacao= scores.history['episode_reward']
##np.save("pontuacao_10k_ep.npy", pontuacao)

pont = []


episodes = 101
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        ##env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    pont.append(score)
env.close()
np.save("pontuacao_random_ep.npy", pont)
print(pont)

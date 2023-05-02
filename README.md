Download Link: https://assignmentchef.com/product/solved-cmu10403-homework4-ddpg
<br>
You are provided with a custom environment in 2Dpusher env.py. In order to make the environment using gym, you can use the following code:

import gym import envs env = gym.make(‘Pushing2D-v0’)

The environment is considered “solved” once the percent successes (i.e., the box reaches the goal within the episode) reaches 95%.

<h1>Problem 1: Deep Deterministic Policy Gradients (DDPG)</h1>

In this problem you will implement DDPG, an off-policy RL algorithm for continuous action spaces. In homework 2 you implemented DQN, another off-policy RL algorithm. Like DQN, DDPG will learn a Q-function. Recall DQN chose actions by computing the Q value for each action and then chose the maximum:

<em>π</em>(<em>a </em>| <em>s</em>) = 1(<em>a </em>= max<em>Q</em>(<em>s,a</em>))

<em>a</em>

While we would like to use this same policy in continuous action spaces, finding the optimal action involves solving an optimization problem. Since solving this optimization problem is expensive, you will <em>amortize </em>the cost of optimization by learning a policy that predicts the optimum. Intuitively, you will solve the following optimization problem:

max<em>Q</em>(<em>s,a </em>= <em>µ</em>(<em>s </em>| <em>θ<sup>µ</sup></em>))

<em>θ</em><em>µ</em>

Using TensorFlow/PyTorch, you can directly take the gradient of this objective w.r.t. the policy parameters, <em>θ<sup>µ</sup></em>. If you work this out by hand by applying the chain rule, you will get the same expression as in the Algorithm 1. There are a few things to note:

<ol>

 <li>You will learn an actor network with parameters <em>θ<sup>µ </sup></em>and a critic network with parameters <em>θ<sup>Q</sup></em>.</li>

 <li>Similar to DQN, you will use a <em>target network </em>for both the actor and the critic. These target networks have parameters{<em>θ<sup>Q</sup></em><sup>0</sup><em>,θ<sup>µ</sup></em><sup>0</sup>} are slowly updated towards the trained weights.</li>

</ol>

Figure 1: DDPG algorithm presented by [3].

<ol start="3">

 <li>The algorithm requires a random process N to offset the deterministic actor policy. For this assignment, you can use an -normal noise process, where with probability , you sample an action uniformly from the action space and otherwise sample from a normal distribution with the mean as indicated by your actor network and standard deviation as a hyperparameter.</li>

 <li>There is a replay buffer <em>R </em>which can have a burn-in period, although this is not required to solve the environment.</li>

 <li>The target values values <em>y<sub>i </sub></em>used to update the critic network is a one-step TD backup where the bootstrapped <em>Q </em>value uses the slow moving target weights {<em>θ<sup>µ</sup></em><sup>0</sup><em>,θ<sup>Q</sup></em><sup>0</sup>}.</li>

 <li>The update for the actor network differs from the traditional score function update used in vanilla policy gradient. Instead, DDPG uses information from the critic about how actions influence value estimates and pushes the policy in a direction to maximize increase in estimated rewards.</li>

</ol>

To implement DDPG, we recommend following the steps below:

<ol>

 <li>Create actor and critic networks; for actor and critic network, use the algo/criticnetwork.py and algo/actornetwork.py respectively. For this environment, a simple fully connected network with two layers should suffice. You can choose which optimizer and hyperparameters to use, so long as you are able to solve the environment. We recommend using Adam as the optimizer. It will automatically adjust the learning rate based on the statistics of the gradients it’s observing. You can check they create the network you wanted using the function create actor network and create critic network and printing the model architectures.</li>

 <li>Connect the two implemented models in the DDPG method in py script.</li>

 <li>In the file run.py, implement the main training and evaluation loops.</li>

</ol>

The file ReplayBuffer.py does not need to be changed. Generally, you can find the places where we expect you to add code by ”NotImplementedError”. For this part you don’t need to modify add hindsight replay experience function. Train your implementation on the Pushing2D-v0 environment until convergence<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>, and answer the following questions:

<ol>

 <li>(15 pts) The neat trick of DDPG is that you can learn a policy by taking gradients of the Q function directly w.r.t. the policy parameters. An alternative approach that seems easier would if we could directly take gradients of the <em>cumulative reward </em>r.t. the policy parameters, without having to learn a Q function. Why is this approach not feasible? Optional (0 pts): How could you make this approach feasible?</li>

 <li>(10 pts) In 2-3 sentences, explain how you implemented the actor update.</li>

 <li>(5 pts) Describe the hyperparameter settings that you used to train DDPG.</li>

 <li>(20 pts) Plot the mean cumulative reward: Every <em>k </em>episodes, freeze the current cloned policy and run 10 test episodes, recording the mean/std of the cumulative reward. Plot the mean cumulative reward <em>µ </em>on the y-axis with ±<em>σ </em>standard deviation as error-bars vs. the number of training episodes on the x-axis. You don’t need to use the noise process N when testing. Hint: You can use matplotlib’s errorbar() function. <a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html">https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html</a></li>

 <li>(10 pts) You might have noticed that the TD error is <em>not </em>a good predictor of whether your DDPG agent is learning. What other metric might you use to measure performance <em>without collecting new transitions from the environment</em>? <em>Why </em>is this a reasonable metric? Implement this metric, and then determine whether this metric is actually useful for predicting the agent’s performance.</li>

 <li>(Extra credit: up to 5 pts) DDPG is known to overestimate the value of states and actions. A recent method, TD3 [2], proposes a correction that avoids this overestimation by learning two Q functions, <em>Q</em><sub>1</sub>(<em>s,a</em>) and <em>Q</em><sub>2</sub>(<em>s,a</em>), and then choosing actions according to the minimum (i.e., acting pessimistically):</li>

</ol>

maxmin(<em>Q</em><sub>1</sub>(<em>s,a</em>)<em>,Q</em><sub>1</sub>(<em>s,a</em>)) <em>a</em>

Extend your DDPG implementation to implement TD3, and conduct an experiment to compare the two algorithms. Provide a plot comparing the two algorithms and write a few sentences explaining your results.

<h1>Problem 2: Hindisght Experience Replay (HER)</h1>

In this section, you will combine HER with DDPG to hopefully learn faster on the Pushing2D-v0 environment (see Figure 2 for the full algorithm). The motivation behind hindsight experience replay is that even if an episode did not successfully reach the goal, we can still use it to learn something useful about the environment. To do so, we turn a problem that usually has sparse rewards into one with less sparse rewards by hallucinating different goal states that would hopefully provide non-zero reward given the actions that we took in an episode and add those to the experience replay buffer.

To use HER in our setup, set hindsight=True in the train method. For this part, you will need to implement the add hindsight replay experience function. To help you form new transitions to add to the replay, the code for the Pushing2D-v0 environment provides a method, apply hindsight, to compute the reward given a new goal state (<em>r</em>(·) in Fig. 2) and to modify each state to set the goal to be the state actually reached.

<ol>

 <li>(5 pts) Describe the hyperparameter settings that you used to train DDPG with HER. Ideally, these should match the hyperparameters you used in Part 1 so we can isolate the impact of the HER component.</li>

 <li>(15 pts) Plot the mean cumulative reward: Every <em>k </em>episodes, freeze the current cloned policy and run 100 test episodes, recording the mean/std of the cumulative reward. Plot the mean cumulative reward <em>µ </em>on the y-axis with ±<em>σ </em>standard deviation as errorbars vs. the number of training episodes on the x-axis. Do this on the same axes as the curve from Part 1 so that you can compare the two curves.</li>

 <li>(5 pts) How does the learning curve for DDPG+HER compare to that for DDPG?</li>

 <li>(10 pts) In the typical multi-goal RL setup, we are given a distribution over goals, <em>p</em>(<em>g</em>). HER trains on a different distribution over goals, ˆ<em>p</em>(<em>g</em>). Mathematically define <em>p</em>ˆ(<em>g</em>). When will ˆ<em>p</em>(<em>g</em>) be very different from <em>p</em>(<em>g</em>)? Why might a big difference between <em>p</em>ˆ(<em>g</em>) and <em>p</em>(<em>g</em>) be problematic?</li>

 <li>(Extra credit: up to 2 pts) How might you solve this distribution shift problem?</li>

 <li>(5 pts) What are settings where you cannot apply HER, or where HER would not be able to use it to speed up training?</li>

</ol>

Figure 2: Hindsight Experience Replay [1].

<a href="#_ftnref1" name="_ftn1">[1]</a> Pushing2D-v0 is considered solved if your implementation can reach the <em>original </em>goal at least 95% of the time. Note that this is not the same as reaching the <em>hindsight </em>goal.
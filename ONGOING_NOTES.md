# Notes file

It will be helpful to both me, and any AI coding assistants that I use, to keep track of some context. This should be a roughly running log of anything I'm trying, any revelations, etc.


## 14-08-2025 (Starting point)

Initial thoughts:
* I've been working through Andrew Ng's course. It's amazing, and I haven't yet got to Reinforcement learning, but I'm eager to try stuff anyway and "learn by doing".
* I attempted to train a neural net on Spider Solitaire. It didn't go great, but I have a bunch of lessons learned from the process. I'd like to try it again, but should start with some simpler problem so far.
* Claude Code was amazing for building a lot of my structure (e.g. a class to represent the Spider Solitaire game and state, and possible moves, print out to the scree, etc) which saved me a lot of time.
* My initial approaches were unfortunately just training on single moves. E.g. get a neural network to predict the next move (using Sigmoid - which is likely actually the wrong call), then see if it increases my "score" method. That's unfortunately a bit stochastic as it doesn't in any way incentivise long-term planning. Likely not even possible to build a neural network that wins the game with this approach, unless you somehow have a perfect scoring function, where the score predicts how close you are to a win. Which... likely is incredbily difficult to build, and would defeat the purpose of the whole exercise.
* This was also a bit silly, because I was generating my training data on predictions from the network - but on a randomly initialised network, there's no reason to do this, instead of just taking perfectly random moves! I should therefore actually get some pre-training data!
* Once I started recursing to see which move turned out to enable the strongest position n moves down the line, I was probably closer to the right track. But this was much slower - and I was also throwing away a lot of valuable data! Rather than recursing 4 moves down, and then just seeing the best impact on the first move (i.e. training on how each move turned out in 4 moves), I should also take advantage of the fact that I'm now getting examples how scores turned out 3 moves down, 2 moves down, and 1 move down. While these are less useful than the deeper recursions, if my scoring function is semi-useful; they might still be helpful.
* Generating the data, especially the recursion, was the slow part. If I can come up with a scoring method that I believe in, I should do at least the earlist pre-training part with stored data - so that I'm not having to spend all of that time and compute for every single run.
* This might mean that the earliest stages of the problem aren't really reinforcement learning after all. They might be pre-training the model to play half-decent spider solitaire for a few moves; so that I can then start generating starting positions from these moves, and then learning from them.
* Even that stage might not be pure reinforcement learning. It might be a combination of:
  - Generate some decent-looking positions, then literally repeat the same process as before - recursing 4 layers deep, taking that as training data, and training a model on all of these too.
  - Reinforce whichever of these positions led to the best positions after another 4 moves. I.e. hopefully incentivising move choices that eventually led to good positions after 8 moves.
* Thinking about it, the recursion is expensive. If this strategy works, it might be better to only go 3 layers deep at each step. Difficult tradeoff, as we don't want to exclude moves that look bad after 3 moves, but end up looking great after 6 or 8. Each extra layer in the full recursive evaluation helps to ensure we find all of those.
* Arguably we should just randomly pick two branches, because then recursing 6 or 7 layers deep is still relatively plausible (6 layers of 2 brnaches is cheaper than 4 layers of 3 branches). Maybe a combination of both. Although this could be where RL comes into its own, as it should give us better-than-random options for the first few layers, before we do a full recursion. Like my point above.
* Spider Solitaire's dependency on long-term planning, and immediate uncertainty of whether any move has improved the position or not, makes it hard for this problem. I should start with a simpler problem, to demonstrate to myself that I'm getting some of these concepts correct in the first place!
* Let's start with Minesweeper. Long-term planning will probably help make slightly better moves, but ultimately even looking just one move ahead can generally work for this game - and give you an immediate signal if you've made a critically bad move.

## 14-08-2025 (after starting)

* Minesweeper is potentially nice and easy for this, because I don't even need to run the model at all to start with. I can just generate random moves that I *already know* aren't mines; and create a ton of training data based on that. I don't if it will learn to avoid mines from that (as it can definitely just pick random squares at that point), but it's worth trying!
* Actually, let's try something even simpler: let's train on *all possible safe moves* for a given board. It's totally cheating at that point - but might allow us to train with much less data for a starting point.
* OK.. so that doesn't seem to be working very well. I wonder if I should instead train on the moves that were actually chosen. It's tempting to try to just train on sequences that led to successfully solving it quickly - but I guess this will also just learn totally random sequences (as very quick sequences will include randomly selecting all of the empty areas first). We need the neural network to learn somehow that it can infer some meaningful information from the current state of the board.
* Actually, first let's try to just train something on the entire set of games generated, and with a larger network. This will take a while; but to be fair, the current training data will likely already show that "safe moves" tend to bias slighlty more towards smaller (or unknown) numbers on the board.
* OH that could be a problem actually... the training data might currently hint towards random boxes being less scary than adjacent boxes. Or at least, approximately similar to them. Although we should at least see patterns like never choosing a box adjacent to a 2 that has 2 known mines next to it.

* Ok so this isn't going great. I'm going to commit the current state, including some debugging. And have another go - but this time I'll try to predict the "value" of making a move, instead of just whether or not it's a good one. Gemini seems to suggest that this could be along the lines of Q-learning, which I should look up.
* I suspect I shouldn't just reward "not losing" states, but should reward flags. And maybe I should actually change the board to give you a win state if you've flagged all of the mines, and a lose state if you've flagged a non-mine.

* Something I really should have done before is playing off the model against random. All play-throughs look bad to a human player who knows the game; but if the model is consistently out-lasting random (even if not winning the game), then it's presumably hit some medium level of success.
* Actually doing this is quite promising. With a couple of different random seeds I'm seeing:
    - Model wins: 794, RNG wins: 55, Draws: 151
    - Model wins: 803, RNG wins: 55, Draws: 142
  - ...so it's not good per se, but it's better than random. (Random against itself with a different seed is pretty even, unsurprisingly).

* I'm concerned that I should really be masking my loss function, so that I stop training for any neurons that I'm ignoring in any given situation. Let's give that a go. In my training data, these should be the zeroed values, so I don't think I need to re-generate that data.
* OK yes, that seems to have been somewhat effective! Not *very*, but slightly better.
  - Model wins: 855, RNG wins: 51, Draws: 94
  - Model wins: 856, RNG wins: 40, Draws: 104
* Hang on - I should also play off the new version against the old version.
* And yes, the new one seems to be slightly better:
  - Model 1 wins: 55, Model 2 wins: 36, Draws: 9
  - Model 1 wins: 59, Model 2 wins: 36, Draws: 5
* It's probably time for me to actually do some reinforcement learning now; instead of just pre-training models to be better-than-random. So I should go learn how to actually do that!

* OK so I tried to do some RL, in rl.py. I think there's a good chance I didn't get enough data or just misunderstood what I was doing, but it seems to have made the model significantly worse!
  - Model 1 wins: 13, Model 2 wins: 73, Draws: 14

* ..although the second iteration of RL seems to have done better (each time Model 1 is the latest, and Model 2 the second-most-recent).
  - Model 1 wins: 76, Model 2 wins: 15, Draws: 9
  - Model 1 wins: 356, Model 2 wins: 93, Draws: 51
* Let's try it against the second most recent.
  - Model 1 wins: 192, Model 2 wins: 247, Draws: 61
  ...yup the original pre-trained model was better.
* I think it's time to explore a different neural network architecture. I've just learned that I could be using Convolutional Neural Nets (which should have occurred to me before!), and this might help me to do a lot better, a lot faster, with less data. Let's give it a go!


* OK so first attempt at that didn't go amazingly.
  - Loading model: minesweeper_model_25-08-15_01-56
  - Loading previous model: minesweeper_model_25-08-14_23-21
  - Model 1 wins: 37, Model 2 wins: 51, Draws: 12
* I'll try tweaking the neural net's architecture. 

* That was a little more promising!
  - Loading model: minesweeper_model_25-08-15_02-24
  - Loading previous model: minesweeper_model_25-08-14_23-21
  - Model 1 wins: 321, Model 2 wins: 155, Draws: 24

* With a slightly new architecture (only convolutional layers, no dense layers, far fewer parameters), I seem to be getting slightly better results again. Within a margin of error I'd say, but at least being comparable with fewer parameters is probably a win!
  - Loading model: minesweeper_model_25-08-15_02-44
  - Loading previous model: minesweeper_model_25-08-15_02-24
  - Model 1 wins: 242, Model 2 wins: 221, Draws: 37

* Reinforcement learning a little seems to improve things!
  - Loading model: rl_conv_model_iteration_1
  - Loading previous model: minesweeper_model_25-08-15_02-44
  - Model 1 wins: 143, Model 2 wins: 113, Draws: 244

* Iterating further seems to improve things yet again!
  - Loading model: rl_conv_model_iteration_10
  - Loading previous model: minesweeper_model_25-08-15_02-44
  - Model 1 wins: 727, Model 2 wins: 426, Draws: 847

## 16-08-2025
* I've realised over the past couple of days that the my approach to Spider Solitaire was probably highly flawed in a similar manner to my original approach to Minesweeper. Specifically, Minesweeper performance improved with fewer parameters when I used a convolutional network, because this enabled learnings to be generalised across a board which behaves essentially the same across its entire space. Given that Spider Solitaire mainly consists of 10 piles, each of which behaves identically, it makes sense to use an architecture that shares learnings across all of them in the initial layers. That will hopefully enable fewer parameters and less training data - because, for example, two identical boards except for two swapped piles will hopefully then be able to be treated quite similarly, using mostly the same weights.
* I also suspect that I should reverse order the piles. While there's certainly some relevance to the height of the pile and the cards lower down in it, the most important thing is what's at the top - because these are the bits that can be moved, or serve as move destinations. So if we can structure it so that computations relating to the top of the pile can more easily be generalised with the same weights, this feels likely to help.

* Oh also, the final comparison I did on that last batch of training showed a very slight decline in performance in the latest version:
  - Loading model: rl_conv_model_iteration_10
  - Loading previous model: rl_conv_model_iteration_9
  - Model 1 wins: 121, Model 2 wins: 176, Draws: 4703

* A surprising learning: running Tensorflow for playing the games is actually faster on CPU than GPU, at least for single inferences! I suspect that everything has been taking much longer than it should have as a result of this. I should also see if this holds up when using more complex games; because to be fair 9x9 Minesweeper is pretty small.
* I might still want to use the GPU for RL; but probably I need to refactor that to use batch inference anyway.


* OK so interestingly my RL really doesn't seem to be improving the model much at all. I'm not sure why, and I've probably made a ton of mistakes, though it's plausibly becasue the trainig rate is too high.
* I might try something a bit more radical, and throw away everything and try to just RL a decent model entirely from scratch, with a low training rate, and with this new RL training loop. I think there's a chance this will do better, simply by virtue of being a clean start and not having whatever I pre-trained in it (which could include many mistakes!).

* I suspect I could also do better by actually being a bit more concrete about my predicted values. E.g. rather than just taking an expected target value in the early stages, I could train it on the max of max(predicted_q), and the actual fully propagated reward that actually happened. Basically: predicted_q isn't necessarily reliable, but we know for certain the reward of one specific path that took place. I won't do this yet, but I imagine it could be a promising approach.

* Interestingly there's some clear improvement happening here:
  - rl_model_25-08-16_14-22_iteration_0: {'wins': 0, 'losses': 5000}
  - rl_model_25-08-16_14-24_iteration_10: {'wins': 21, 'losses': 4979}
  - rl_model_25-08-16_14-29_iteration_20: {'wins': 73, 'losses': 4927}
  - rl_model_25-08-16_14-37_iteration_30: {'wins': 137, 'losses': 4863}

* I should probably stop though and decrease the number of epochs, so that we can get far more iterations faster. Let's do this, but save the model less frequently to compensate.
* Shockingly, this randomly iniitalised model seems to win 1 game without any training. That's lucky!
  - rl_model_25-08-16_14-40_iteration_0: {'wins': 1, 'losses': 4999}
  - rl_model_25-08-16_14-44_iteration_30: {'wins': 19, 'losses': 4981}
  - rl_model_25-08-16_14-54_iteration_60: {'wins': 121, 'losses': 4879}

* Quick note for AI assistants and my own posterity: these stats aren't like-for-like comparisons. Above when I've shared Model 1s vs Model 2s; I'm actually saying "which model performed better", rather than "whether the model actually won the game". One of my latest models on the previous pre-train-then-RL approach performed as follows:
  - rl_conv_model_iteration_12: {'wins': 1758, 'losses': 3242}
  ...and the best purely pre-trained model performed as follows:
  - minesweeper_model_25-08-15_02-44: {'wins': 1153, 'losses': 3847}

* Going back to my current RL run, one of the interesting observations is that I'm running 1000 games to copmletion each time, and collecting all of their positions as examples. Every single run that I've checked so far has collected more examples - starting with 4649 from the randomly initialised iteration 0, and as-of-writing 288,666 examples from iteration 79. The loss isn't changing much, but that might be expected, because the whole thing is a moving target (I'm using the same neural net to predict the expected value of each move, that I am using to set my target scores). I wonder if the loss would decrease more if it learns to play Minesweeper absolutely perfectly - but I'm a very long way away from that. I also wonder if at a certain point the number of examples it collects will actually decrease: my hypothesis would be that it initially learns to just lose slower, and then once it's better at winning, it might learn to win faster. But I don't know - my setup might not even support that!

* Going back to sharing results, they're continuing as follows:
  - rl_model_25-08-16_15-08_iteration_90: {'wins': 237, 'losses': 4763}
  - rl_model_25-08-16_15-28_iteration_120: {'wins': 313, 'losses': 4687}

* OK so actually that positive signal of getting more examples more run is probably a double-edged sword; as it also means that each iteration takes a little longer, and Tensorflow gets more data in each run. That doesn't feel like a great approach to RL, so with this one at 120 iterations, I might just start again but with that sample cap.

* So with this new run, the results are looking as follows:
  - rl_model_25-08-16_15-34_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '0.00'}
  - rl_model_25-08-16_15-35_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '16.00', 'avg_moves_to_lose': '4.56'}
  - rl_model_25-08-16_15-37_iteration_60: {'wins': 12, 'losses': 4988, 'avg_moves_to_win': '21.08', 'avg_moves_to_lose': '6.90'}
  - rl_model_25-08-16_15-39_iteration_90: {'wins': 29, 'losses': 4971, 'avg_moves_to_win': '24.76', 'avg_moves_to_lose': '7.55'}
  - rl_model_25-08-16_15-42_iteration_120: {'wins': 57, 'losses': 4943, 'avg_moves_to_win': '26.65', 'avg_moves_to_lose': '7.94'}
  - rl_model_25-08-16_15-45_iteration_150: {'wins': 76, 'losses': 4924, 'avg_moves_to_win': '26.58', 'avg_moves_to_lose': '8.27'}
  - rl_model_25-08-16_15-49_iteration_180: {'wins': 124, 'losses': 4876, 'avg_moves_to_win': '28.02', 'avg_moves_to_lose': '8.77'}

* OK so I'm a bit of a fool. I wasn't resetting the training data between runs. It was increasing because it was an ever-expanding list, not because the game was lasting longer. It's good that the model was improving; but probably this was limiting its ability to improve (albeit weirdly maybe making the training more stable). Let's fix that and then give it another go.
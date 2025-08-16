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
  - rl_model_25-08-16_15-54_iteration_0: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '9.50', 'avg_moves_to_lose': '5.63'}
  - rl_model_25-08-16_15-56_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.60'}
  - rl_model_25-08-16_15-57_iteration_60: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.60'}

* OK I'm increasing NUM_IN_RUN to try to address this - maybe it's just too little data per iteration.
  - rl_model_25-08-16_15-58_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '6.34'}
  - rl_model_25-08-16_16-07_iteration_30: {'wins': 8, 'losses': 4992, 'avg_moves_to_win': '20.38', 'avg_moves_to_lose': '6.77'}

* And now I'm gonna increase my BATCH_SIZE to 500 because this is a bit slow per batch.
  - rl_model_25-08-16_16-13_iteration_30: {'wins': 10, 'losses': 4990, 'avg_moves_to_win': '19.30', 'avg_moves_to_lose': '6.92'}
  - rl_model_25-08-16_16-18_iteration_60: {'wins': 9, 'losses': 4991, 'avg_moves_to_win': '22.56', 'avg_moves_to_lose': '7.32'}
  - rl_model_25-08-16_16-23_iteration_90: {'wins': 24, 'losses': 4976, 'avg_moves_to_win': '19.92', 'avg_moves_to_lose': '7.57'}
  - rl_model_25-08-16_16-28_iteration_120: {'wins': 25, 'losses': 4975, 'avg_moves_to_win': '21.04', 'avg_moves_to_lose': '7.93'}
  - rl_model_25-08-16_16-33_iteration_150: {'wins': 34, 'losses': 4966, 'avg_moves_to_win': '22.97', 'avg_moves_to_lose': '8.08'}
  - rl_model_25-08-16_16-38_iteration_180: {'wins': 29, 'losses': 4971, 'avg_moves_to_win': '24.34', 'avg_moves_to_lose': '8.32'}
  - rl_model_25-08-16_16-44_iteration_210: {'wins': 35, 'losses': 4965, 'avg_moves_to_win': '25.34', 'avg_moves_to_lose': '8.57'}
  - rl_model_25-08-16_16-49_iteration_240: {'wins': 49, 'losses': 4951, 'avg_moves_to_win': '26.16', 'avg_moves_to_lose': '8.71'}

* OK I'm going to try changing my approach a little. I'll delete all of my existing models to clear things out, and then I'm going to actually change the input format to have 3 dimensions: one says is the cell visible or not, one says if it is flagged or not, and one says how many adjacent mines there are. This will hopefully give the network less to learn (as it won't need to waste time on interpreting these facts from the input).
  - rl_model_25-08-16_17-07_iteration_0: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '10.00', 'avg_moves_to_lose': '4.79'}
  - rl_model_25-08-16_17-12_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.60'}
  - rl_model_25-08-16_17-17_iteration_60: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}
  - rl_model_25-08-16_17-22_iteration_90: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}

* OK so this version isn't working at all. I must be doing something quite wrong here!
* Hmm so one quick bug - not the cause I suspect - but if the first move was a flag, there would be no mines for it! Have fixed that. And I've attempted to calculate the average value for non-zero adajacent numbers of mines (seems to be about 1.4) so I'm roughly normalising this field in the input vector by dividing it by 1.5.
  - rl_model_25-08-16_17-25_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '1.58'}
  - rl_model_25-08-16_17-30_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.60'}

* From a _weirdly terrible_ baseline of losing in 1.58 moves, 3.6 is an improvement. Let's see where it goes from there.
  - rl_model_25-08-16_17-35_iteration_60: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.60'}
  - rl_model_25-08-16_17-40_iteration_90: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}

* I've improved a few inefficiencies by finding vectorised way of doing stuff that had previously been in Python loops; and keeping a running track of the remaining valid moves, instead of having to recalculate them every time. Doesn't really help the main issue, but hopefully will help to accelerate future training runs (although the time is dominated by inference and training at this point).

* I'm going to give it another go now, but also increase the batch size, and make the model itself smaller. I'll commit first and then make those tweaks.
  - rl_model_25-08-16_18-35_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '1.78'}
  - rl_model_25-08-16_18-39_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '14.50', 'avg_moves_to_lose': '5.73'}
  - rl_model_25-08-16_18-43_iteration_60: {'wins': 7, 'losses': 4993, 'avg_moves_to_win': '19.43', 'avg_moves_to_lose': '6.25'}
  - rl_model_25-08-16_18-47_iteration_90: {'wins': 15, 'losses': 4985, 'avg_moves_to_win': '23.60', 'avg_moves_to_lose': '6.47'}
  - rl_model_25-08-16_18-51_iteration_120: {'wins': 19, 'losses': 4981, 'avg_moves_to_win': '23.53', 'avg_moves_to_lose': '7.03'}
  - rl_model_25-08-16_18-55_iteration_150: {'wins': 22, 'losses': 4978, 'avg_moves_to_win': '23.09', 'avg_moves_to_lose': '7.07'}
  - rl_model_25-08-16_18-59_iteration_180: {'wins': 26, 'losses': 4974, 'avg_moves_to_win': '24.04', 'avg_moves_to_lose': '7.24'}
  - rl_model_25-08-16_19-03_iteration_210: {'wins': 24, 'losses': 4976, 'avg_moves_to_win': '24.42', 'avg_moves_to_lose': '7.28'}

* OK I think this model is too small. I wanted to shrink it - and I did - but way too much. I'll try one that's still a bit smaller than the previous ones, but bigger than this. Especially 4 neurons in the penultimate layer feels meagre.

* Slightly larger model.
  - rl_model_25-08-16_19-10_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.24'}
  - rl_model_25-08-16_19-14_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}

* Very slight improvement, but training this is taking ages. I'm gonna increase my batch size.
  - rl_model_25-08-16_19-17_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '1.73'}

* Maybe I should go back to some sort of check-pointing. I initially got rid of it because it made everything feel a bit less reproducible - but I could easily fix that by saving slightly more state!

  - rl_model_25-08-16_19-20_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}
  - rl_model_25-08-16_19-23_iteration_60: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}
  - rl_model_25-08-16_19-26_iteration_90: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.59'}

* This isn't looking good at all! Let's try to change things up a bit. One of Claude's suggestions is usign leaky_relu. It says to avoid dead neurons, and I guess is inspired by the static performance here, although I'd be a little surprised if that were the issue. It also says I may have just got unlucky on the initialisation lottery. I might try the latter first - just reinitialise the same thing - and then try more model weights or Leaky ReLu.

* Same approach, just reinitialised with new random weights:
  - rl_model_25-08-16_19-31_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '2.90'}
  - rl_model_25-08-16_19-34_iteration_30: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '6.00', 'avg_moves_to_lose': '5.63'}
  - rl_model_25-08-16_19-38_iteration_60: {'wins': 23, 'losses': 4977, 'avg_moves_to_win': '17.57', 'avg_moves_to_lose': '8.53'}
  - rl_model_25-08-16_19-42_iteration_90: {'wins': 70, 'losses': 4930, 'avg_moves_to_win': '20.97', 'avg_moves_to_lose': '9.54'}
  - rl_model_25-08-16_19-45_iteration_120: {'wins': 149, 'losses': 4851, 'avg_moves_to_win': '23.56', 'avg_moves_to_lose': '10.59'}

* Wait... what on earth does avg_moves_to_win of 6.0 mean? We need 10 flags at least to make that happen. Oh unless it was just a very specifically laid-out minesweeper board with a huge empty space. I expect that was probably it. And a pretty lucky model, in that case.

  - rl_model_25-08-16_19-49_iteration_150: {'wins': 233, 'losses': 4767, 'avg_moves_to_win': '24.47', 'avg_moves_to_lose': '11.25'}
  - rl_model_25-08-16_19-53_iteration_180: {'wins': 260, 'losses': 4740, 'avg_moves_to_win': '24.74', 'avg_moves_to_lose': '11.99'}
  - rl_model_25-08-16_19-58_iteration_210: {'wins': 294, 'losses': 4706, 'avg_moves_to_win': '25.55', 'avg_moves_to_lose': '12.18'}
  - rl_model_25-08-16_20-02_iteration_240: {'wins': 396, 'losses': 4604, 'avg_moves_to_win': '25.92', 'avg_moves_to_lose': '12.68'}
  - rl_model_25-08-16_20-06_iteration_270: {'wins': 496, 'losses': 4504, 'avg_moves_to_win': '26.27', 'avg_moves_to_lose': '12.84'}
  - rl_model_25-08-16_20-11_iteration_300: {'wins': 546, 'losses': 4454, 'avg_moves_to_win': '26.85', 'avg_moves_to_lose': '13.18'}
  - rl_model_25-08-16_20-16_iteration_330: {'wins': 563, 'losses': 4437, 'avg_moves_to_win': '26.49', 'avg_moves_to_lose': '13.51'}
  - rl_model_25-08-16_20-21_iteration_360: {'wins': 641, 'losses': 4359, 'avg_moves_to_win': '26.83', 'avg_moves_to_lose': '13.44'}
  - rl_model_25-08-16_20-26_iteration_390: {'wins': 691, 'losses': 4309, 'avg_moves_to_win': '26.63', 'avg_moves_to_lose': '13.89'}
  - rl_model_25-08-16_20-30_iteration_420: {'wins': 772, 'losses': 4228, 'avg_moves_to_win': '26.88', 'avg_moves_to_lose': '13.80'}
  - rl_model_25-08-16_20-35_iteration_450: {'wins': 801, 'losses': 4199, 'avg_moves_to_win': '26.71', 'avg_moves_to_lose': '13.96'}
  - rl_model_25-08-16_20-40_iteration_480: {'wins': 877, 'losses': 4123, 'avg_moves_to_win': '26.90', 'avg_moves_to_lose': '13.80'}
  - rl_model_25-08-16_20-45_iteration_510: {'wins': 878, 'losses': 4122, 'avg_moves_to_win': '27.12', 'avg_moves_to_lose': '13.80'}
  - rl_model_25-08-16_20-50_iteration_540: {'wins': 909, 'losses': 4091, 'avg_moves_to_win': '26.82', 'avg_moves_to_lose': '13.98'}
  - rl_model_25-08-16_20-55_iteration_570: {'wins': 955, 'losses': 4045, 'avg_moves_to_win': '26.85', 'avg_moves_to_lose': '13.86'}
  - rl_model_25-08-16_20-59_iteration_600: {'wins': 966, 'losses': 4034, 'avg_moves_to_win': '26.81', 'avg_moves_to_lose': '14.11'}
  - ...
  - rl_model_25-08-16_21-42_iteration_870: {'wins': 1061, 'losses': 3939, 'avg_moves_to_win': '26.90', 'avg_moves_to_lose': '14.24'}
  - rl_model_25-08-16_21-47_iteration_900: {'wins': 1077, 'losses': 3923, 'avg_moves_to_win': '26.67', 'avg_moves_to_lose': '14.39'}

  * This seems to have plateaued a bit. I'll commit this, and then try a larger network with leaky_relu.
  - rl_model_25-08-16_21-53_iteration_0: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '14.00', 'avg_moves_to_lose': '2.72'}
  - rl_model_25-08-16_21-56_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '10.50', 'avg_moves_to_lose': '4.79'}
  - rl_model_25-08-16_22-00_iteration_60: {'wins': 19, 'losses': 4981, 'avg_moves_to_win': '19.32', 'avg_moves_to_lose': '8.58'}
  - rl_model_25-08-16_22-04_iteration_90: {'wins': 72, 'losses': 4928, 'avg_moves_to_win': '21.06', 'avg_moves_to_lose': '9.47'}
  - rl_model_25-08-16_22-08_iteration_120: {'wins': 172, 'losses': 4828, 'avg_moves_to_win': '23.48', 'avg_moves_to_lose': '10.04'}
  - rl_model_25-08-16_22-12_iteration_150: {'wins': 308, 'losses': 4692, 'avg_moves_to_win': '24.36', 'avg_moves_to_lose': '10.78'}
  - rl_model_25-08-16_22-16_iteration_180: {'wins': 411, 'losses': 4589, 'avg_moves_to_win': '25.31', 'avg_moves_to_lose': '11.49'}
  - rl_model_25-08-16_22-21_iteration_210: {'wins': 480, 'losses': 4520, 'avg_moves_to_win': '25.06', 'avg_moves_to_lose': '11.66'}
  - rl_model_25-08-16_22-26_iteration_240: {'wins': 498, 'losses': 4502, 'avg_moves_to_win': '25.35', 'avg_moves_to_lose': '12.16'}
  - rl_model_25-08-16_22-30_iteration_270: {'wins': 463, 'losses': 4537, 'avg_moves_to_win': '24.94', 'avg_moves_to_lose': '12.04'}
  - rl_model_25-08-16_22-35_iteration_300: {'wins': 475, 'losses': 4525, 'avg_moves_to_win': '25.21', 'avg_moves_to_lose': '12.21'}
  - rl_model_25-08-16_22-40_iteration_330: {'wins': 479, 'losses': 4521, 'avg_moves_to_win': '25.54', 'avg_moves_to_lose': '12.32'}
  - rl_model_25-08-16_22-45_iteration_360: {'wins': 487, 'losses': 4513, 'avg_moves_to_win': '25.28', 'avg_moves_to_lose': '12.64'}
  - rl_model_25-08-16_22-50_iteration_390: {'wins': 548, 'losses': 4452, 'avg_moves_to_win': '25.52', 'avg_moves_to_lose': '12.75'}
  - rl_model_25-08-16_22-54_iteration_420: {'wins': 528, 'losses': 4472, 'avg_moves_to_win': '25.67', 'avg_moves_to_lose': '12.77'}
  - rl_model_25-08-16_22-59_iteration_450: {'wins': 545, 'losses': 4455, 'avg_moves_to_win': '25.42', 'avg_moves_to_lose': '12.91'}
  - rl_model_25-08-16_23-04_iteration_480: {'wins': 534, 'losses': 4466, 'avg_moves_to_win': '25.76', 'avg_moves_to_lose': '13.16'}
  - rl_model_25-08-16_23-09_iteration_510: {'wins': 536, 'losses': 4464, 'avg_moves_to_win': '25.46', 'avg_moves_to_lose': '13.13'}
  - rl_model_25-08-16_23-14_iteration_540: {'wins': 571, 'losses': 4429, 'avg_moves_to_win': '25.82', 'avg_moves_to_lose': '13.33'}
  - rl_model_25-08-16_23-19_iteration_570: {'wins': 561, 'losses': 4439, 'avg_moves_to_win': '25.75', 'avg_moves_to_lose': '13.23'}
  - rl_model_25-08-16_23-24_iteration_600: {'wins': 587, 'losses': 4413, 'avg_moves_to_win': '25.67', 'avg_moves_to_lose': '13.37'}
  
* This seems to be plateauing, despite still continuing to gain a bit. Let's try something else. I'll commit this and then tweak the model architecture a bit.

* OK I'm making my early convolutional layers look slightly further away, and then I'm adding some dense layers at the end. This feels a bit trial-end-errory and I couldn't say if I think it's likely to work; but I could see an argument that some dense layers could enable some full-board planning and understanding that can't be achieved with pure convolutional layers.
* On the other hand, this current architecture design has 159973 parameters - which is a massive explosion from the previous; and it's basically all in the dense layers. I struggle to see this helping to speed up learning, as we might just need so much more training data to make progress. In fact maybe ReLu would have been a better choice here, because vanishing gradients could potentially help it to learn a bit faster.

  - rl_model_25-08-16_23-33_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '1.87'}
  - rl_model_25-08-16_23-37_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '8.50', 'avg_moves_to_lose': '5.30'}
  - rl_model_25-08-16_23-41_iteration_60: {'wins': 4, 'losses': 4996, 'avg_moves_to_win': '6.50', 'avg_moves_to_lose': '5.57'}
  - rl_model_25-08-16_23-45_iteration_90: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '13.00', 'avg_moves_to_lose': '5.88'}
  - rl_model_25-08-16_23-49_iteration_120: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '12.50', 'avg_moves_to_lose': '6.42'}

* Yeah this isn't working. Also I realised that, despite the massive increase in parameters, I actually have fewer neurons in the two penultimate layers than in the output layer. Which seems counterintuitive, and like they'd have to learn some strange tricks for it to actually work.
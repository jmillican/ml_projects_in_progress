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

* Let's try another architecture back to pure convolutional layers, but with larger masks, more filters, and another layer. This has 21538 parameters - so a fair few more than before - but far fewer than when I had dense layers.
  - rl_model_25-08-16_23-54_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '1.34'}

* There's an interesting pattern where it seems to be alternately getting around 20k and then around 4k training examples from each iteration. I suspect this means we're in some sort of pathological situation, perhaps where each iteration has it learning the opposite lessons. I'll let it play out for a bit, but I'm not sure if it's promising.

  - rl_model_25-08-16_23-57_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '6.00', 'avg_moves_to_lose': '5.95'}
  - rl_model_25-08-17_00-02_iteration_60: {'wins': 12, 'losses': 4988, 'avg_moves_to_win': '18.58', 'avg_moves_to_lose': '7.53'}

* OK that siutation seems to have resolved itself, and it seems to slowly be learning roughly how to win and lose the game.

  - rl_model_25-08-17_00-05_iteration_90: {'wins': 28, 'losses': 4972, 'avg_moves_to_win': '20.04', 'avg_moves_to_lose': '8.55'}
  - rl_model_25-08-17_00-09_iteration_120: {'wins': 60, 'losses': 4940, 'avg_moves_to_win': '20.13', 'avg_moves_to_lose': '9.20'}
  - rl_model_25-08-17_00-13_iteration_150: {'wins': 132, 'losses': 4868, 'avg_moves_to_win': '22.23', 'avg_moves_to_lose': '10.04'}
  - rl_model_25-08-17_00-17_iteration_180: {'wins': 170, 'losses': 4830, 'avg_moves_to_win': '22.59', 'avg_moves_to_lose': '10.62'}
  - rl_model_25-08-17_00-21_iteration_210: {'wins': 202, 'losses': 4798, 'avg_moves_to_win': '23.06', 'avg_moves_to_lose': '11.37'}
  - rl_model_25-08-17_00-25_iteration_240: {'wins': 243, 'losses': 4757, 'avg_moves_to_win': '23.90', 'avg_moves_to_lose': '11.82'}
  - rl_model_25-08-17_00-29_iteration_270: {'wins': 281, 'losses': 4719, 'avg_moves_to_win': '24.33', 'avg_moves_to_lose': '12.30'}
  - rl_model_25-08-17_00-33_iteration_300: {'wins': 355, 'losses': 4645, 'avg_moves_to_win': '24.93', 'avg_moves_to_lose': '12.90'}
  - rl_model_25-08-17_00-37_iteration_330: {'wins': 413, 'losses': 4587, 'avg_moves_to_win': '25.34', 'avg_moves_to_lose': '13.16'}
  - rl_model_25-08-17_00-41_iteration_360: {'wins': 426, 'losses': 4574, 'avg_moves_to_win': '25.63', 'avg_moves_to_lose': '13.43'}
  - rl_model_25-08-17_00-46_iteration_390: {'wins': 464, 'losses': 4536, 'avg_moves_to_win': '25.16', 'avg_moves_to_lose': '13.44'}
  - rl_model_25-08-17_00-50_iteration_420: {'wins': 587, 'losses': 4413, 'avg_moves_to_win': '25.91', 'avg_moves_to_lose': '13.69'}
  - rl_model_25-08-17_00-54_iteration_450: {'wins': 531, 'losses': 4469, 'avg_moves_to_win': '25.29', 'avg_moves_to_lose': '13.71'}
  - rl_model_25-08-17_00-58_iteration_480: {'wins': 613, 'losses': 4387, 'avg_moves_to_win': '25.68', 'avg_moves_to_lose': '14.10'}
  - rl_model_25-08-17_01-02_iteration_510: {'wins': 636, 'losses': 4364, 'avg_moves_to_win': '25.68', 'avg_moves_to_lose': '14.29'}
  - rl_model_25-08-17_01-07_iteration_540: {'wins': 643, 'losses': 4357, 'avg_moves_to_win': '25.72', 'avg_moves_to_lose': '14.26'}
  - rl_model_25-08-17_01-11_iteration_570: {'wins': 664, 'losses': 4336, 'avg_moves_to_win': '25.70', 'avg_moves_to_lose': '14.55'}

* OK so this continued to progress, but it's learning quite a bit slower than other approaches I took. I'm going to commit it, then let's try a slightly smaller model.
* Actually this model is larger, but fewer layers - and much more in the very first layer. Hopefully to identify a bunch more possible filters that it needs to consider.

  - rl_model_25-08-17_01-17_iteration_0: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '10.00', 'avg_moves_to_lose': '4.24'}
  - rl_model_25-08-17_01-20_iteration_30: {'wins': 12, 'losses': 4988, 'avg_moves_to_win': '19.92', 'avg_moves_to_lose': '8.54'}
  - rl_model_25-08-17_01-24_iteration_60: {'wins': 230, 'losses': 4770, 'avg_moves_to_win': '25.28', 'avg_moves_to_lose': '11.03'}
  - rl_model_25-08-17_01-28_iteration_90: {'wins': 297, 'losses': 4703, 'avg_moves_to_win': '26.01', 'avg_moves_to_lose': '12.63'}
  - rl_model_25-08-17_01-32_iteration_120: {'wins': 545, 'losses': 4455, 'avg_moves_to_win': '27.40', 'avg_moves_to_lose': '13.15'}
  - rl_model_25-08-17_01-37_iteration_150: {'wins': 832, 'losses': 4168, 'avg_moves_to_win': '27.90', 'avg_moves_to_lose': '13.68'}
  - rl_model_25-08-17_01-42_iteration_180: {'wins': 887, 'losses': 4113, 'avg_moves_to_win': '27.92', 'avg_moves_to_lose': '13.85'}
  - rl_model_25-08-17_01-47_iteration_210: {'wins': 1119, 'losses': 3881, 'avg_moves_to_win': '28.09', 'avg_moves_to_lose': '13.95'}
  - rl_model_25-08-17_01-52_iteration_240: {'wins': 1085, 'losses': 3915, 'avg_moves_to_win': '27.75', 'avg_moves_to_lose': '14.32'}
  - rl_model_25-08-17_01-57_iteration_270: {'wins': 1190, 'losses': 3810, 'avg_moves_to_win': '27.62', 'avg_moves_to_lose': '14.41'}
  - ...
  - rl_model_25-08-17_02-57_iteration_630: {'wins': 1971, 'losses': 3029, 'avg_moves_to_win': '27.68', 'avg_moves_to_lose': '14.20'}
  - rl_model_25-08-17_03-02_iteration_660: {'wins': 2037, 'losses': 2963, 'avg_moves_to_win': '27.74', 'avg_moves_to_lose': '14.22'}
  - ...
  - rl_model_25-08-17_03-46_iteration_930: {'wins': 2145, 'losses': 2855, 'avg_moves_to_win': '27.94', 'avg_moves_to_lose': '14.12'}
  - ...
  - rl_model_25-08-17_04-20_iteration_1140: {'wins': 2246, 'losses': 2754, 'avg_moves_to_win': '28.09', 'avg_moves_to_lose': '14.02'}
  - ...
  - rl_model_25-08-17_09-14_iteration_2970: {'wins': 2288, 'losses': 2712, 'avg_moves_to_win': '28.01', 'avg_moves_to_lose': '13.78'}
  - rl_model_25-08-17_09-19_iteration_3000: {'wins': 2286, 'losses': 2714, 'avg_moves_to_win': '27.97', 'avg_moves_to_lose': '14.09'}
  - rl_model_25-08-17_09-23_iteration_3030: {'wins': 2289, 'losses': 2711, 'avg_moves_to_win': '28.03', 'avg_moves_to_lose': '13.80'}
  - ...
  - rl_model_25-08-17_10-40_iteration_3510: {'wins': 2280, 'losses': 2720, 'avg_moves_to_win': '27.89', 'avg_moves_to_lose': '14.06'}

## 17-08-2025

* OK so this has definitely plataued now. It's promising progress, but clearly not improving any more. I'll commit this and then try a larger architecture with a little more exploration.

* Larger convolutional layers, and then one final dense output layer at the end. Slightly more exploration; and discounting future benefits less.
  - rl_model_25-08-17_10-58_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '2.40'}
  - rl_model_25-08-17_11-02_iteration_30: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '6.00', 'avg_moves_to_lose': '5.41'}
  - rl_model_25-08-17_11-07_iteration_60: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '8.00', 'avg_moves_to_lose': '5.66'}
  - rl_model_25-08-17_11-11_iteration_90: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '10.50', 'avg_moves_to_lose': '5.56'}
  - rl_model_25-08-17_11-15_iteration_120: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '11.00', 'avg_moves_to_lose': '5.72'}
  - rl_model_25-08-17_11-19_iteration_150: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '13.17', 'avg_moves_to_lose': '6.41'}
  - rl_model_25-08-17_11-22_iteration_180: {'wins': 5, 'losses': 4995, 'avg_moves_to_win': '10.80', 'avg_moves_to_lose': '6.78'}
  - rl_model_25-08-17_11-26_iteration_210: {'wins': 8, 'losses': 4992, 'avg_moves_to_win': '18.50', 'avg_moves_to_lose': '6.94'}
  - ...
  - rl_model_25-08-17_11-41_iteration_330: {'wins': 25, 'losses': 4975, 'avg_moves_to_win': '17.28', 'avg_moves_to_lose': '8.14'}
  - ...
  - rl_model_25-08-17_11-59_iteration_480: {'wins': 50, 'losses': 4950, 'avg_moves_to_win': '18.32', 'avg_moves_to_lose': '8.75'}

* So this was progressing reasonably. Much slower than others, but it has far more parameters and was still learning. I wonder if I should actually reduce the size of the convolutional layers though - especially the last one, as this will help to reduce the massive increase in parameters we get between this and the dense layer.

* Let's try this 131081 parameter model.
  - rl_model_25-08-17_16-56_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '2.23'}
  - rl_model_25-08-17_17-00_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '9.50', 'avg_moves_to_lose': '5.68'}
  - rl_model_25-08-17_17-08_iteration_90: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '5.00', 'avg_moves_to_lose': '5.67'}
  - rl_model_25-08-17_17-11_iteration_120: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '11.00', 'avg_moves_to_lose': '6.38'}

* OK this isn't going great yet. Let's try again with just new parameters, and maybe give GeLu a go - just to see how it goes.
  - rl_model_25-08-17_17-13_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '2.10'}
  - rl_model_25-08-17_17-18_iteration_30: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '5.00', 'avg_moves_to_lose': '5.41'}
  - rl_model_25-08-17_17-23_iteration_60: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '4.00', 'avg_moves_to_lose': '5.56'}
  - rl_model_25-08-17_17-27_iteration_90: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '6.00', 'avg_moves_to_lose': '5.46'}
  - rl_model_25-08-17_17-32_iteration_120: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '28.00', 'avg_moves_to_lose': '5.37'}

* Yeah this isn't working. Let's just try doing the same again.
  - rl_model_25-08-17_17-33_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '2.29'}
  - rl_model_25-08-17_17-38_iteration_30: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '8.00', 'avg_moves_to_lose': '5.28'}
  - rl_model_25-08-17_17-42_iteration_60: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '8.00', 'avg_moves_to_lose': '5.46'}
  - rl_model_25-08-17_17-47_iteration_90: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '14.50', 'avg_moves_to_lose': '5.85'}
  - rl_model_25-08-17_17-51_iteration_120: {'wins': 4, 'losses': 4996, 'avg_moves_to_win': '12.25', 'avg_moves_to_lose': '6.64'}
  - rl_model_25-08-17_17-55_iteration_150: {'wins': 3, 'losses': 4997, 'avg_moves_to_win': '11.33', 'avg_moves_to_lose': '6.53'}
  - rl_model_25-08-17_17-59_iteration_180: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '16.00', 'avg_moves_to_lose': '7.00'}
  - rl_model_25-08-17_18-03_iteration_210: {'wins': 11, 'losses': 4989, 'avg_moves_to_win': '16.00', 'avg_moves_to_lose': '7.25'}
  - rl_model_25-08-17_18-07_iteration_240: {'wins': 27, 'losses': 4973, 'avg_moves_to_win': '18.78', 'avg_moves_to_lose': '7.66'}
  - rl_model_25-08-17_18-11_iteration_270: {'wins': 24, 'losses': 4976, 'avg_moves_to_win': '19.12', 'avg_moves_to_lose': '7.87'}
  - ...
  - rl_model_25-08-17_18-22_iteration_350: {'wins': 35, 'losses': 4965, 'avg_moves_to_win': '20.60', 'avg_moves_to_lose': '7.97'}
  - rl_model_25-08-17_18-29_iteration_400: {'wins': 31, 'losses': 4969, 'avg_moves_to_win': '20.39', 'avg_moves_to_lose': '8.56'}

* OK this seems to be progressing, but very slowly. I wonder if I should get global state instead by essentially just adding it to every single parameter in additional channels in the input later. Claude suggested this earlier when I asked about providing global state to them (I think calling it Broadcasting or something?) - and I initially dismissed it, because it's duplicative and feels wasteful; but I guess it probably wouldn't actually add more parameters, and the cost of duplication in the input layer seems minimal.
* This model is 27068 parameters - so a fair bit bigger than some of my earlier successes were looking; but far smaller than these large ones that were seeming to take ages.

  - rl_model_25-08-17_18-42_iteration_30: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '7.35'}
  - rl_model_25-08-17_18-46_iteration_60: {'wins': 3, 'losses': 4997, 'avg_moves_to_win': '15.00', 'avg_moves_to_lose': '8.47'}
  - rl_model_25-08-17_18-50_iteration_90: {'wins': 3, 'losses': 4997, 'avg_moves_to_win': '17.33', 'avg_moves_to_lose': '8.67'}
  - rl_model_25-08-17_18-54_iteration_120: {'wins': 6, 'losses': 4994, 'avg_moves_to_win': '16.67', 'avg_moves_to_lose': '9.08'}
  - rl_model_25-08-17_18-58_iteration_150: {'wins': 13, 'losses': 4987, 'avg_moves_to_win': '19.31', 'avg_moves_to_lose': '9.58'}
  - rl_model_25-08-17_19-02_iteration_180: {'wins': 29, 'losses': 4971, 'avg_moves_to_win': '18.83', 'avg_moves_to_lose': '9.98'}
  - rl_model_25-08-17_19-06_iteration_210: {'wins': 63, 'losses': 4937, 'avg_moves_to_win': '20.52', 'avg_moves_to_lose': '9.77'}
  - ...
  - rl_model_25-08-17_19-33_iteration_400: {'wins': 190, 'losses': 4810, 'avg_moves_to_win': '24.06', 'avg_moves_to_lose': '11.51'}
  - rl_model_25-08-17_19-40_iteration_450: {'wins': 239, 'losses': 4761, 'avg_moves_to_win': '24.77', 'avg_moves_to_lose': '12.28'}
  - rl_model_25-08-17_19-47_iteration_500: {'wins': 296, 'losses': 4704, 'avg_moves_to_win': '24.65', 'avg_moves_to_lose': '12.41'}
  - rl_model_25-08-17_19-55_iteration_550: {'wins': 368, 'losses': 4632, 'avg_moves_to_win': '25.27', 'avg_moves_to_lose': '12.83'}
  - rl_model_25-08-17_20-02_iteration_600: {'wins': 446, 'losses': 4554, 'avg_moves_to_win': '25.72', 'avg_moves_to_lose': '12.93'}
  - rl_model_25-08-17_20-24_iteration_750: {'wins': 581, 'losses': 4419, 'avg_moves_to_win': '26.17', 'avg_moves_to_lose': '13.29'}
  - rl_model_25-08-17_20-47_iteration_900: {'wins': 637, 'losses': 4363, 'avg_moves_to_win': '26.31', 'avg_moves_to_lose': '13.41'}

* I just realising that I'm doing nothing to normalise the global input state; that can't help at all. I might just do something simple like dividing mines_remaining by 10, and cells_revealed by 81. This will put them into the 0-1 range, instead of truly normalising them, but I suspect would still be worthwhile; and my assumption is already that they're secondary in importance to the rest of the board.

  - rl_model_25-08-17_21-10_iteration_1050: {'wins': 717, 'losses': 4283, 'avg_moves_to_win': '26.68', 'avg_moves_to_lose': '13.52'}

* Let's give it another go, but normalising the inputs.
* Actually, first let's do some profiling to see about making this whole thing a bit faster.
* OK so I think I've addressed a few of the efficiency concerns.

  - rl_model_25-08-17_22-36_iteration_0: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '9.50', 'avg_moves_to_lose': '3.47'}
  - rl_model_25-08-17_22-39_iteration_30: {'wins': 28, 'losses': 4972, 'avg_moves_to_win': '17.79', 'avg_moves_to_lose': '7.97'}

* Wow, this is too good to be true straight out the gate. I think I must have got lucky with starting parameters - but let's see how it evolves.

  - rl_model_25-08-17_22-42_iteration_60: {'wins': 77, 'losses': 4923, 'avg_moves_to_win': '21.17', 'avg_moves_to_lose': '9.50'}
  - rl_model_25-08-17_22-45_iteration_90: {'wins': 74, 'losses': 4926, 'avg_moves_to_win': '21.70', 'avg_moves_to_lose': '10.00'}
  - rl_model_25-08-17_22-49_iteration_120: {'wins': 88, 'losses': 4912, 'avg_moves_to_win': '23.12', 'avg_moves_to_lose': '10.70'}
  - rl_model_25-08-17_22-52_iteration_150: {'wins': 127, 'losses': 4873, 'avg_moves_to_win': '24.18', 'avg_moves_to_lose': '11.18'}
  - rl_model_25-08-17_22-55_iteration_180: {'wins': 162, 'losses': 4838, 'avg_moves_to_win': '25.64', 'avg_moves_to_lose': '12.06'}
  - rl_model_25-08-17_22-59_iteration_210: {'wins': 204, 'losses': 4796, 'avg_moves_to_win': '26.22', 'avg_moves_to_lose': '12.60'}
  - rl_model_25-08-17_23-03_iteration_240: {'wins': 257, 'losses': 4743, 'avg_moves_to_win': '26.52', 'avg_moves_to_lose': '13.76'}
  - rl_model_25-08-17_23-06_iteration_270: {'wins': 319, 'losses': 4681, 'avg_moves_to_win': '26.56', 'avg_moves_to_lose': '13.89'}

* One thing I'll note on this is that my GPU isn't being fully utilised, so we might be able to increase batch size for some performance gains.

  - rl_model_25-08-17_23-36_iteration_500: {'wins': 710, 'losses': 4290, 'avg_moves_to_win': '26.99', 'avg_moves_to_lose': '14.82'}
  - rl_model_25-08-17_23-49_iteration_600: {'wins': 840, 'losses': 4160, 'avg_moves_to_win': '27.09', 'avg_moves_to_lose': '15.10'}
  - rl_model_25-08-18_00-10_iteration_750: {'wins': 979, 'losses': 4021, 'avg_moves_to_win': '27.16', 'avg_moves_to_lose': '15.13'}
  - rl_model_25-08-18_00-30_iteration_900: {'wins': 1093, 'losses': 3907, 'avg_moves_to_win': '27.36', 'avg_moves_to_lose': '15.45'}
  - rl_model_25-08-18_08-28_iteration_4500: {'wins': 1332, 'losses': 3668, 'avg_moves_to_win': '27.17', 'avg_moves_to_lose': '15.02'}
  - rl_model_25-08-18_08-47_iteration_4650: {'wins': 1332, 'losses': 3668, 'avg_moves_to_win': '27.17', 'avg_moves_to_lose': '15.02'}
  - rl_model_25-08-18_09-07_iteration_4800: {'wins': 1332, 'losses': 3668, 'avg_moves_to_win': '27.17', 'avg_moves_to_lose': '15.02'}
  - rl_model_25-08-18_09-27_iteration_4950: {'wins': 1332, 'losses': 3668, 'avg_moves_to_win': '27.17', 'avg_moves_to_lose': '15.02'}
  - rl_model_25-08-18_09-47_iteration_5100: {'wins': 1332, 'losses': 3668, 'avg_moves_to_win': '27.17', 'avg_moves_to_lose': '15.02'}

## 18-08-2025

* This seems to have fully converged, and not to an amazing spot! I'm going to commit, try going back to ReLu, and add another smaller convolutional layer at the start (to hopefully extract more interesting features before it goes into a wider layer). It might also be time for me to start seriously conisdering changing my Max Q calculation to take into account both the predicted Max Q, and the observed one (from the one play-through of the game which we do to completion).
  - rl_model_25-08-18_09-56_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.23'}
  - rl_model_25-08-18_09-59_iteration_30: {'wins': 13, 'losses': 4987, 'avg_moves_to_win': '16.92', 'avg_moves_to_lose': '7.33'}
  - rl_model_25-08-18_10-02_iteration_60: {'wins': 59, 'losses': 4941, 'avg_moves_to_win': '19.66', 'avg_moves_to_lose': '9.14'}
  - rl_model_25-08-18_10-05_iteration_90: {'wins': 95, 'losses': 4905, 'avg_moves_to_win': '21.46', 'avg_moves_to_lose': '10.39'}
  - rl_model_25-08-18_10-08_iteration_120: {'wins': 185, 'losses': 4815, 'avg_moves_to_win': '24.57', 'avg_moves_to_lose': '12.33'}
  - rl_model_25-08-18_10-11_iteration_150: {'wins': 295, 'losses': 4705, 'avg_moves_to_win': '25.47', 'avg_moves_to_lose': '13.71'}
  - rl_model_25-08-18_10-15_iteration_180: {'wins': 410, 'losses': 4590, 'avg_moves_to_win': '26.28', 'avg_moves_to_lose': '14.77'}
  - rl_model_25-08-18_10-19_iteration_210: {'wins': 470, 'losses': 4530, 'avg_moves_to_win': '25.93', 'avg_moves_to_lose': '16.00'}
  - rl_model_25-08-18_10-23_iteration_240: {'wins': 656, 'losses': 4344, 'avg_moves_to_win': '26.75', 'avg_moves_to_lose': '16.47'}
  - rl_model_25-08-18_10-26_iteration_270: {'wins': 776, 'losses': 4224, 'avg_moves_to_win': '27.06', 'avg_moves_to_lose': '16.70'}
  - rl_model_25-08-18_10-30_iteration_300: {'wins': 768, 'losses': 4232, 'avg_moves_to_win': '26.91', 'avg_moves_to_lose': '17.28'}
  - rl_model_25-08-18_10-37_iteration_350: {'wins': 1145, 'losses': 3855, 'avg_moves_to_win': '27.43', 'avg_moves_to_lose': '17.56'}
  - rl_model_25-08-18_10-44_iteration_400: {'wins': 1500, 'losses': 3500, 'avg_moves_to_win': '28.34', 'avg_moves_to_lose': '17.57'}
  - rl_model_25-08-18_10-51_iteration_450: {'wins': 1570, 'losses': 3430, 'avg_moves_to_win': '28.25', 'avg_moves_to_lose': '17.58'}
  - rl_model_25-08-18_10-58_iteration_500: {'wins': 1717, 'losses': 3283, 'avg_moves_to_win': '27.90', 'avg_moves_to_lose': '17.71'}
  - rl_model_25-08-18_11-04_iteration_550: {'wins': 1944, 'losses': 3056, 'avg_moves_to_win': '28.24', 'avg_moves_to_lose': '17.47'}
  - rl_model_25-08-18_11-11_iteration_600: {'wins': 2042, 'losses': 2958, 'avg_moves_to_win': '28.08', 'avg_moves_to_lose': '17.25'}
  - rl_model_25-08-18_11-31_iteration_750: {'wins': 2123, 'losses': 2877, 'avg_moves_to_win': '28.39', 'avg_moves_to_lose': '17.91'}
  - rl_model_25-08-18_11-51_iteration_900: {'wins': 2291, 'losses': 2709, 'avg_moves_to_win': '28.28', 'avg_moves_to_lose': '18.07'}
  - rl_model_25-08-18_12-11_iteration_1050: {'wins': 2367, 'losses': 2633, 'avg_moves_to_win': '28.30', 'avg_moves_to_lose': '17.36'}

* OK this approach was incredibly promising! I realise though that I also need to add a channel to show if something is a valid square; so that the model's convolutional layers can distinguish between invalid locations and un-revealed squares (both currently all-zeros in the input).
* It also occurred to me, although I won't implement this yet, that minesweeper has horizontal, vertical and rotational symmetry (4x rotational symmetry because my board is square; otherwise 2x). In theory every training example I do could probably be flipped once - and then each of them rotated 4 times (flip horizontally and rotate twice should be equivalent to vertical, so we don't need to do that). That might help the training to learn faster; although I don't know if there might be other ways to better exploit this. I might try something along these later on; though I'll just start with the extra channel - because the previous attempt was going pretty well.

  - rl_model_25-08-18_12-39_iteration_0: {'wins': 0, 'losses': 5000, 'avg_moves_to_win': '0.00', 'avg_moves_to_lose': '3.83'}
  - rl_model_25-08-18_12-42_iteration_30: {'wins': 2, 'losses': 4998, 'avg_moves_to_win': '16.50', 'avg_moves_to_lose': '7.27'}

* Thinking about it actually; this symmetry might imply a little about the numbers of filters that I should go for. E.g. if I've identified 8 different permutations that essentially all need to behave the same; then I potentially need to have a multiple of 8 filters - as I'd expect to see this sort of duplication. OR alternatively, I should see if Tensorflow has a way of running the same filter in 8 different rotations - that would arguably be even better, because then I can have far far fewer parmeters.

  - rl_model_25-08-18_12-45_iteration_60: {'wins': 103, 'losses': 4897, 'avg_moves_to_win': '22.57', 'avg_moves_to_lose': '9.82'}
  - rl_model_25-08-18_12-48_iteration_90: {'wins': 120, 'losses': 4880, 'avg_moves_to_win': '23.90', 'avg_moves_to_lose': '10.63'}

* Claude reckons I have a decent idea there; but that Tensorflow may not natively support it. It seems like e2cnn in PyTorch might support it - so maybe I should consider switching to PyTorch (I understand it's more modern and more popular anyway). But I'll continue seeing how this model training plays out for a while.

  - rl_model_25-08-18_12-50_iteration_120: {'wins': 151, 'losses': 4849, 'avg_moves_to_win': '24.95', 'avg_moves_to_lose': '11.58'}

* At this point it seems to be learning comparably fast, or maybe slightly slower than the previous one. I wonder if that's just because we have more parameters - and probably the new information I've given it is once-again most useful in longer, more complex games after it has learned basic strategy.

  - rl_model_25-08-18_12-53_iteration_150: {'wins': 201, 'losses': 4799, 'avg_moves_to_win': '25.84', 'avg_moves_to_lose': '12.10'}
  - rl_model_25-08-18_12-56_iteration_180: {'wins': 246, 'losses': 4754, 'avg_moves_to_win': '25.30', 'avg_moves_to_lose': '12.99'}
  - rl_model_25-08-18_13-00_iteration_210: {'wins': 474, 'losses': 4526, 'avg_moves_to_win': '25.95', 'avg_moves_to_lose': '13.63'}
  - rl_model_25-08-18_13-03_iteration_240: {'wins': 612, 'losses': 4388, 'avg_moves_to_win': '26.33', 'avg_moves_to_lose': '13.79'}
  - rl_model_25-08-18_13-06_iteration_270: {'wins': 871, 'losses': 4129, 'avg_moves_to_win': '26.99', 'avg_moves_to_lose': '14.13'}
  - rl_model_25-08-18_13-10_iteration_300: {'wins': 1007, 'losses': 3993, 'avg_moves_to_win': '27.21', 'avg_moves_to_lose': '14.13'}
  - rl_model_25-08-18_13-15_iteration_350: {'wins': 1142, 'losses': 3858, 'avg_moves_to_win': '27.19', 'avg_moves_to_lose': '15.21'}
  - rl_model_25-08-18_13-20_iteration_400: {'wins': 1491, 'losses': 3509, 'avg_moves_to_win': '27.53', 'avg_moves_to_lose': '15.38'}
  - rl_model_25-08-18_13-26_iteration_450: {'wins': 1614, 'losses': 3386, 'avg_moves_to_win': '27.57', 'avg_moves_to_lose': '14.51'}
  - rl_model_25-08-18_13-31_iteration_500: {'wins': 1794, 'losses': 3206, 'avg_moves_to_win': '27.81', 'avg_moves_to_lose': '14.80'}
  - rl_model_25-08-18_13-37_iteration_550: {'wins': 1962, 'losses': 3038, 'avg_moves_to_win': '27.50', 'avg_moves_to_lose': '14.96'}
  - rl_model_25-08-18_13-42_iteration_600: {'wins': 1998, 'losses': 3002, 'avg_moves_to_win': '27.44', 'avg_moves_to_lose': '14.76'}
  - rl_model_25-08-18_13-58_iteration_750: {'wins': 2310, 'losses': 2690, 'avg_moves_to_win': '27.48', 'avg_moves_to_lose': '14.82'}
  - rl_model_25-08-18_14-14_iteration_900: {'wins': 2320, 'losses': 2680, 'avg_moves_to_win': '27.54', 'avg_moves_to_lose': '13.86'}
  - rl_model_25-08-18_14-31_iteration_1050: {'wins': 2463, 'losses': 2537, 'avg_moves_to_win': '27.21', 'avg_moves_to_lose': '13.92'}
  - rl_model_25-08-18_14-47_iteration_1200: {'wins': 2495, 'losses': 2505, 'avg_moves_to_win': '27.15', 'avg_moves_to_lose': '13.62'}
  - rl_model_25-08-18_15-03_iteration_1350: {'wins': 2487, 'losses': 2513, 'avg_moves_to_win': '27.16', 'avg_moves_to_lose': '13.62'}
  - rl_model_25-08-18_15-19_iteration_1500: {'wins': 2520, 'losses': 2480, 'avg_moves_to_win': '27.06', 'avg_moves_to_lose': '13.32'}
  - rl_model_25-08-18_15-35_iteration_1650: {'wins': 2551, 'losses': 2449, 'avg_moves_to_win': '27.14', 'avg_moves_to_lose': '13.33'}
  - rl_model_25-08-18_15-51_iteration_1800: {'wins': 2531, 'losses': 2469, 'avg_moves_to_win': '26.99', 'avg_moves_to_lose': '13.06'}
  - rl_model_25-08-18_16-07_iteration_1950: {'wins': 2538, 'losses': 2462, 'avg_moves_to_win': '26.98', 'avg_moves_to_lose': '13.32'}
  - rl_model_25-08-18_16-23_iteration_2100: {'wins': 2563, 'losses': 2437, 'avg_moves_to_win': '26.95', 'avg_moves_to_lose': '13.32'}

* OK this is very promising, but I think I should expand the number kernels a little at this point.
* I'll also change the reward function to reward wins more than 10x flags. If it can learn to win without flagging anything, that's better than having to flag things.

  - rl_model_25-08-18_16-31_iteration_0: {'wins': 1, 'losses': 4999, 'avg_moves_to_win': '7.00', 'avg_moves_to_lose': '4.51'}
  - rl_model_25-08-18_16-34_iteration_30: {'wins': 66, 'losses': 4934, 'avg_moves_to_win': '24.62', 'avg_moves_to_lose': '9.99'}
  - rl_model_25-08-18_16-37_iteration_60: {'wins': 251, 'losses': 4749, 'avg_moves_to_win': '26.04', 'avg_moves_to_lose': '11.68'}
  - rl_model_25-08-18_16-40_iteration_90: {'wins': 610, 'losses': 4390, 'avg_moves_to_win': '27.29', 'avg_moves_to_lose': '14.00'}
  - rl_model_25-08-18_16-44_iteration_120: {'wins': 1044, 'losses': 3956, 'avg_moves_to_win': '27.18', 'avg_moves_to_lose': '15.54'}

* Wow, this is learning so much faster. I can't help but wonder if this is more about the change to the reward than the model size or anything though: previously a win was worth the same as 10 flags - and you could imagine local minima where it's easier to attempt to get more flags than to actually win, and so the game might be more likely to lose. Now it's much more incentivised to actually win the game.
* Honestly I wonder if I should have only used win/loss rewards, and nothing intermediate. And maybe even make flags toggleable again, instead of an instant death for flagging a non-mine (more standard rules). Perfect gameplay wouldn't require any flags anyway, and it could learn that!

  - rl_model_25-08-18_16-47_iteration_150: {'wins': 1666, 'losses': 3334, 'avg_moves_to_win': '27.99', 'avg_moves_to_lose': '15.32'}
  - rl_model_25-08-18_16-51_iteration_180: {'wins': 1911, 'losses': 3089, 'avg_moves_to_win': '28.07', 'avg_moves_to_lose': '15.25'}
  - rl_model_25-08-18_16-54_iteration_210: {'wins': 2212, 'losses': 2788, 'avg_moves_to_win': '27.94', 'avg_moves_to_lose': '15.58'}
  - rl_model_25-08-18_16-58_iteration_240: {'wins': 2389, 'losses': 2611, 'avg_moves_to_win': '27.88', 'avg_moves_to_lose': '15.37'}

* This is looking really good; but I'm going to stop it now, and then try and set all rewards aside from winning and loss to zero.
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
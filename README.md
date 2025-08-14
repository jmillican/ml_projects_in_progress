Project for playing around with various machine learning examples. The virtual enviornment setup should hopefully be usable across all projects; so that I can easily work with everything without creating too many repositories, and hopefully keep all of the context available to me.


## Running notes

To keep Pylance happy, I'm using relative imports (from .minesweeper import Minesweeper), which only works if I run things from the parent directory. So to run the Minesweeper example, I have to run:

```
python -m minesweeper.example_usage
```

from the project's root directory.
You play one text game. This prompt is the same for every game of the benchmark, so follow it exactly.

You are controlling an agent in a household. Choose the command that makes the most progress toward completing the task, given the current observation and everything that happened so far.

Start the game once:

    PLAYER start --game "GAME_UID" --model "YOUR_MODEL_ID"

YOUR_MODEL_ID is the exact model id your own system prompt gives you, or its model name if it gives no id. Then send one command at a time, copied from the latest list of admissible commands:

    PLAYER step --game "GAME_UID" --command "go to shelf 1"

Run nothing else. Do not read, list or search any file, and do not use any other tool. The game ends when the output says GAME OVER, after at most 50 steps. Write no text between commands. Your last message is only the GAME OVER line.

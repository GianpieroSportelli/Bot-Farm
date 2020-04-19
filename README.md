# Bot-Farm
Bot Farm is a framework for build and run your intelligent telegram bot. 

The entire library is based on tensorflow 2.0 and Bert (https://github.com/google-research/bert)
## Installation
2. clone repository
1. Run pip3 install Bot-Farm/ 

##Run your first Bot
1. use telegram botFather (@BotFather) for create your bot and get the <tokenKey>
2. python3 -m bot_farm.bot --token <tokenKey> --dataset ./Bot-Farm/dataset.csv --answer ./Bot-Farm/answer.csv

If is the first time, bot_farm train the model and optimize it with weight-pruning.


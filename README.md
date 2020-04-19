# Bot-Farm
Bot Farm is a framework for build and run your intelligent telegram bot. 

The entire library is based on tensorflow 2.0 and Bert (https://github.com/google-research/bert)
## Installation
pip3 install git+https://github.com/GianpieroSportelli/Bot-Farm.git 

## Run your first Bot
1. clone repository (git clone https://github.com/GianpieroSportelli/Bot-Farm.git)
2. use telegram botFather (@BotFather) for create your bot and get the `your_token_key`
3. python3 -m bot_farm.bot --token `your_token_key` --dataset ./Bot-Farm/dataset.csv --answer ./Bot-Farm/answer.csv

If is the first time, bot_farm train the model and optimize it with weight-pruning.


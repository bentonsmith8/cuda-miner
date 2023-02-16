import requests
from collections import Counter
MINERS = 'https://maiti.info/anindya/services/mining/miners.php'

def get_data():
    contents = requests.get(MINERS).content
    contents = contents.decode('utf-8')
    contents = contents.split('\n')[:-2]
    return contents

def find_block_num(data):
    miners = [m.split(',')[2] for m in data]
    return Counter(miners)

def main():

    data = get_data()
    print('Total blocks', len(data))
    coins = find_block_num(data)
    #sort the miners by the number of coins they have
    m_coins = sorted(coins.items(), key=lambda x: x[1], reverse=True)
    for m, coins in m_coins:
        print(m, coins)

if __name__ == '__main__':
    main()
import requests
from multiprocessing import Pool
"""
This script queries the scryfall api for the urls of magic card arts, and then
pulls the card arts from those urls and saves them to a folder
"""


def get_card(page_num):
    count = 0
    payload = {'page': str(page_num)}
    data = requests.get("https://api.scryfall.com/cards", params=payload)
    json = data.json()['data']
    for j in range(len(json)):
        image = requests.get(json[j]['image_uris']['png']).content
        with open("D:/mtg_card_data/card_" + str(count) + "_page_" + str(page_num) + ".png", 'wb') as file:
            file.write(image)
        count += 1


if __name__ == "__main__":
    # From Scryfall website
    number_of_pages = 1460
    count = 0
    pool = Pool(processes=6)
    pool.map(get_card, range(4, 1461))

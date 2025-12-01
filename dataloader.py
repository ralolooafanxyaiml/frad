import os
import random

OUTPUT_FILE = "input_code.txt"

BOOKS = [
    {'title': 'Friedrich Nietzsche - Thus Spoke Zarathustra', 'url': 'https://www.gutenberg.org/cache/epub/1998/pg1998.txt'},
    {'title': 'Friedrich Nietzsche - Beyond Good and Evil', 'url': 'https://www.gutenberg.org/cache/epub/4363/pg4363.txt'},
    {'title': 'Fyodor Dostoyevsky - Notes from the Underground', 'url': 'https://www.gutenberg.org/cache/epub/600/pg600.txt'},
    {'title': 'Fyodor Dostoyevsky - Crime and Punishment', 'url': 'https://www.gutenberg.org/cache/epub/2554/pg2554.txt'},
    {'title': 'Fyodor Dostoyevsky - The Brothers Karamazov', 'url': 'https://www.gutenberg.org/files/28054/28054-0.txt'},
    {'title': 'Franz Kafka - The Metamorphosis', 'url': 'https://www.gutenberg.org/cache/epub/5200/pg5200.txt'},
    {'title': 'Franz Kafka - The Trial', 'url': 'https://www.gutenberg.org/cache/epub/7849/pg7849.txt'},
    {'title': 'Arthur Schopenhauer - Essays', 'url': 'https://www.gutenberg.org/cache/epub/10732/pg10732.txt'},
    {'title': 'Marcus Aurelius - Meditations', 'url': 'https://www.gutenberg.org/cache/epub/2680/pg2680.txt'},
    {'title': 'Niccolo Machiavelli - The Prince', 'url': 'https://www.gutenberg.org/cache/epub/1232/pg1232.txt'},
    {'title': 'Thomas Hobbes - Leviathan', 'url': 'https://www.gutenberg.org/cache/epub/3207/pg3207.txt'},
    {'title': 'Sun Tzu - The Art of War', 'url': 'https://www.gutenberg.org/cache/epub/132/pg132.txt'},
    {'title': 'Dante Alighieri - The Divine Comedy', 'url': 'https://www.gutenberg.org/cache/epub/8800/pg8800.txt'},
    {'title': 'John Milton - Paradise Lost', 'url': 'https://www.gutenberg.org/cache/epub/20/pg20.txt'},
    {'title': 'Goethe - Faust', 'url': 'https://www.gutenberg.org/cache/epub/14591/pg14591.txt'},
    {'title': 'HP Lovecraft - The Dunwich Horror', 'url': 'https://www.gutenberg.org/cache/epub/50133/pg50133.txt'},
    {'title': 'HP Lovecraft - The Call of Cthulhu', 'url': 'https://www.gutenberg.org/files/15210/15210-0.txt'},
    {'title': 'Edgar Allan Poe - The Works Vol 1', 'url': 'https://www.gutenberg.org/cache/epub/2147/pg2147.txt'},
    {'title': 'Edgar Allan Poe - The Works Vol 2', 'url': 'https://www.gutenberg.org/cache/epub/2148/pg2148.txt'},
    {'title': 'Mary Shelley - Frankenstein', 'url': 'https://www.gutenberg.org/cache/epub/84/pg84.txt'},
    {'title': 'Mary Shelley - The Last Man', 'url': 'https://www.gutenberg.org/cache/epub/18247/pg18247.txt'},
    {'title': 'Bram Stoker - Dracula', 'url': 'https://www.gutenberg.org/cache/epub/345/pg345.txt'},
    {'title': 'Robert Louis Stevenson - Dr Jekyll and Mr Hyde', 'url': 'https://www.gutenberg.org/cache/epub/43/pg43.txt'},
    {'title': 'Oscar Wilde - The Picture of Dorian Gray', 'url': 'https://www.gutenberg.org/cache/epub/174/pg174.txt'},
    {'title': 'Robert W Chambers - The King in Yellow', 'url': 'https://www.gutenberg.org/cache/epub/8492/pg8492.txt'},
    {'title': 'Joseph Conrad - Heart of Darkness', 'url': 'https://www.gutenberg.org/cache/epub/219/pg219.txt'},
    {'title': 'HG Wells - The Island of Doctor Moreau', 'url': 'https://www.gutenberg.org/cache/epub/159/pg159.txt'},
    {'title': 'Henry James - The Turn of the Screw', 'url': 'https://www.gutenberg.org/cache/epub/209/pg209.txt'},
    {'title': 'Arthur Machen - The Great God Pan', 'url': 'https://www.gutenberg.org/cache/epub/389/pg389.txt'},
    {'title': 'Ambrose Bierce - The Devils Dictionary', 'url': 'https://www.gutenberg.org/cache/epub/972/pg972.txt'}
        ]

def clean_gutenberg_text(text)
    start_market = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_market = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_market)

    if start_idx != -1 and end_idx != -1:
        return text[start_idx + len(start_market):end_idx].strip()

    return text

def download_and_merge():
    print(f"Downloading Data... to: {OUTPUT_FILE}\n")

    total_chars = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for book in BOOKS:
            print("Downloading...")
            try:
                response = requests.get(book["url"])
                response.raise_for_status()
                 
                raw_text = response.text
                clean_text = clean_gutenberg_text(raw_text)

                outfile.write(f"\n\n--- {book['title']} ---\n\n")
                outfile.write(clean_text)

                total_chars += len(clean_text)

            except Exception as e:
                print(f"Error {e}!")

    print("Finished.")

if __name__ == "__main__":
    download_and_merge()
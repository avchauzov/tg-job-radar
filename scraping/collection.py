import asyncio  # Add this import
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient

from scraping import KEYWORDS_LENGTH, MAX_NGRAM_LENGTH, MIN_NGRAM_LENGTH, SESSION_NAME, TG_API_HASH, TG_API_ID
from scraping.functions import clean_text, iterate_lines, ngram_search


channel_username = 'cryptojobslist'

client = TelegramClient(SESSION_NAME, TG_API_ID, TG_API_HASH)


async def main():
	await client.start()
	print("You're connected!")
	
	async for message in client.iter_messages(
			entity=channel_username,
			offset_date=datetime.now(tz=timezone.utc) - timedelta(hours=1024),
			reverse=True
			):
		job_description = message.text
		
		cleaned_text = clean_text(job_description)
		lines = iterate_lines(cleaned_text)
		
		for line in lines:
			for length in range(MIN_NGRAM_LENGTH, MAX_NGRAM_LENGTH + 1):
				keywords_subset = [key for key, value in KEYWORDS_LENGTH.items() if value == length]
				if ngram_search(line, keywords_subset, length):
					print(message.text)
					print()
					
					break  # remove URLs


async def run():
	async with client:
		await main()


if __name__ == '__main__':
	asyncio.run(run())

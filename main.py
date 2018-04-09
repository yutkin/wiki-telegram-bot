import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler

import logging

from emoji import emojize
from collections import deque
import shelve

from utils import *

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

GREETING_MSG = 'Привет, {name}! Я {botname}!:waving_hand:\nО чём хочешь узнать?'
NOT_FOUND_MSG = 'По запросу ничего не найдено! :frowning_face:'
ERROR_MSG = 'Что-то пошло не так... :thinking_face:'
MULTIPLE_MEANINGS_MSG = 'Много различных значений, уточните запрос. :winking_face:'
HISTORY_HAS_BEEN_CLEANED_MSG = 'История запросов очищена.'
HISTORY_IS_EMPTY_MSG = 'История запросов чиста.'
HISTORY_LIST_MSG = 'История запросов:'

TOKEN = ''
HISTORY_MAX_LEN = 10
DB_NAME = 'history.db'

wiki = WikipediaDataSet('articles_meta.csv', 'dataset.npy', 'cc.ru.300.bin')
logger.info('Wikipedia dataset has been constructed')
wiki.build_LSH_index()
logger.info('LSH Index has been built')


def start(bot, update):
    msg = GREETING_MSG.format(name=update.message.from_user.first_name, botname=bot.username)
    update.message.reply_text(emojize(msg))


def history(bot, update):
    with shelve.open(DB_NAME) as db:
        hist = db.get(str(update.message.chat_id))
    if hist:
        rows = []
        for i, (title, url) in enumerate(hist):
            rows.append('{}. {} [{}]'.format(i+1, title, url))
        update.message.reply_text('{}\n{}'.format(HISTORY_LIST_MSG, '\n'.join(rows)),
                                  disable_web_page_preview=True)
    else:
        update.message.reply_text(HISTORY_IS_EMPTY_MSG)


def add_to_history(chat_id, title, url):
    key = str(chat_id)
    with shelve.open(DB_NAME) as db:
        hist = db.get(key)

    hist = hist or deque(maxlen=HISTORY_MAX_LEN)
    hist.append((title, url))

    with shelve.open(DB_NAME) as db:
        db[key] = hist


def clear_history(bot, update):
    with shelve.open(DB_NAME) as db:
        db[str(update.message.chat_id)] = deque(maxlen=HISTORY_MAX_LEN)
    update.message.reply_text(emojize(HISTORY_HAS_BEEN_CLEANED_MSG))


def button_pressed(bot, update):
    query = update.callback_query
    query.answer()
    bot.send_chat_action(chat_id=query.message.chat_id,
                         action=telegram.ChatAction.TYPING)

    url = WIKI_ARTICLE_URL.format(pageid=query.data)
    title = get_title_by_url(url)
    summary = get_wiki_article_summary(query.data)

    summary = summary if summary else ''
    title = title if title else '?'

    recommended_articles = wiki.find_k_nearest_neighbors(title)
    keyboard = []
    for article in recommended_articles:
        keyboard.append([
            InlineKeyboardButton(article.title, callback_data=article.pageid)
        ])
    reply_markup = InlineKeyboardMarkup(keyboard)

    add_to_history(str(query.message.chat_id), title, url)

    bot.edit_message_text('{}\n{}'.format(url, summary),
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id,
                          reply_markup=reply_markup,
                          disable_web_page_preview=True)


def query_article(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    query = update.message.text.strip().lower()
    try:
        found = search_wiki_article(query)
        if found:
            title, pageid = found
            url = WIKI_ARTICLE_URL.format(pageid=pageid)
            summary = get_wiki_article_summary(pageid)
            if summary:
                recommended_articles = wiki.find_k_nearest_neighbors(title)
                keyboard = []
                for article in recommended_articles:
                    keyboard.append([
                        InlineKeyboardButton(article.title, callback_data=article.pageid)
                    ])
                reply_markup = InlineKeyboardMarkup(keyboard)

                add_to_history(str(update.message.chat_id), title, url)

                update.message.reply_text('{}\n{}'.format(url, summary),
                                          reply_markup=reply_markup,
                                          disable_web_page_preview=True)
            else:
                update.message.reply_text(emojize(MULTIPLE_MEANINGS_MSG))
        else:
            update.message.reply_text(emojize(NOT_FOUND_MSG))
    except Exception as AnyException:
        update.message.reply_text(emojize(ERROR_MSG))
        logger.error(AnyException)


def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    updater = Updater(TOKEN, request_kwargs={'read_timeout': 30,
                                             'connect_timeout': 30})

    dp = updater.dispatcher

    dp.add_handler(CallbackQueryHandler(button_pressed))
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("hist", history))
    dp.add_handler(CommandHandler("clear_hist", clear_history))
    dp.add_handler(MessageHandler(Filters.text, query_article))

    dp.add_error_handler(error)

    updater.start_polling(clean=True)

    logger.info('Bot Started')

    updater.idle()


if __name__ == '__main__':
    main()

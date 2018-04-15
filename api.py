# -*- coding: utf-8 -*-

import re
from datetime import datetime, timedelta
from time import mktime
from urllib import quote_plus

from flask import Flask, jsonify, request
from flask_cache import Cache
from flask_cors import CORS
from nltk.tokenize import TreebankWordTokenizer
from pymorphy2 import MorphAnalyzer
from selenium import webdriver

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
CORS(app)

DATE_FORMAT = "%Y-%m-%d"  # e.g. 2018-02-24
DEFAULT_DATE_RANGE = 30  # days
MAX_REQUESTS = 30
MIN_RESOLUTION = 1  # day
RE_RUS_LETTERS = re.compile(u"[^абвгдеёжзийклмнопрстуфхцчшщъыьэюя ]")
TWITTER_SEARCH_URL = "https://twitter.com/search?q={query}%20until%3A{until}%20since%3A{since}&l=ru&src=typd"
TWITTER_DATE_FORMAT = "%Y-%m-%d"

with open("data/stopwords-ru.txt") as f:
    STOPWORDS = map(lambda w: w.decode("utf-8"), f.read().splitlines())

morph = MorphAnalyzer()
tokenizer = TreebankWordTokenizer()

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--incognito")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(chrome_options=options)

import numpy as np

from keras.models import load_model
from sklearn.cluster import KMeans
from data_helpers import build_fv, load_data
from lingua import Text

clf = load_model("data/clf_7167")
x, y, vocabulary, vocabulary_inv_list = load_data()


def mark(text):
    sentence = [s.normal_form for s in Text(text).tokenize()]
    fv = build_fv(sentence, 48, vocabulary)
    m = clf.predict(np.matrix(fv))[0][0]
    return -1 if m > 0.66 else 0 if m > 0.33 else 1


mark("тест")


def _error(msg):
    """...

    Args:
        msg: The error message.

    Returns:
        Jsonified error.

    """
    return jsonify(status='error', msg=msg)


def _extract_keywords(tweets):
    """...

    Args:
        tweets: List of tweets.

    """
    neg, neu, pos = {}, {}, {}

    for tweet in tweets:
        for word in tokenizer.tokenize(RE_RUS_LETTERS.sub(u"", tweet["text"].lower())):
            keyword = morph.parse(word)[0].normal_form
            if keyword in STOPWORDS:
                continue

            if tweet["verdict"] == "negative":
                neg[keyword] = neg.get(keyword, 0) + 1
            elif tweet["verdict"] == "neutral":
                neu[keyword] = neu.get(keyword, 0) + 1
            elif tweet["verdict"] == "positive":
                pos[keyword] = pos.get(keyword, 0) + 1

    for cat in [neg, neu, pos]:
        for keyword, count in cat.items():
            if count < 2:
                del cat[keyword]

    return {
        "negative": neg,
        "neutral": neu,
        "positive": pos
    }


def _extract_topics(tweets):
    """...

    Maybe by tags?

    Args:
        tweets: List of tweets.

    """
    topics = []

    all_words = set()
    words_by_tweets = []
    for tweet in tweets:
        words_by_tweets.append([s.normal_form for s in Text(tweet["text"].encode("utf-8")).tokenize() if s.part_of_speech == "NOUN"])
        all_words.update(words_by_tweets[-1])

    all_words = list(all_words)

    features = []
    for words in words_by_tweets:
        features.append([1 if w in words else 0 for w in all_words])

    features = np.array(features)
    km = KMeans(n_clusters=20)
    km.fit(features)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    indexes = [i for i in range(20) if len([cluster for cluster in km.labels_ if cluster == i]) > 1]
    for i in indexes[:5]:
        topics.append({"keywords": []})
        for ind in order_centroids[i, :3]:
            topics[-1]["keywords"].append(all_words[ind])

        topics[-1]["ids"] = [tweets[j]["id"] for j, cluster in enumerate(km.labels_) if cluster == i]

    return topics


def _get_count(root_el, name):
    """...

    Args:
        root_el: Tweet's root element.
        name: Count name.

    """
    count = root_el\
        .find_element_by_css_selector(".js-action{} > span".format(name)).get_attribute("data-tweet-stat-count")

    if not count:
        count = root_el\
            .find_element_by_css_selector(".js-action{} span.ProfileTweet-actionCountForPresentation".format(name)).text

    return int(count or 0)


def _get_dates(from_date, to_date):
    """...

    Args:
        from_date (optional): The oldest UTC timestamp from which the Tweets will be provided.
        to_date (optional): The latest, most recent UTC timestamp to which the Tweets will be provided.

    """
    if not from_date and not to_date:
        to_date = datetime.now()
        from_date = datetime.now() - timedelta(days=DEFAULT_DATE_RANGE)
    elif not from_date:
        to_date = datetime.strptime(to_date, DATE_FORMAT)
        from_date = to_date - timedelta(days=DEFAULT_DATE_RANGE)
    elif not to_date:
        from_date = datetime.strptime(from_date, DATE_FORMAT)
        to_date = from_date + timedelta(days=DEFAULT_DATE_RANGE)
    else:
        from_date = datetime.strptime(from_date, DATE_FORMAT)
        to_date = datetime.strptime(to_date, DATE_FORMAT)

    return from_date, to_date


def _get_verdict(text):
    """...

    Args:
        text: The text to be evaluated.

    Returns:
        (verdict, score) ...

    """
    sentence = [s.normal_form for s in Text(text.encode("utf-8")).tokenize()]
    fv = build_fv(sentence, 48, vocabulary)
    score = 2 * float(clf.predict(np.matrix(fv))[0][0]) - 1
    return "positive" if score > 0.33 else "neutral" if score > -0.33 else "negative", score


@cache.memoize(3600)
def _get_tweets(query, from_date, to_date):
    """...

    Args:
        query: ...
        from_date: ...
        to_date: ...

    Returns:
        List of tweet dicts?

    """
    current_date = to_date
    resolution = timedelta(days=max(1, ((to_date - from_date) / MAX_REQUESTS).days))

    tweets = []
    while current_date > from_date:
        app.logger.debug("Getting tweets from {} to {}".format(
            (current_date - resolution).strftime(TWITTER_DATE_FORMAT),
            current_date.strftime(TWITTER_DATE_FORMAT)
        ))

        driver.get(TWITTER_SEARCH_URL.format(query=quote_plus(query.encode("utf-8")),
                                             since=(current_date-resolution).strftime(TWITTER_DATE_FORMAT),
                                             until=current_date.strftime(TWITTER_DATE_FORMAT)))

        # parse web page
        for root_el in driver.find_elements_by_css_selector(".tweet"):
            try:
                tweets.append({
                    "id": int(root_el.get_attribute("data-tweet-id")),
                    "ts": int(root_el.find_element_by_css_selector("._timestamp").get_attribute("data-time")),
                    "user": root_el.find_element_by_css_selector(".username").text,
                    "reply_count": _get_count(root_el, "Reply"),
                    "retweet_count":  _get_count(root_el, "Retweet"),
                    "favorite_count":  _get_count(root_el, "Favorite"),
                    "text": root_el.find_element_by_css_selector(".tweet-text").text,
                    "since": int(mktime((current_date-resolution).timetuple())),
                    "until": int(mktime(current_date.timetuple()))
                })

                verdict, score = _get_verdict(tweets[-1]["text"])
                tweets[-1]["verdict"] = verdict
                tweets[-1]["score"] = score
            except Exception as e:
                app.logger.warning("Unable to add tweet. It says: {}".format(e))

        current_date -= resolution

    return tweets


@app.route('/tweets', methods=['GET'])
def get_tweets():
    """Gets tweets for the specified query and time period.

    Note:
        If a time period is not specified the time parameters will default to the last 30 days.

    Params:
        q (required): A UTF-8, URL-encoded search query of 500 characters maximum, including operators.
        count (optional): The number of tweets to return per page, up to a maximum of 100. Defaults to 15.
        fromDate (optional): The oldest UTC timestamp from which the Tweets will be provided.
        toDate (optional): The latest, most recent UTC timestamp to which the Tweets will be provided.

    Returns:
        JSON with list of tweets, that satisfy specified params.

    """
    query = request.args.get('q').strip()
    from_date, to_date = _get_dates(request.args.get('fromDate'), request.args.get('toDate'))

    # validate
    if not query:
        return _error("query can't be empty")
    elif from_date > to_date:
        return _error("fromDate can't be after toDate")
    elif (to_date - from_date).days < 1:
        return _error("min resolution is 1 day")

    # search tweets
    tweets = _get_tweets(query, from_date, to_date)

    # send results
    return jsonify(tweets=tweets,
                   keywords=_extract_keywords(tweets),
                   topics=_extract_topics(tweets),
                   from_date=int(mktime(from_date.timetuple())),
                   to_date=int(mktime(to_date.timetuple())))


if __name__ == '__main__':
    app.run(debug=True)

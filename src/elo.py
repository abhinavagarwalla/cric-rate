from datetime import datetime
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

__version__  = '0.1.dev'
__all__ = ['Elo', 'Rating', 'CountedRating', 'TimedRating', 'rate', 'adjust',
           'expect', 'rate_1vs1', 'adjust_1vs1', 'quality_1vs1', 'setup',
           'global_env', 'WIN', 'DRAW', 'LOSS', 'K_FACTOR', 'RATING_CLASS',
           'INITIAL', 'BETA']


#: The actual score for win.
WIN = 1.
#: The actual score for draw.
DRAW = 0.5
#: The actual score for loss.
LOSS = 0.

#: Default K-factor.
K_FACTOR = 10
#: Default rating class.
RATING_CLASS = float
#: Default initial rating.
INITIAL = 1200
#: Default Beta value.
BETA = 200

MARGIN_RUN = 0.2
MARGIN_RUN_NORM = 50.
MARGIN_WKTS = 0.2
visual = True

class Rating(object):

    try:
        __metaclass__ = __import__('abc').ABCMeta
    except ImportError:
        # for Python 2.5
        pass

    value = None

    def __init__(self, value=None):
        if value is None:
            value = global_env().initial
        self.value = value

    def rated(self, value):
        """Creates a :class:`Rating` object for the recalculated rating.

        :param value: the recalculated rating value.
        """
        return type(self)(value)

    def __int__(self):
        """Type-casting to ``int``."""
        return int(self.value)

    def __long__(self):
        """Type-casting to ``long``."""
        return long(self.value)

    def __float__(self):
        """Type-casting to ``float``."""
        return float(self.value)

    def __nonzero__(self):
        """Type-casting to ``bool``."""
        return bool(int(self))

    def __eq__(self, other):
        return float(self) == float(other)

    def __lt__(self, other):
        """Is Rating < number.

        :param other: the operand
        :type other: number
        """
        return self.value < other

    def __le__(self, other):
        """Is Rating <= number.

        :param other: the operand
        :type other: number
        """
        return self.value <= other

    def __gt__(self, other):
        """Is Rating > number.

        :param other: the operand
        :type other: number
        """
        return self.value > other

    def __ge__(self, other):
        """Is Rating >= number.

        :param other: the operand
        :type other: number
        """
        return self.value >= other

    def __iadd__(self, other):
        """Rating += number.

        :param other: the operand
        :type other: number
        """
        self.value += other
        return self

    def __isub__(self, other):
        """Rating -= number.

        :param other: the operand
        :type other: number
        """
        self.value -= other
        return self

    def __repr__(self):
        c = type(self)
        ext_params = inspect.getargspec(c.__init__)[0][2:]
        kwargs = ', '.join('%s=%r' % (param, getattr(self, param))
                           for param in ext_params)
        if kwargs:
            kwargs = ', ' + kwargs
        args = ('.'.join([c.__module__, c.__name__]), self.value, kwargs)
        return '%s(%.3f%s)' % args


try:
    Rating.register(float)
except AttributeError:
    pass


class CountedRating(Rating):
    """Increases count each rating recalculation."""

    times = None

    def __init__(self, value=None, times=0):
        self.times = times
        super(CountedRating, self).__init__(value)

    def rated(self, value):
        rated = super(CountedRating, self).rated(value)
        rated.times = self.times + 1
        return rated


class TimedRating(Rating):
    """Writes the final rated time."""

    rated_at = None

    def __init__(self, value=None, rated_at=None):
        self.rated_at = rated_at
        super(TimedRating, self).__init__(value)

    def rated(self, value):
        rated = super(TimedRating, self).rated(value)
        rated.rated_at = datetime.utcnow()
        return rated


class Elo(object):

    def __init__(self, k_factor=K_FACTOR, rating_class=RATING_CLASS,
                 initial=INITIAL, beta=BETA):
        self.k_factor = k_factor
        self.rating_class = rating_class
        self.initial = initial
        self.beta = beta

    def expect(self, rating, other_rating):
        """The "E" function in Elo. It calculates the expected score of the
        first rating by the second rating.
        """
        # http://www.chess-mind.com/en/elo-system
        diff = float(other_rating) - float(rating)
        f_factor = 2 * self.beta  # rating disparity
        return 1. / (1 + 10 ** (diff / f_factor))

    def adjust(self, rating, series):
        """Calculates the adjustment value."""
        return sum(score - self.expect(rating, other_rating)
                   for score, other_rating in series)

    def rate(self, rating, series):
        """Calculates new ratings by the game result series."""
        rating = self.ensure_rating(rating)
        k = self.k_factor(rating) if callable(self.k_factor) else self.k_factor
        new_rating = float(rating) + k * self.adjust(rating, series)
        if hasattr(rating, 'rated'):
            new_rating = rating.rated(new_rating)
        return new_rating

    def adjust_1vs1(self, rating1, rating2, drawn=False):
        return self.adjust(rating1, [(DRAW if drawn else WIN, rating2)])

    def rate_1vs1(self, rating1, rating2, drawn=False):
        scores = (DRAW, DRAW) if drawn else (WIN, LOSS)
        return (self.rate(rating1, [(scores[0], rating2)]),
                self.rate(rating2, [(scores[1], rating1)]))

    def quality_1vs1(self, rating1, rating2):
        return 2 * (0.5 - abs(0.5 - self.expect(rating1, rating2)))

    def create_rating(self, value=None, *args, **kwargs):
        if value is None:
            value = self.initial
        return self.rating_class(value, *args, **kwargs)

    def ensure_rating(self, rating):
        if isinstance(rating, self.rating_class):
            return rating
        return self.rating_class(rating)

    def make_as_global(self):
        """Registers the environment as the global environment.

        >>> env = Elo(initial=2000)
        >>> Rating()
        elo.Rating(1200.000)
        >>> env.make_as_global()  #doctest: +ELLIPSIS
        elo.Elo(..., initial=2000.000, ...)
        >>> Rating()
        elo.Rating(2000.000)

        But if you need just one environment, use :func:`setup` instead.
        """
        return setup(env=self)

    def __repr__(self):
        c = type(self)
        rc = self.rating_class
        if callable(self.k_factor):
            f = self.k_factor
            k_factor = '.'.join([f.__module__, f.__name__])
        else:
            k_factor = '%.3f' % self.k_factor
        args = ('.'.join([c.__module__, c.__name__]), k_factor,
                '.'.join([rc.__module__, rc.__name__]), self.initial, self.beta)
        return ('%s(k_factor=%s, rating_class=%s, '
                'initial=%.3f, beta=%.3f)' % args)

class ModElo(Elo):
    def rate_1vs1(self, rating1, rating2, winnerby, margin, drawn=False):
        scores = (DRAW, DRAW) if drawn else (WIN, LOSS)
        return (self.rate(rating1, [(scores[0], rating2)], winnerby, margin),
                self.rate(rating2, [(scores[1], rating1)], winnerby, margin))

    def rate(self, rating, series, winnerby, margin):
        """Calculates new ratings by the game result series."""
        rating = self.ensure_rating(rating)
        k = self.k_factor(rating) if callable(self.k_factor) else self.k_factor
        if winnerby=="runs":
            k *= (1+MARGIN_RUN)**(margin/MARGIN_RUN_NORM)
        if winnerby=="wickets":
            k *= (1+MARGIN_WKTS)**(margin)
        new_rating = float(rating) + k * self.adjust(rating, series)
        if hasattr(rating, 'rated'):
            new_rating = rating.rated(new_rating)
        return new_rating

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

ratings_id = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
"Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
"West Indies":1000,"Zimbabwe":1000,}

def main():
    env = ModElo(initial=1000)
    ratings = []
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    for i in df_train.index:
        if df_train.Team1[i] not in teams_id.keys():
            continue
        if df_train.Team2[i] not in teams_id.keys():
            continue    

        loser_rate = ratings_id[df_train.Team1[i]]
        winner_rate = ratings_id[df_train.Winner[i]]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_rate = ratings_id[df_train.Team2[i]]

        loser_id = df_train.Team1[i]
        winner_id = df_train.Winner[i]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_id = df_train.Team2[i]

        ratings_id[winner_id], ratings_id[loser_id] = env.rate_1vs1(winner_rate, loser_rate, 
            df_train.Winnerby[i], df_train.Margin[i])
        # print loser_id, winner_id, rate_1vs1(winner_rate, loser_rate)
        # print ratings_id
        ratings.append(ratings_id.copy())
    return ratings

def visualize(ratings):
    full_ratings = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
    "Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
    "West Indies":1000,"Zimbabwe":1000,}
    xr = range(len(ratings))
    for i in full_ratings.keys():
        full_ratings[i] = [j[i] for j in ratings]
        lowess = sm.nonparametric.lowess(full_ratings[i], xr, frac=0.2)
        plt.plot(lowess[:, 0], lowess[:, 1])
    plt.show()

ratings = main()
if visual:
    visualize(ratings)

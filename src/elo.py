from datetime import datetime
import inspect

#: The actual score for win.
WIN = 1.
#: The actual score for draw.
DRAW = 0.5
#: The actual score for loss.
LOSS = 0.

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

    def __init__(self, k_factor=10, rating_class=float,
                 initial=1200, beta=200):
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
    def __init__(self, k_factor=10, rating_class=float,
                 initial=1200, beta=200, margin_run=0.2, 
                 margin_run_norm=50., margin_wkts=0.2,
                 k_factor_run=10, k_factor_wkts=10):
        self.k_factor = k_factor
        self.rating_class = rating_class
        self.initial = initial
        self.beta = beta
        self.margin_run = margin_run
        self.margin_run_norm = margin_run_norm
        self.margin_wkts = margin_wkts
        self.k_factor_run = k_factor_run
        self.k_factor_wkts = k_factor_wkts

    def rate_1vs1(self, rating1, rating2, winnerby, margin, drawn=False):
        scores = (DRAW, DRAW) if drawn else (WIN, LOSS)
        return (self.rate(rating1, [(scores[0], rating2)], winnerby, margin),
                self.rate(rating2, [(scores[1], rating1)], winnerby, margin))

    def rate(self, rating, series, winnerby, margin):
        """Calculates new ratings by the game result series."""
        rating = self.ensure_rating(rating)
        k = self.k_factor(rating) if callable(self.k_factor) else self.k_factor
        if winnerby=="runs":
            k = self.k_factor_run*((1+self.margin_run)**(1.*margin/self.margin_run_norm))
        if winnerby=="wickets":
            k = self.k_factor_wkts*((1+self.margin_wkts)**(margin))
        new_rating = float(rating) + k * self.adjust(rating, series)
        if hasattr(rating, 'rated'):
            new_rating = rating.rated(new_rating)
        return new_rating

class CustomElo(Elo):
    def __init__(self, rating_class=float,
                 initial=1200, beta=200, kf_wt_rating=1., 
                 kf_wt_margin_runs=.1, kf_wt_margin_wkts=.1,
                 kf_wt_winnerby=.1,
                 kf_wt_tossdecision=.1,kf_wt_tosswinner=.1):
        self.rating_class = rating_class
        self.initial = initial
        self.beta = beta
        self.kf_wt_rating = kf_wt_rating
        self.kf_wt_margin_runs = kf_wt_margin_runs
        self.kf_wt_margin_wkts = kf_wt_margin_wkts
        self.kf_wt_winnerby = kf_wt_winnerby
        self.kf_wt_tossdecision = kf_wt_tossdecision
        self.kf_wt_tosswinner = kf_wt_tosswinner

    def rate_1vs1(self, rating1, rating2, feats, drawn=False):
        scores = (DRAW, DRAW) if drawn else (WIN, LOSS)
        return (self.rate(rating1, [(scores[0], rating2)], feats),
                self.rate(rating2, [(scores[1], rating1)], feats))

    def rate(self, rating, series, feats):
        """Calculates new ratings by the game result series."""
        rating = self.ensure_rating(rating)
        k = self.kf_wt_rating*rating + \
            self.kf_wt_winnerby*feats["kf_wt_winnerby"] + \
            self.kf_wt_tossdecision*feats["kf_wt_tossdecision"] + \
            self.kf_wt_tosswinner*feats["kf_wt_tosswinner"]
        if feats["kf_wt_winnerby"]:
            k += self.kf_wt_margin_runs*feats["kf_wt_margin"]
        else:
            k += self.kf_wt_margin_wkts*feats["kf_wt_margin"]
        new_rating = float(rating) + k * self.adjust(rating, series)
        if hasattr(rating, 'rated'):
            new_rating = rating.rated(new_rating)
        return new_rating
import numpy as np
import scipy
import scipy.sparse
import tempfile


class InstanceGenerator:
    def __init__(self, generator_function, *args, **kwargs):
        self.rng = np.random.RandomState()
        self.generator_function = generator_function
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        # new temporary file
        file = tempfile.NamedTemporaryFile(mode="w+", suffix=".lp")
        self.generator_function(rng=self.rng, filename=file.name, *self.args, **self.kwargs)
        return file

    def seed(self, seed):
        self.rng.seed(seed)


def generate_setcover(rng, filename, nrows, ncols, density, max_coef=100):
    """
    Generate a set cover instance with specified characteristics, and writes
    it to a file in the LP format.

    Algorithm described in:
        E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
        and subgradient optimization: A computational study, Mathematical
        Programming, 12 (1980), 37-60.

    Parameters
    ----------
    rng: numpy.random.RandomState
        Random number generator
    filename: str
        An existing file, to which the instance is to be written in the LP format.
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    indices = np.empty((nnzrs,), dtype=int)

    # sample column indices
    indices[: 2 * ncols] = np.arange(2 * ncols) % ncols  # force at leats 2 rows per col
    indices[2 * ncols :] = (
        rng.choice(ncols * (nrows - 2), size=nnzrs - (2 * ncols), replace=False) % ncols
    )  # remaining column indexes are random

    # count the resulting number of rows, for each column
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample row indices
    i = 0
    indptr = [0]
    indices[:nrows] = rng.permutation(nrows)  # pre-fill to force at least 1 column per row
    for n in col_nrows:

        # column is already filled, nothing to do
        if i + n <= nrows:
            pass

        # column is empty, fill with random rows
        elif i >= nrows:
            indices[i : i + n] = rng.choice(nrows, size=n, replace=False)

        # column is partially filled, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows : i + n] = rng.choice(remaining_rows, size=i + n - nrows, replace=False)

        i += n
        indptr.append(i)

    # sample objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr), shape=(nrows, ncols)
    ).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, "w") as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i] : indptr[i + 1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))


def generate_cauctions(
    rng,
    filename,
    n_items=100,
    n_bids=500,
    min_value=1,
    max_value=100,
    value_deviation=0.5,
    add_item_prob=0.9,
    max_n_sub_bids=5,
    additivity=0.2,
    budget_factor=1.5,
    resale_factor=0.5,
    integers=False,
    warnings=False,
):
    """
    Generate a Combinatorial Auction instance with specified characteristics, and writes
    it to a file in the LP format.

    Algorithm described in:
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.
    section 4.3., the 'arbitrary' scheme.

    Parameters
    ----------
    rng : numpy.random.RandomState
        A random number generator.
    filename: str
        An existing file, to which the instance is to be written in the LP format.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, add_item_prob, rng):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return rng.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * rng.rand(n_items)

    # item compatibilities
    compats = np.triu(rng.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = rng.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = rng.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while rng.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, rng)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, add_item_prob, rng)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])
        ]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # write problem
    with open(filename, "w") as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]
        file.write("maximize\nOBJ:")
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" +{price} x{i+1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                for i in item_bids:
                    file.write(f" +1 x{i+1}")
                file.write(f" <= 1\n")

        file.write("\nbinary\n")
        for i in range(len(bids)):
            file.write(f" x{i+1}")

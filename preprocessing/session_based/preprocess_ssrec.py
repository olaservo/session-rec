import numpy as np
import pandas as pd
from _datetime import timezone, datetime, timedelta
import time

# data config (all methods)
DATA_PATH = '../../ssrec/data/'
DATA_PATH_PROCESSED = '../data/ssrec/prepared/'
DATA_FILE = 'sessions_viewed_with_purchased.csv'
DATA_VIEWS_FILE = 'views'
DATA_PURCHASES_FILE = 'purchases'

# filtering default config
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5
# TODO: find optimal number or use stat?
MAX_SESSION_LENGTH = 100

# days test default config
DAYS_TEST = 2

# slicing default config
NUM_SLICES = 7
DAYS_OFFSET = 0
DAYS_SHIFT = 1
DAYS_TRAIN = 5


def preprocess(path=DATA_PATH, file=DATA_FILE, path_proc=DATA_PATH_PROCESSED,
               min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, use_slices=False,
               dry_run=False):
    df_views, df_purchases = load_data(path + file)
    df_views = filter_data(df_views, min_item_support, min_session_length)
    if dry_run:
        print('Loaded and filtered data')
        print(df_views)
        print(f'Full data set for Views: \n\tEvents: {len(df_views)}\n\tSessions: {df_views.SessionId.nunique()}\n'
              f'\tItems: {df_views.ItemId.nunique()}')
        print(df_purchases)
        return
    if use_slices:
        slice_data(df_views, path_proc + file, num_slices=NUM_SLICES, days_offset=DAYS_OFFSET, days_shift=DAYS_SHIFT,
                   days_train=DAYS_TRAIN, days_test=DAYS_TEST)
    else:
        split_data(df_views, path_proc + DATA_VIEWS_FILE, days_test=DAYS_TEST)
    df_purchases.to_csv(path_proc + DATA_PURCHASES_FILE + '.txt', sep='\t', index=False)


def load_data(file):
    start = time.perf_counter()
    data = pd.read_csv(file)

    del (data['eventId'])
    data['SessionId'] = data['sessionId']
    data['ItemId'] = data['styleId']
    data['TimeO'] = pd.to_datetime(data['eventTime'])
    data['Time'] = data.TimeO.apply(lambda t: t.timestamp())
    data['UserId'] = data['shopperId']
    del (data['sessionId'])
    del (data['styleId'])
    del (data['shopperId'])
    del (data['eventTime'])

    data.sort_values(['SessionId', 'Time'], inplace=True)

    # Split purchases into separate df
    purchases_data = data[data['purchasedQuantity'] > 0]
    del (data['purchasedQuantity'])
    del (purchases_data['purchasedQuantity'])

    end = time.perf_counter()
    print(f'Loaded data in {end-start:0.4f}')

    return data, purchases_data


def filter_data(data, min_item_support, min_session_length):
    # TODO: should views include purchased items from session?
    # TODO: Truncate sessions to a max length
    start = time.perf_counter()
    # filter item support
    data['support'] = data.groupby('ItemId')['ItemId'].transform('count')
    print(f'Rows before filtering for min_item_support of {min_item_support}: {len(data.index)}')
    data = data[data['support'] >= min_item_support]
    print(f'Rows after filtering for min_item_support of {min_item_support}: {len(data.index)}')
    end = time.perf_counter()
    print(f'Filtered for min_item_support in {end-start:0.4f}')
    data = data.drop(['support'], axis=1)

    start = time.perf_counter()
    # filter session length
    data['session_length'] = data.groupby('SessionId')['SessionId'].transform('count')
    print(f'Rows before filtering for min_session_length of {min_session_length}: {len(data.index)}')
    data = data[data['session_length'] >= min_session_length]
    print(f'Rows after filtering for min_session_length of {min_session_length}: {len(data.index)}')
    end = time.perf_counter()
    print(f'Filtered for min_session_length in {end-start:0.4f}')
    data = data.drop(['session_length'], axis=1)

    # output
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data


def split_data(data, output_file, days_test):
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)

    start = time.perf_counter()
    # TODO: should we split by session start or session end?
    # The original method from this repo is splitting by end (max)
    data['max_session_time'] = data.groupby('SessionId')['Time'].transform('max')
    train = data[data['max_session_time'] < test_from.timestamp()]
    test = data[data['max_session_time'] >= test_from.timestamp()]
    end = time.perf_counter()
    print(f'Split test and train data by date in {end-start:0.4f}')

    start = time.perf_counter()
    # Filter test set to only include sessions where the item it is also in the training set
    test = test[np.in1d(test.ItemId, train.ItemId)]
    end = time.perf_counter()
    print(f'Filter test data by ItemId in {end-start:0.4f}')
    start = time.perf_counter()
    # Filter out sessions that are split across train and test
    # get unique first to speed up intersection
    train_session_ids = train.SessionId.unique()
    test_session_ids = test.SessionId.unique()
    intersecting_session_ids = np.intersect1d(train_session_ids, test_session_ids, assume_unique=True)
    train = train[~train['SessionId'].isin(intersecting_session_ids)]
    test = test[~test['SessionId'].isin(intersecting_session_ids)]
    end = time.perf_counter()
    print(f'Filter test and train data by SessionId in {end-start:0.4f}')

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)

# TODO: optimize if using this
def slice_data(data, output_file, num_slices, days_offset, days_shift, days_train, days_test):
    for slice_id in range(0, num_slices):
        split_data_slice(data, output_file, slice_id, days_offset + (slice_id * days_shift), days_train, days_test)

# TODO: optimize if using this
def split_data_slice(data, output_file, slice_id, days_offset, days_train, days_test):
    data_start = datetime.fromtimestamp(data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp(data.Time.max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))

    start = datetime.fromtimestamp(data.Time.min(), timezone.utc) + timedelta(days_offset)
    middle = start + timedelta(days_train)
    end = middle + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle.date().isoformat(), end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle.date().isoformat()))

    train.to_csv(output_file + '_train_full.' + str(slice_id) + '.txt', sep='\t', index=False)

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(),
                 end.date().isoformat()))

    test.to_csv(output_file + '_test.' + str(slice_id) + '.txt', sep='\t', index=False)


# ------------------------------------- 
# MAIN TEST
# --------------------------------------
if __name__ == '__main__':
    preprocess(dry_run=True)

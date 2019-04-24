# MIMIC IIIv14 on postgres 9.4
import numpy as np
import pandas as pd
from pandas import DataFrame
import psycopg2
from sklearn import metrics
from datetime import datetime
from datetime import timedelta
import sys
import os
import time

from os import environ
from os.path import isfile, isdir, splitext
import argparse
import cPickle
import numpy as np
import numpy.random as npr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datapackage_io_util import (
    load_datapackage_schema,
    load_sanitized_df_from_csv,
    save_sanitized_df_to_csv,
    sanitize_df,
)

# Output filenames
static_filename = 'static_data.csv'
static_columns_filename = 'static_colnames.txt'

dynamic_filename = 'vitals_hourly_data.csv'
columns_filename = 'vitals_colnames.txt'
subjects_filename = 'subjects.npy'
times_filename = 'fenceposts.npy'
dynamic_hd5_filename = 'vitals_hourly_data.h5'
dynamic_hd5_filt_filename = 'all_hourly_data.h5'

codes_filename = 'C.npy'
codes_hd5_filename = 'C.h5'
idx_hd5_filename = 'C_idx.h5'

outcome_filename = 'outcomes_hourly_data.csv'
outcome_hd5_filename = 'outcomes_hourly_data.h5'
outcome_columns_filename = 'outcomes_colnames.txt'

# SQL command params
dbname = 'mimic'
schema_name = 'mimiciii'

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']

def add_outcome_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]
    on_hrs = set()

    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))

    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
    hours = list(off_hrs) + list(on_hrs)
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hours, 'on':on_vals}) #icustay_id': icustay_id})


def add_blank_indicators(out_gb):
    subject_id = out_gb['subject_id'].unique()[0]
    hadm_id = out_gb['hadm_id'].unique()[0]
    #icustay_id = out_gb['icustay_id'].unique()[0]
    max_hrs = out_gb['max_hours'].unique()[0]

    hrs = range(max_hrs + 1)
    vals = list([0]*len(hrs))
    return pd.DataFrame({'subject_id': subject_id, 'hadm_id':hadm_id,
                        'hours_in':hrs, 'on':vals})#'icustay_id': icustay_id,

def get_values_by_name_from_df_column_or_index(data_df, colname):
    """ Easily get values for named field, whether a column or an index

    Returns
    -------
    values : 1D array
    """
    try:
        values = data_df[colname]
    except KeyError as e:
        if colname in data_df.index.names:
            values = data_df.index.get_level_values(colname)
        else:
            raise e
    return values

def continuous_outcome_processing(out_data, data, icustay_timediff):
    """

    Args
    ----
    out_data : pd.DataFrame
        index=None
        Contains subset of icustay_id corresp to specific sessions where outcome observed.
    data : pd.DataFrame
        index=icustay_id
        Contains full population of static demographic data

    Returns
    -------
    out_data : pd.DataFrame
    """
    out_data['intime'] = out_data['icustay_id'].map(data['intime'].to_dict())
    out_data['outtime'] = out_data['icustay_id'].map(data['outtime'].to_dict())
    out_data['max_hours'] = out_data['icustay_id'].map(icustay_timediff)
    out_data['starttime'] = out_data['starttime'] - out_data['intime']
    out_data['starttime'] = out_data.starttime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data['endtime'] = out_data['endtime'] - out_data['intime']
    out_data['endtime'] = out_data.endtime.apply(lambda x: x.days*24 + x.seconds//3600)
    out_data = out_data.groupby(['icustay_id'])

    return out_data
#
def fill_missing_times(df_by_sid_hid_itemid):
    max_hour = df_by_sid_hid_itemid.index.get_level_values(max_hours)[0]
    missing_hours = list(set(range(max_hour+1)) - set(df_by_sid_hid_itemid['hours_in'].unique()))
    # Add rows
    sid = df_by_sid_hid_itemid.subject_id.unique()[0]
    hid = df_by_sid_hid_itemid.hadm_id.unique()[0]
    icustay_id = df_by_sid_hid_itemid.icustay_id.unique()[0]
    itemid = df_by_sid_hid_itemid.itemid.unique()[0]
    filler = pd.DataFrame({'subject_id':[sid]*len(missing_hours),
                           'hadm_id':[hid]*len(missing_hours),
                           'icustay_id':[icustay_id]*len(missing_hours),
                           'itemid':[itemid]*len(missing_hours),
                           'hours_in':missing_hours,
                           'value':[np.nan]*len(missing_hours),
                            'max_hours': [max_hour]*len(missing_hours)})
    return pd.concat([df_by_sid_hid_itemid, filler], axis=0)

def save_pop(
        data_df, outPath, static_filename, pop_size_int,
        static_data_schema, host=None
    ):
    # Connect to local postgres version of mimic

    # Serialize to disk
    csv_fpath = os.path.join(outPath, static_filename)
    save_sanitized_df_to_csv(csv_fpath, data_df, static_data_schema)
    """
    # Lower cost to doing this conversion and of serializing it afterwards
    #  (http://matthewrocklin.com/blog/work/2015/03/16/Fast-Serialization)
    data['admission_type'] = data['admission_type'].astype('category')
    data['gender'] = data['gender'].astype('category')
    data['first_careunit'] = data['first_careunit'].astype('category')
    data['ethnicity'] = data['ethnicity'].astype('category')

    # Process the timestamps
    data['intime'] = pd.to_datetime(data['intime']) #, format="%m/%d/%Y"))
    data['outtime'] = pd.to_datetime(data['outtime'])
    data['admittime'] = pd.to_datetime(data['admittime'])
    data['dischtime'] = pd.to_datetime(data['dischtime'])
    data['deathtime'] = pd.to_datetime(data['deathtime'])
    # Serialize to disk
    data.to_csv(os.path.join(outPath, static_filename))
    """
    return data_df

# From Dave's approach!
def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = DataFrame.from_csv(mimic_mapping_filename, index_col=None).fillna('').astype(str)
    var_map = var_map.ix[(var_map['LEVEL2'] != '') & (var_map.COUNT>0)]
    var_map = var_map.ix[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)

    return var_map

def get_variable_ranges(range_filename):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [ 'LEVEL2', 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH' ]
    to_rename = dict(zip(columns, [ c.replace(' ', '_') for c in columns ]))
    to_rename['LEVEL2'] = 'VARIABLE'
    var_ranges = DataFrame.from_csv(range_filename, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename_axis(to_rename, axis=1, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges['VARIABLE'] = map(str.lower, var_ranges['VARIABLE'])
    var_ranges.set_index('VARIABLE', inplace=True)
    var_ranges = var_ranges.ix[var_ranges.notnull().all(axis=1)]

    return var_ranges

UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]

def standardize_units(X, name_col='itemid', unit_col='valueuom', value_col='value', inplace=True):
    if not inplace: X = X.copy()
    name_col_vals = get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except:
        print("Can't call *.str")
        print(name_col_vals)
        print(unit_col_vals)
        raise

    #name_filter, unit_filter = [
    #    (lambda n: col.contains(n, case=False, na=False)) for col in (name_col_vals, unit_col_vals)
    #]
    # TODO(mmd): Why does the above not work, but the below does?
    name_filter = lambda n: name_col_vals.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None: needs_conversion_filter_idx |= name_filter(unit) | unit_filter(unit)
        if rng_check_fn is not None: needs_conversion_filter_idx |= rng_check_fn(X[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        X.loc[idx, value_col] = convert_fn(X[value_col][idx])

    return X

def range_unnest(df, col, out_col_name=None, reset_index=False):
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y+1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat

# TODO(mmd): improve args
def save_numerics(
    data, X, I, var_map, var_ranges, outPath, dynamic_filename, columns_filename, subjects_filename,
    times_filename, dynamic_hd5_filename, group_by_level2, apply_var_limit, min_percent
):
    assert len(data) > 0 and len(X) > 0, "Must provide some input data to process."

    var_map = var_map[
        ['LEVEL2', 'ITEMID', 'LEVEL1']
    ].rename_axis(
        {'LEVEL2': 'LEVEL2', 'LEVEL1': 'LEVEL1', 'ITEMID': 'itemid'}, axis=1
    ).set_index('itemid')

    X['value'] = pd.to_numeric(X['value'], 'coerce')
    X.astype({k: int for k in ID_COLS}, inplace=True)

    to_hours = lambda x: max(0, x.days*24 + x.seconds // 3600)

    X = X.set_index('icustay_id').join(data[['intime']])
    X['hours_in'] = (X['charttime'] - X['intime']).apply(to_hours)

    X.drop(columns=['charttime', 'intime'], inplace=True)
    X.set_index('itemid', append=True, inplace=True)

    # Pandas has a bug with the below for small X
    #X = X.join([var_map, I]).set_index(['label', 'LEVEL1', 'LEVEL2'], append=True)
    X = X.join(var_map).join(I).set_index(['label', 'LEVEL1', 'LEVEL2'], append=True)
    standardize_units(X, name_col='LEVEL1', inplace=True)

    if apply_var_limit > 0: 
        X = apply_variable_limits(X, var_ranges, 'LEVEL2')

    group_item_cols = ['LEVEL2'] if group_by_level2 else ITEM_COLS
    X = X.groupby(ID_COLS + group_item_cols + ['hours_in']).agg(['mean', 'std', 'count'])
    X.columns = X.columns.droplevel(0)
    X.columns.names = ['Aggregation Function']

    data['max_hours'] = (data['outtime'] - data['intime']).apply(to_hours)

    # TODO(mmd): Maybe can just create the index directly?
    missing_hours_fill = range_unnest(data, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN

    # TODO(mmd): The below is a bit wasteful.
    #itemids = var_map.join(I['label']).reset_index()[group_item_cols].drop_duplicates()
    #itemids['tmp'] = np.NaN

    #missing_hours_fill = missing_hours_fill.merge(itemids, on='tmp', how='outer')

    fill_df = data.reset_index()[ID_COLS].join(missing_hours_fill.set_index('icustay_id'), on='icustay_id')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

    # Pivot table droups NaN columns so you lose any uniformly NaN.
    X = X.unstack(level = group_item_cols)
    X.columns = X.columns.reorder_levels(order=group_item_cols + ['Aggregation Function'])
   

    #X = X.reset_index().pivot_table(index=ID_COLS + ['hours_in'], columns=group_item_cols, values=X.columns)
    X = X.reindex(fill_df.index)

    #X.columns = X.columns.droplevel(0).reorder_levels(order=[1, 0])
    #if group_by_level2:
    #    X.columns.names = ['LEVEL2', 'Aggregation Function'] # Won't work with ungrouped!
    #else:
    #    X.columns.names = ['itemid', 'Aggregation Function']
    #    X.columms = X.MultiIndex.from_frame(X[ITEM_COLS])

    X = X.sort_index(axis=0).sort_index(axis=1)

    print "Shape of X : ", X.shape

    # Turn back into columns
    if columns_filename is not None:
        col_names  = [str(x) for x in X.columns.values]
        with open(os.path.join(outPath, columns_filename), 'w') as f: f.write('\n'.join(col_names))

    # Get the max time for each of the subjects so we can reconstruct!
    if subjects_filename is not None:
        np.save(os.path.join(outPath, subjects_filename), data['subject_id'].as_matrix())
    if times_filename is not None: 
        np.save(os.path.join(outPath, times_filename), data['max_hours'].as_matrix())

    #fix nan in count to be zero
    idx = pd.IndexSlice
    if group_by_level2:
        X.loc[:, idx[:, 'count']] = X.loc[:, idx[:, 'count']].fillna(0)
    else:
        X.loc[:, idx[:,:,:,:, 'count']] = X.loc[:, idx[:,:,:,:, 'count']].fillna(0)
    
    # Drop columns that have very few recordings
    n = round((1-min_percent/100.0)*X.shape[0])
    drop_col = []
    for k in X.columns:
        if k[-1] == 'mean':
            if X[k].isnull().sum() > n:
                drop_col.append(k[:-1])
    X = X.drop(columns = drop_col)

    ########
    if dynamic_filename is not None: np.save(os.path.join(outPath, dynamic_filename), X.as_matrix())
    if dynamic_hd5_filename is not None: X.to_hdf(os.path.join(outPath, dynamic_hd5_filename), 'X')

    return X

def save_icd9_codes(data, codes, outPath, codes_filename, codes_h5_filename):
    codes.set_index(ID_COLS, inplace=True)
    codes.to_hdf(os.path.join(outPath, codes_h5_filename), 'C')
    return codes

def save_outcome(
    data, dbname, schema_name, outPath, outcome_filename, outcome_hd5_filename,
    outcome_columns_filename, outcome_schema, host=None
):
    """ Retrieve outcomes from DB and save to disk

    Vent and vaso are both there already - so pull the start and stop times from there! :)

    Returns
    -------
    Y : Pandas dataframe
        Obeys the outcomes data spec
    """
    icuids_to_keep = get_values_by_name_from_df_column_or_index(data, 'icustay_id')
    icuids_to_keep = set([str(s) for s in icuids_to_keep])

    # Add a new column called intime so that we can easily subtract it off
    data = data.reset_index()
    data = data.set_index('icustay_id')
    data['intime'] = pd.to_datetime(data['intime']) #, format="%m/%d/%Y"))
    data['outtime'] = pd.to_datetime(data['outtime'])
    icustay_timediff_tmp = data['outtime'] - data['intime']
    icustay_timediff = pd.Series([timediff.days*24 + timediff.seconds//3600
                                  for timediff in icustay_timediff_tmp], index=data.index.values)


    # Setup access to PSQL db
    args = dict(dbname=dbname, host=host) if host is not None else dict(dbname=dbname)
    con = psycopg2.connect(**args)
    cur = con.cursor()

    # Query on ventilation data
    cur.execute('SET search_path to ' + schema_name)
    query = """
    select i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
    FROM icustay_detail i
    INNER JOIN ventdurations v ON i.icustay_id = v.icustay_id
    where v.icustay_id in ({icuids})
    and v.starttime between intime and outtime
    and v.endtime between intime and outtime;
    """.format(icuids=','.join(icuids_to_keep))
    vent_data = pd.read_sql_query(query, con)
    vent_data = continuous_outcome_processing(vent_data, data, icustay_timediff)
    vent_data = vent_data.apply(add_outcome_indicators)
    vent_data.rename(columns = {'on':'vent'}, inplace=True)
    vent_data = vent_data.reset_index()

    # Get the patients without the intervention in there too so that we
    ids_with = vent_data['icustay_id']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)
    #ids_without = map(int, ids_without)

    # Create a new fake dataframe with blanks on all vent entries
    out_data = data.copy(deep=True)
    out_data = out_data.reset_index()
    out_data = out_data.set_index('icustay_id')
    out_data = out_data.iloc[out_data.index.isin(ids_without)]
    out_data = out_data.reset_index()
    out_data = out_data[['subject_id', 'hadm_id', 'icustay_id']]
    out_data['max_hours'] = out_data['icustay_id'].map(icustay_timediff)

    # Create all 0 column for vent
    out_data = out_data.groupby('icustay_id')
    out_data = out_data.apply(add_blank_indicators)
    out_data.rename(columns = {'on':'vent'}, inplace=True)
    out_data = out_data.reset_index()

    # Concatenate all the data vertically
    Y = pd.concat([vent_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']],
                   out_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent']]],
                  axis=0)

    # Start merging all other interventions
    table_names = ['vasopressordurations', 'adenosinedurations', 'dobutaminedurations', 'dopaminedurations', 'epinephrinedurations', 'isupreldurations', 
                    'milrinonedurations', 'norepinephrinedurations', 'phenylephrinedurations', 'vasopressindurations']
    column_names = ['vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 
                    'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']

    # TODO(mmd): This section doesn't work. What is its purpose?
    for t, c in zip(table_names, column_names):
        # TOTAL VASOPRESSOR DATA
        cur.execute('SET search_path to ' + schema_name)
        query = """
        select i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
        FROM icustay_detail i
        INNER JOIN {table} v ON i.icustay_id = v.icustay_id
        where v.icustay_id in ({icuids})
        and v.starttime between intime and outtime
        and v.endtime between intime and outtime;
        """.format(icuids=','.join(icuids_to_keep), table=t)
        new_data = pd.read_sql_query(query,con)
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns = {'on':c}, inplace=True)
        new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population, in which c was never
        # performed.
        if not c in new_data:
            print "Column ", c, " not in data."
            continue

        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', c]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[c] = Y[c].astype(int)
        #Y = Y.sort_values(['subject_id', 'icustay_id', 'hours_in']) #.merge(df3,on='name')
        Y = Y.reset_index(drop=True)
        print 'Extracted ' + c + ' from ' + t


    tasks=["colloid_bolus", "crystalloid_bolus", "nivdurations"]

    for task in tasks:
        cur.execute('SET search_path to ' + schema_name)
        if task=='nivdurations':
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.starttime, v.endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.starttime between intime and outtime
            and v.endtime between intime and outtime;
            """.format(icuids=','.join(icuids_to_keep), table=task)
        else:
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.charttime AS starttime, 
                   v.charttime AS endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.charttime between intime and outtime
            """.format(icuids=','.join(icuids_to_keep), table=task)
        new_data = pd.read_sql_query(query, con=con)
        if new_data.shape[0] == 0:
            continue
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns = {'on':task}, inplace=True)
        new_data = new_data.reset_index()
        new_data.to_csv('new_task.csv')
        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', task]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[task] = Y[task].astype(int)
        Y = Y.reset_index(drop=True)
        print 'Extracted ' + task
    

    # TODO: ADD THE RBC/PLT/PLASMA DATA
    # TODO: ADD DIALYSIS DATA
    # TODO: ADD INFECTION DATA

    cur.close()
    con.close()

    Y = Y.filter(items=['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent'] + column_names + tasks)
    Y.subject_id = Y.subject_id.astype(int)
    Y.icustay_id = Y.icustay_id.astype(int)
    Y.hours_in = Y.hours_in.astype(int)
    Y.vent = Y.vent.astype(int)
    Y.vaso = Y.vaso.astype(int)
    y_id_cols = ID_COLS + ['hours_in']
    Y = Y.sort_values(y_id_cols)
    Y.set_index(y_id_cols, inplace=True)

    print 'Shape of Y : ', Y.shape

    # SAVE AS NUMPY ARRAYS AND TEXT FILES
    #np_Y = Y.as_matrix()
    #np.save(os.path.join(outPath, outcome_filename), np_Y)

    # Turn back into columns
    df = Y.reset_index()
    df = sanitize_df(df, outcome_schema) 
    csv_fpath = os.path.join(outPath, outcome_filename)
    save_sanitized_df_to_csv(csv_fpath, df, outcome_schema)

    col_names  = list(df.columns.values)
    col_names = col_names[3:]
    with open(os.path.join(outPath, outcome_columns_filename), 'w') as f:
        f.write('\n'.join(col_names))

    # TODO(mmd): Why does df have the index? Is sanitize making multiindex?
    # SAVE THE DATA AS A PANDAS OBJECT
    # TODO(mike hughes): Why writing out Y after you've separately sanitized df?
    Y.to_hdf(os.path.join(outPath, outcome_hd5_filename), 'Y')
    return df

# Apply the variable limits to remove things
# TODO(mmd): controlled printing.
def apply_variable_limits(df, var_ranges, var_names_index_col='LEVEL2'):
    idx_vals        = df.index.get_level_values(var_names_index_col)
    non_null_idx    = ~df.value.isnull()
    var_names       = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    for var_name in var_names:
        var_name_lower = var_name.lower()
        if var_name_lower not in var_range_names:
            print("No known ranges for %s" % var_name)
            continue

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x] for x in ('OUTLIER_LOW','OUTLIER_HIGH','VALID_LOW','VALID_HIGH')
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx  = (df.value < outlier_low_val)
        outlier_high_idx = (df.value > outlier_high_val)
        valid_low_idx    = ~outlier_low_idx & (df.value < valid_low_val)
        valid_high_idx   = ~outlier_high_idx & (df.value > valid_high_val)

        var_outlier_idx   = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        df.loc[var_outlier_idx, 'value'] = np.nan
        df.loc[var_valid_low_idx, 'value'] = valid_low_val
        df.loc[var_valid_high_idx, 'value'] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0:
            print(
                "%s had %d / %d rows cleaned:\n"
                "  %d rows were strict outliers, set to np.nan\n"
                "  %d rows were low valid outliers, set to %.2f\n"
                "  %d rows were high valid outliers, set to %.2f\n"
                "" % (
                    var_name,
                    n_outlier + n_valid_low + n_valid_high, sum(running_idx),
                    n_outlier, n_valid_low, valid_low_val, n_valid_high, valid_high_val
                )
            )

    return df

def plot_variable_histograms(col_names, df):
    # Plot some of the data, just to make sure it looks ok
    for c, vals in df.iteritems():
        n = vals.dropna().count()
        if n < 2: continue

        # get median, variance, skewness
        med = vals.dropna().median()
        var = vals.dropna().var()
        skew = vals.dropna().skew()

        # plot
        fig = plt.figure(figsize=(13, 6))
        plt.subplots(figsize=(13,6))
        vals.dropna().plot.hist(bins=100, label='HIST (n={})'.format(n))

        # fake plots for KS test, median, etc
        plt.plot([], label=' ',color='lightgray')
        plt.plot([], label='Median: {}'.format(format(med,'.2f')),
                 color='lightgray')
        plt.plot([], label='Variance: {}'.format(format(var,'.2f')),
                 color='lightgray')
        plt.plot([], label='Skew: {}'.format(format(skew,'.2f')),
                 color='light:gray')

        # add title, labels etc.
        plt.title('{} measurements in ICU '.format(str(c)))
        plt.xlabel(str(c))
        plt.legend(loc="upper left", bbox_to_anchor=(1,1),fontsize=12)
        plt.xlim(0, vals.quantile(0.99))
        fig.savefig(os.path.join(outPath, (str(c) + '_HIST_.png')), bbox_inches='tight')

# Main, where you can call what makes sense.
if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_path', type=str, default= '/scratch/{}/phys_acuity_modelling/data'.format(os.environ['USER']),
                    help='Enter the path you want the output')
    ap.add_argument('--resource_path',
        type=str,
        default=os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR/resources/"))
    ap.add_argument('--extract_pop', type=int, default=1,
                    help='Whether or not to extract population data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')

    ap.add_argument('--extract_numerics', type=int, default=1,
                    help='Whether or not to extract numerics data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--extract_outcomes', type=int, default=1,
                    help='Whether or not to extract outcome data: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--extract_codes', type=int, default=1,
                    help='Whether or not to extract ICD9 codes: 0 - no extraction, ' +
                    '1 - extract if not present in the data directory, 2 - extract even if there is data')
    ap.add_argument('--pop_size', type=int, default=0,
                    help='Size of population to extract')
    ap.add_argument('--exit_after_loading', type=int, default=0)
    ap.add_argument('--var_limits', type=int, default=1,
                    help='Whether to create a version of the data with variable limits included. ' +
                    '1 - apply variable limits, 0 - do not apply variable limits')
    ap.add_argument('--plot_hist', type=int, default=1,
                    help='Whether to plot the histograms of the data')
    ap.add_argument('--psql_host', type=str, default=None,
                    help='Postgres host. Try "/var/run/postgresql/" for Unix domain socket errors.')
    ap.add_argument('--group_by_level2', action='store_false', dest='group_by_level2', default=True,
                    help='Do group by level2.')
    
    ap.add_argument('--min_percent', type=float, default=0.0,
                    help='Minimum percentage of row numbers need to be observations for each numeric column.' +
                    'min_percent = 1 means columns with more than 99 percent of nan will be removed')
    ap.add_argument('--min_age', type=int, default=15,
                    help='Minimum age of patients to be included')
    ap.add_argument('--min_duration', type=int, default=12,
                    help='Minimum hours of stay to be included')
    ap.add_argument('--max_duration', type=int, default=240,
                    help='Maximum hours of stay to be included')
                   

    #############
    # Parse args
    args = vars(ap.parse_args())
    for key in sorted(args.keys()):
        print key, args[key]

    if not isdir(args['resource_path']):
        raise ValueError("Invalid resource_path: %s" % args['resource_path'])

    mimic_mapping_filename = os.path.join(args['resource_path'], 'itemid_to_variable_map.csv')
    range_filename = os.path.join(args['resource_path'], 'variable_ranges.csv')

    # Load specs for output tables
    static_data_schema = load_datapackage_schema(
        os.path.join(args['resource_path'], 'static_data_spec.json'))
    outcome_data_schema = load_datapackage_schema(
        os.path.join(args['resource_path'], 'outcome_data_spec.json'))

    if not isdir(args['out_path']):
        print 'ERROR: OUTPATH %s DOES NOT EXIST' % args['out_path']
        sys.exit()
    else:
        outPath = args['out_path']


    # Modify the filenames
    if args['pop_size'] > 0:
        pop_size = str(args['pop_size'])

        static_filename = splitext(static_filename)[0] + '_' + pop_size + splitext(static_filename)[1]
        dynamic_filename = splitext(dynamic_filename)[0] + '_' + pop_size + splitext(dynamic_filename)[1]
        #columns_filename = splitext(columns_filename)[0] + '_' + pop_size + splitext(columns_filename)[1]
        subjects_filename = splitext(subjects_filename)[0] + '_' + pop_size + splitext(subjects_filename)[1]
        times_filename = splitext(times_filename)[0] + '_' + pop_size + splitext(times_filename)[1]
        dynamic_hd5_filename = splitext(dynamic_hd5_filename)[0] + '_' + pop_size + splitext(dynamic_hd5_filename)[1]
        outcome_filename = splitext(outcome_filename)[0] + '_' + pop_size + splitext(outcome_filename)[1]
        dynamic_hd5_filt_filename = splitext(dynamic_hd5_filt_filename)[0] + '_' + pop_size + splitext(dynamic_hd5_filt_filename)[1]
        outcome_hd5_filename = splitext(outcome_hd5_filename)[0] + '_' + pop_size + splitext(outcome_hd5_filename)[1]
        #outcome_columns_filename = splitext(outcome_columns_filename)[0] + '_' + pop_size + splitext(outcome_columns_filename)[1]
        codes_filename = splitext(codes_filename)[0] + '_' + pop_size + splitext(codes_filename)[1]
        codes_hd5_filename = splitext(codes_hd5_filename)[0] + '_' + pop_size + splitext(codes_hd5_filename)[1]
        idx_hd5_filename = splitext(idx_hd5_filename)[0] + '_' + pop_size + splitext(idx_hd5_filename)[1]

    query_args = {'dbname': dbname}
    if args['psql_host'] is not None: query_args['host'] = args['psql_host']

    #############
    # Population extraction

    data = None
    if (args['extract_pop'] == 0 | (args['extract_pop'] == 1) ) & isfile(os.path.join(outPath, static_filename)):
        data = DataFrame.from_csv(os.path.join(outPath, static_filename))
        data = sanitize_df(data, static_data_schema)

        """
        data['admission_type'] = data['admission_type'].astype('category')
        data['gender'] = data['gender'].astype('category')
        data['first_careunit'] = data['first_careunit'].astype('category')
        data['ethnicity'] = data['ethnicity'].astype('category')

        data['intime'] = pd.to_datetime(data['intime']) #, format="%m/%d/%Y"))
        data['outtime'] = pd.to_datetime(data['outtime'])
        data['admittime'] = pd.to_datetime(data['admittime'])
        data['dischtime'] = pd.to_datetime(data['dischtime'])
        data['deathtime'] = pd.to_datetime(data['deathtime'])
        """
    elif (args['extract_pop'] == 1 & (not isfile(os.path.join(outPath, static_filename)))) | (args['extract_pop'] == 2):
        con = psycopg2.connect(**query_args)
        cur = con.cursor()

        pop_size_string = ''
        if args['pop_size'] > 0:
            pop_size_string = 'LIMIT ' + str(args['pop_size'])

        min_age_string = str(args['min_age'])
        min_dur_string = str(args['min_duration'])
        max_dur_string = str(args['max_duration'])
        min_day_string = str(float(args['min_duration'])/24)
        cur.execute('SET search_path to ' + schema_name)
        query = \
        """
        select distinct i.subject_id, i.hadm_id, i.icustay_id,
            i.gender, i.admission_age as age, a.insurance,
            a.deathtime, i.ethnicity, i.admission_type, s.first_careunit,
            CASE when a.deathtime between i.intime and i.outtime THEN 1 ELSE 0 END AS mort_icu,
            CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
            i.hospital_expire_flag,
            i.hospstay_seq, i.los_icu,
            i.admittime, i.dischtime,
            i.intime, i.outtime
          FROM icustay_detail i
          INNER JOIN admissions a ON i.hadm_id = a.hadm_id
          INNER JOIN icustays s ON i.icustay_id = s.icustay_id
          WHERE s.first_careunit NOT like 'NICU'
          and i.hadm_id is not null and i.icustay_id is not null
          and i.hospstay_seq = 1
          and i.icustay_seq = 1
          and i.admission_age >= {min_age}
          and i.los_icu >= {min_day}
          and (i.outtime >= (i.intime + interval '{min_dur} hours'))
          and (i.outtime <= (i.intime + interval '{max_dur} hours'))
          ORDER BY subject_id
          {limit}
          ;
          """.format(limit=pop_size_string, min_age=min_age_string, min_dur=min_dur_string, 
                     max_dur=max_dur_string, min_day=min_day_string)

        data_df = pd.read_sql_query(query, con)
        cur.close()
        con.close()

        data_df = sanitize_df(data_df, static_data_schema)
        data = save_pop(data_df, outPath, static_filename, args['pop_size'], static_data_schema)

    if data is None: print 'SKIPPED static_data'
    else:            print "loaded static_data"

    #############
    # If there is numerics extraction
    X = None
    if (args['extract_numerics'] == 0 | (args['extract_numerics'] == 1) ) & isfile(os.path.join(outPath, dynamic_hd5_filename)):
        X = pd.read_hdf(os.path.join(outPath, dynamic_hd5_filename))
    elif (args['extract_numerics'] == 1 & (not isfile(os.path.join(outPath, dynamic_hd5_filename)))) | (args['extract_numerics'] == 2):
        print "Extracting vitals data..."
        start_time = time.time()

        ########
        # Step 1) Get the set of variables we want for the patients we've identified!
        icuids_to_keep = get_values_by_name_from_df_column_or_index(data, 'icustay_id')
        icuids_to_keep = set([str(s) for s in icuids_to_keep])
        data = data.copy(deep=True).reset_index().set_index('icustay_id')

        # Select out SID, TIME, ITEMID, VALUE form each of the sources!
        var_map = get_variable_mapping(mimic_mapping_filename)
        var_ranges = get_variable_ranges(range_filename)

        chartitems_to_keep = var_map.loc[var_map['LINKSTO'] == 'chartevents'].ITEMID
        chartitems_to_keep = set([ str(i) for i in chartitems_to_keep ])

        labitems_to_keep = var_map.loc[var_map['LINKSTO'] == 'labevents'].ITEMID
        labitems_to_keep = set([ str(i) for i in labitems_to_keep ])

        con = psycopg2.connect(**query_args)
        cur = con.cursor()

        print "  starting db query with %d subjects..." % (len(icuids_to_keep))
        cur.execute('SET search_path to ' + schema_name)
        query = \
        """
        select c.subject_id, i.hadm_id, c.icustay_id, c.charttime, c.itemid, c.value, valueuom
        FROM icustay_detail i
        INNER JOIN chartevents c ON i.icustay_id = c.icustay_id
        where c.icustay_id in ({icuids})
          and c.itemid in ({chitem})
          and c.charttime between intime and outtime
          and c.error is distinct from 1
          and c.valuenum is not null

        UNION ALL

        select distinct i.subject_id, i.hadm_id, i.icustay_id, l.charttime, l.itemid, l.value, valueuom
        FROM icustay_detail i
        INNER JOIN labevents l ON i.hadm_id = l.hadm_id
        where i.icustay_id in ({icuids})
          and l.itemid in ({lbitem})
          and l.charttime between (intime - interval '6' hour) and outtime
          and l.valuenum > 0 -- lab values cannot be 0 and cannot be negative
        ;
        """.format(icuids=','.join(icuids_to_keep), chitem=','.join(chartitems_to_keep), lbitem=','.join(labitems_to_keep))
        X = pd.read_sql_query(query, con)

        itemids = set(X.itemid.astype(str))

        query_d_items = \
        """
        SELECT itemid, label, dbsource, linksto, category, unitname
        FROM d_items
        WHERE itemid in ({itemids})
        ;
        """.format(itemids=','.join(itemids))
        I = pd.read_sql_query(query_d_items, con).set_index('itemid')

        cur.close()
        con.close()
        print "  db query finished after %.3f sec" % (time.time() - start_time)
        X = save_numerics(
            data, X, I, var_map, var_ranges, outPath, dynamic_filename, columns_filename, subjects_filename,
            times_filename, dynamic_hd5_filename, group_by_level2=args['group_by_level2'], apply_var_limit=args['var_limits'],
            min_percent=args['min_percent']
        )

    if X is None: print "SKIPPED vitals_hourly_data"
    else:         print "LOADED vitals_hourly_data"

    #############
    # If there is codes extraction
    C = None
    if ( (args['extract_codes'] == 0) or (args['extract_codes'] == 1) ) and isfile(os.path.join(outPath, codes_hd5_filename)):
        C = pd.read_hdf(os.path.join(outPath, codes_hd5_filename))
    elif ( (args['extract_codes'] == 1) and (not isfile(os.path.join(outPath, codes_hd5_filename))) ) or (args['extract_codes'] == 2):
        hadm_ids_to_keep = get_values_by_name_from_df_column_or_index(data, 'hadm_id')
        hadm_ids_to_keep = set([str(hadm_id) for hadm_id in hadm_ids_to_keep])
        con = psycopg2.connect(**query_args)
        cur = con.cursor()

        cur.execute('SET search_path to ' + schema_name)
        # TODO(mmd): skipping some icd9_codes means that position in the list of icd9_codes doesn't mean the same
        # thing to everyrow.
        query = \
        """
        SELECT
            i.icustay_id, d.subject_id, d.hadm_id,
            array_agg(d.icd9_code ORDER BY seq_num ASC) AS icd9_codes
        FROM mimiciii.diagnoses_icd d INNER JOIN icustays i
        ON i.hadm_id = d.hadm_id AND i.subject_id = d.subject_id
        WHERE d.hadm_id IN ({hadm_ids}) AND seq_num IS NOT NULL
        GROUP BY i.icustay_id, d.subject_id, d.hadm_id
        ;
        """.format(hadm_ids = ','.join(hadm_ids_to_keep))
        codes = pd.read_sql_query(query,con)
        cur.close()
        con.close()

        C = save_icd9_codes(data, codes, outPath, codes_filename, codes_hd5_filename)

    if C is None: print "SKIPPED codes_data"
    else:         print "LOADED codes_data"

    #############
    # If there is outcome extraction
    Y = None
    if ( (args['extract_outcomes'] == 0) | (args['extract_outcomes'] == 1) ) & isfile(os.path.join(outPath, outcome_hd5_filename)):
        Y = pd.read_hdf(os.path.join(outPath, outcome_hd5_filename))
    elif ( (args['extract_outcomes'] == 1) & (not isfile(os.path.join(outPath, outcome_hd5_filename))) ) | (args['extract_outcomes'] == 2):
        Y = save_outcome(
            data, dbname, schema_name, outPath, outcome_filename, outcome_hd5_filename,
            outcome_columns_filename, outcome_data_schema, host=args['psql_host'],
        )


    print(X.shape, X.index.names, X.columns.names)
    print(Y.shape, Y.index.names, Y.columns.names, Y.columns)
    print(C.shape, C.index.names, C.columns.names)
    print(data.shape, data.index.names, data.columns.names)
    if args['exit_after_loading']:
        sys.exit()

    shared_idx = X.index
    shared_sub = list(X.index.get_level_values('icustay_id').unique())
    #X = X.loc[shared_idx]
    # TODO(mmd): Why does this work?
    Y = Y.loc[shared_idx]
    # Problems start here.
    C = C.loc[shared_idx]
    data = data.loc[shared_sub]
    data = data.reset_index().set_index(ID_COLS)

    # Map the lowering function to all column names
    X.columns = pd.MultiIndex.from_tuples(
        [tuple((str(l).lower() for l in cols)) for cols in X.columns], names=X.columns.names
    )
    if args['group_by_level2']:
        var_names = list(X.columns.get_level_values('LEVEL2'))
    else:
        var_names = list(X.columns.get_level_values('itemid'))

    Y.columns = map(lambda x: str(x).lower(), Y.columns)
    out_names = list(Y.columns.values[3:])
    C.columns = map(str.lower, C.columns)
    icd_names = list(C.columns.values[1:])
    data.columns = map(lambda x: str(x).lower(), data.columns)
    static_names = list(data.columns.values[3:])

    print 'Shape of X : ', X.shape
    print 'Shape of Y : ', Y.shape
    print 'Shape of C : ', C.shape
    print 'Shape of static : ', data.shape
    print 'Variable names : ', ",".join(var_names)
    print 'Output names : ', ",".join(out_names)
    print 'Ic_dfD9 names : ', ",".join(icd_names)
    print 'Static data : ', ",".join(static_names)

    X.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'vitals_labs')
    Y.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'interventions')
    C.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'c_df')
    data.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'patients', format='table')
    #fencepost.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'fencepost')

    #############
    #X.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'X')
    #print 'FINISHED VAR LIMITS'

    X_mean = X.iloc[:, X.columns.get_level_values(-1)=='mean']
    X_mean.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'vitals_labs_mean')

    #TODO: Log the variables that are in 0-1 space, like
    #to_log = ['fio2', 'glucose']
    #for feat in to_log:
    #    X[feat] = x[feat].apply(np.log)

    #############
    # Plot the histograms
    if args['plot_hist'] == 1:
        plot_variable_histograms(var_names, X)

    #############
    # Print the total proportions!
    rows, vars = X.shape
    print ''
    for l, vals in X.iteritems():
        ratio = 1.0 * vals.dropna().count() / rows
        print str(l) + ': ' + str(round(ratio, 3)*100) + '% present'

    #############
    # Print the per subject proportions!
    df = X.groupby(['subject_id']).count()
    for k in [1, 2, 3]:
        print '% of subjects had at least ' + str(k) + ' present'
        d = df > k
        d = d.sum(axis=0)
        d = d / len(df)
        d = d.reset_index()
        for index, row in d.iterrows():
            print str(index) + ': ' + str(round(row[0], 3)*100) + '%'
        print '\n'

    print 'Done!'

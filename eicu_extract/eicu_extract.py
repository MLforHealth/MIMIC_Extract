import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_utils.querier import *
from extract_utils.datapackage_io_util import (
    load_datapackage_schema,
    load_sanitized_df_from_csv,
    save_sanitized_df_to_csv,
    sanitize_df,
)

import os, psycopg2, re, sys, time, numpy as np, pandas as pd
from sklearn import metrics
from datetime import datetime
from datetime import timedelta

from os.path import isfile, isdir, splitext
import argparse
import pickle as cPickle
import numpy.random as npr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.path.join(CURRENT_DIR, 'SQL_Queries')
STATICS_QUERY_PATH = os.path.join(SQL_DIR, 'statics.sql')

# Output filenames
joint_hd5_filename = 'all_hourly_data.h5'

# SQL command params
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

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

    return data_df

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

    print("Shape of X : ", X.shape)

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


def save_icd9_codes(codes, outPath, codes_h5_filename):
    codes.set_index(ID_COLS, inplace=True)
    codes.to_hdf(os.path.join(outPath, codes_h5_filename), 'C')
    return codes

def save_outcome(
    data, querier, outPath, outcome_filename, outcome_hd5_filename,
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
    query = """
    select i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
    FROM icustay_detail i
    INNER JOIN ventilation_durations v ON i.icustay_id = v.icustay_id
    where v.icustay_id in ({icuids})
    and v.starttime between intime and outtime
    and v.endtime between intime and outtime;
    """

    old_template_vars = querier.exclusion_criteria_template_vars
    querier.exclusion_criteria_template_vars = dict(icuids=','.join(icuids_to_keep))

    vent_data = querier.query(query_string=query)
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
    table_names = [
        'vasopressor_durations',
        'adenosine_durations',
        'dobutamine_durations',
        'dopamine_durations',
        'epinephrine_durations',
        'isuprel_durations',
        'milrinone_durations',
        'norepinephrine_durations',
        'phenylephrine_durations',
        'vasopressin_durations'
    ]
    column_names = ['vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 
                    'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin']

    # TODO(mmd): This section doesn't work. What is its purpose?
    for t, c in zip(table_names, column_names):
        # TOTAL VASOPRESSOR DATA
        query = """
        select i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
        FROM icustay_detail i
        INNER JOIN {table} v ON i.icustay_id = v.icustay_id
        where v.icustay_id in ({icuids})
        and v.starttime between intime and outtime
        and v.endtime between intime and outtime;
        """
        new_data = querier.query(query_string=query, extra_template_vars=dict(table=t))
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns={'on': c}, inplace=True)
        new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population, in which c was never
        # performed.
        if not c in new_data:
            print("Column ", c, " not in data.")
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
        print('Extracted ' + c + ' from ' + t)


    tasks=["colloid_bolus", "crystalloid_bolus", "nivdurations"]

    for task in tasks:
        if task=='nivdurations':
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.starttime, v.endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.starttime between intime and outtime
            and v.endtime between intime and outtime;
            """
        else:
            query = """
            select i.subject_id, i.hadm_id, v.icustay_id, v.charttime AS starttime, 
                   v.charttime AS endtime
            FROM icustay_detail i
            INNER JOIN {table} v ON i.icustay_id = v.icustay_id
            where v.icustay_id in ({icuids})
            and v.charttime between intime and outtime
            """

        new_data = querier.query(query_string=query, extra_template_vars=dict(table=task))
        if new_data.shape[0] == 0: continue
        new_data = continuous_outcome_processing(new_data, data, icustay_timediff)
        new_data = new_data.apply(add_outcome_indicators)
        new_data.rename(columns = {'on':task}, inplace=True)
        new_data = new_data.reset_index()
        Y = Y.merge(
            new_data[['subject_id', 'hadm_id', 'icustay_id', 'hours_in', task]],
            on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
            how='left'
        )

        # Sort the values
        Y.fillna(0, inplace=True)
        Y[task] = Y[task].astype(int)
        Y = Y.reset_index(drop=True)
        print('Extracted ' + task)


    # TODO: ADD THE RBC/PLT/PLASMA DATA
    # TODO: ADD DIALYSIS DATA
    # TODO: ADD INFECTION DATA
    # TODO: Move queries to files
    querier.exclusion_criteria_template_vars = old_template_vars

    Y = Y.filter(items=['subject_id', 'hadm_id', 'icustay_id', 'hours_in', 'vent'] + column_names + tasks)
    Y.subject_id = Y.subject_id.astype(int)
    Y.icustay_id = Y.icustay_id.astype(int)
    Y.hours_in = Y.hours_in.astype(int)
    Y.vent = Y.vent.astype(int)
    Y.vaso = Y.vaso.astype(int)
    y_id_cols = ID_COLS + ['hours_in']
    Y = Y.sort_values(y_id_cols)
    Y.set_index(y_id_cols, inplace=True)

    print('Shape of Y : ', Y.shape)

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

# Main, where you can call what makes sense.
if __name__ == '__main__':
    print("Running!")
    # Construct the argument parse and parse the arguments
    # TODO(mmd): Write args to file. Accept config files.
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_path', type=str, default= './data', help='Enter the path you want the output')
    ap.add_argument('--queries_path',
        type=str,
        default=os.path.expandvars("$MIMIC_EXTRACT_CODE_DIR/eicu_extract/SQL_Queries/"))

    ap.add_argument('--pop_size', type=int, default=0,
                    help='Size of population to extract')

    ap.add_argument('--psql_host', type=str, default=None,
                    help='Postgres host. Try "/var/run/postgresql/" for Unix domain socket errors.')
    ap.add_argument('--psql_dbname', type=str, default='mimic',
                    help='Postgres database name.')
    ap.add_argument('--psql_schema_name', type=str, default='mimiciii',
                    help='Postgres database name.')
    ap.add_argument('--psql_user', type=str, default=None,
                    help='Postgres user.')
    ap.add_argument('--psql_password', type=str, default=None,
                    help='Postgres password.')

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
        print(key, args[key])

    if not isdir(args['resource_path']):
        raise ValueError("Invalid resource_path: %s" % args['resource_path'])

    # TODO: Skip schemas for now...

    if not isdir(args['out_path']):
        print('ERROR: OUTPATH %s DOES NOT EXIST' % args['out_path'])
        sys.exit()
    else:
        outPath = args['out_path']


    # Modify the filenames
    if args['pop_size'] > 0:
        pop_size = str(args['pop_size'])
        joint_hd5_filename_split = splitext(joint_hd5_filename)
        joint_hd5_filename = f"{joint_hd5_filename_split[0]}_{pop_size}{joint_hd5_filename_split[1]}"

    dbname = args['psql_dbname']
    schema_name = args['psql_schema_name']
    query_args = {'dbname': dbname}
    if args['psql_host'] is not None: query_args['host'] = args['psql_host']
    if args['psql_user'] is not None: query_args['user'] = args['psql_user']
    if args['psql_password'] is not None: query_args['password'] = args['psql_password']

    querier = Querier(query_args=query_args, schema_name=schema_name)

    #############
    # Population extraction

    print("Building data from scratch.")
    pop_size_string = f"LIMIT {args['pop_size']}" if args['pop_size'] > 0 else ""

    min_age_string = str(args['min_age'])
    min_dur_string = str(args['min_duration'])
    max_dur_string = str(args['max_duration'])
    min_day_string = str(float(args['min_duration'])/24)

    template_vars = dict(
        limit=pop_size_string, min_age=min_age_string, min_dur=min_dur_string, max_dur=max_dur_string,
        min_day=min_day_string
    )

    data_df = querier.query(query_file=STATICS_QUERY_PATH, extra_template_vars=template_vars)

    print("Storing data @ %s" % os.path.join(outPath, static_filename))
    data = save_pop(data_df, outPath, static_filename, args['pop_size'], static_data_schema)

    if data is None: print('SKIPPED static_data')
    else:
        # So all subsequent queries will limit to just that already extracted in data_df.
        querier.add_exclusion_criteria_from_df(data, columns=['hadm_id', 'subject_id'])
        print("loaded static_data")

    #############
    # If there is numerics extraction
    X = None
    if (args['extract_numerics'] == 0 | (args['extract_numerics'] == 1) ) & isfile(os.path.join(outPath, dynamic_hd5_filename)):
        print("Reloading X from %s" % os.path.join(outPath, dynamic_hd5_filename))
        X = pd.read_hdf(os.path.join(outPath, dynamic_hd5_filename))
    elif (args['extract_numerics'] == 1 & (not isfile(os.path.join(outPath, dynamic_hd5_filename)))) | (args['extract_numerics'] == 2):
        print("Extracting vitals data...")
        start_time = time.time()

    if X is None: print("SKIPPED vitals_hourly_data")
    else:         print("LOADED vitals_hourly_data")

    #############
    # If there is codes extraction
    C = None
    if ( (args['extract_codes'] == 0) or (args['extract_codes'] == 1) ) and isfile(os.path.join(outPath, codes_hd5_filename)):
        print("Reloading codes from %s" % os.path.join(outPath, codes_hd5_filename))
        C = pd.read_hdf(os.path.join(outPath, codes_hd5_filename))
    elif ( (args['extract_codes'] == 1) and (not isfile(os.path.join(outPath, codes_hd5_filename))) ) or (args['extract_codes'] == 2):
        print("Saving codes...")
        codes = querier.query(query_file=CODES_QUERY_PATH)
        C = save_icd9_codes(codes, outPath, codes_hd5_filename)

    if C is None: print("SKIPPED codes_data")
    else:         print("LOADED codes_data")

    #############
    # If there is notes extraction
    N = None
    if ( (args['extract_notes'] == 0) or (args['extract_codes'] == 1) ) and isfile(os.path.join(outPath, notes_hd5_filename)):
        print("Reloading Notes.")
        N = pd.read_hdf(os.path.join(outPath, notes_hd5_filename))
    elif ( (args['extract_notes'] == 1) and (not isfile(os.path.join(outPath, notes_hd5_filename))) ) or (args['extract_notes'] == 2):
        print("Saving notes...")
        notes = querier.query(query_file=NOTES_QUERY_PATH)
        N = save_notes(notes, outPath, notes_hd5_filename)

    if N is None: print("SKIPPED notes_data")
    else:         print("LOADED notes_data")

    #############
    # If there is outcome extraction
    Y = None
    if ( (args['extract_outcomes'] == 0) | (args['extract_outcomes'] == 1) ) & isfile(os.path.join(outPath, outcome_hd5_filename)):
        print("Reloading outcomes")
        Y = pd.read_hdf(os.path.join(outPath, outcome_hd5_filename))
    elif ( (args['extract_outcomes'] == 1) & (not isfile(os.path.join(outPath, outcome_hd5_filename))) ) | (args['extract_outcomes'] == 2):
        print("Saving Outcomes...")
        Y = save_outcome(
            data, querier, outPath, outcome_filename, outcome_hd5_filename,
            outcome_columns_filename, outcome_data_schema, host=args['psql_host'],
        )


    if X is not None: print("Numerics", X.shape, X.index.names, X.columns.names)
    if Y is not None: print("Outcomes", Y.shape, Y.index.names, Y.columns.names, Y.columns)
    if C is not None: print("Codes", C.shape, C.index.names, C.columns.names)
    if N is not None: print("Notes", N.shape, N.index.names, N.columns.names)

    # TODO(mmd): Do we want to align N like the others? Seems maybe wrong?

    print(data.shape, data.index.names, data.columns.names)
    if args['exit_after_loading']:
        sys.exit()

    shared_idx = X.index
    shared_sub = list(X.index.get_level_values('icustay_id').unique())
    #X = X.loc[shared_idx]
    # TODO(mmd): Why does this work?
    Y = Y.loc[shared_idx]
    # Problems start here.
    if C is not None: C = C.loc[shared_idx]
    data = data[data.index.get_level_values('icustay_id').isin(set(shared_sub))]
    data = data.reset_index().set_index(ID_COLS)

    # Map the lowering function to all column names
    X.columns = pd.MultiIndex.from_tuples(
        [tuple((str(l).lower() for l in cols)) for cols in X.columns], names=X.columns.names
    )
    if args['group_by_level2']:
        var_names = list(X.columns.get_level_values('LEVEL2'))
    else:
        var_names = list(X.columns.get_level_values('itemid'))

    Y.columns = Y.columns.str.lower()
    out_names = list(Y.columns.values[3:])
    if C is not None:
        C.columns = C.columns.str.lower()
        icd_names = list(C.columns.values[1:])
    data.columns = data.columns.str.lower()
    static_names = list(data.columns.values[3:])

    print('Shape of X : ', X.shape)
    print('Shape of Y : ', Y.shape)
    if C is not None: print('Shape of C : ', C.shape)
    print('Shape of static : ', data.shape)
    print('Variable names : ', ",".join(var_names))
    print('Output names : ', ",".join(out_names))
    if C is not None: print('Ic_dfD9 names : ', ",".join(icd_names))
    print('Static data : ', ",".join(static_names))

    X.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'vitals_labs')
    Y.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'interventions')
    if C is not None: C.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'codes')
    data.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'patients', format='table')
    #fencepost.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'fencepost')

    #############
    #X.to_hdf(os.path.join(outPath, dynamic_hd5_filt_filename), 'X')
    #print('FINISHED VAR LIMITS')

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
    print('')
    for l, vals in X.iteritems():
        ratio = 1.0 * vals.dropna().count() / rows
        print(str(l) + ': ' + str(round(ratio, 3)*100) + '% present')

    #############
    # Print the per subject proportions!
    df = X.groupby(['subject_id']).count()
    for k in [1, 2, 3]:
        print('% of subjects had at least ' + str(k) + ' present')
        d = df > k
        d = d.sum(axis=0)
        d = d / len(df)
        d = d.reset_index()
        for index, row in d.iterrows():
            print(str(index) + ': ' + str(round(row[0], 3)*100) + '%')
        print('\n')

    print('Done!')

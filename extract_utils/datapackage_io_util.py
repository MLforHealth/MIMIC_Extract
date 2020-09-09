import numpy as np
import pandas as pd
import datapackage

def load_datapackage_schema(json_fpath, resource_id=0):
    """ Load schema object

    Returns
    -------
    schema : schema object, with attributes
        field_names
        fields : list of dict
            Each dict provides info about the field (data type, etc)
    """
    spec = datapackage.DataPackage(json_fpath)
    schema = spec.resources[resource_id].schema
    return schema

def load_sanitized_df_from_csv(csv_fpath, schema):
    """ Load dataframe from CSV that meets provided schema.

    Returns
    -------
    data_df : pandas DataFrame
        Will have fields provided by schema
        Will have field types (categorical, datetime, etc) provided by schema.
    """
    data_df = pd.read_csv(csv_fpath)
    return sanitize_df(data_df, schema)

def save_sanitized_df_to_csv(csv_fpath, data_df, schema=None):
    """ Save sanitized df to .csv file

    Returns
    -------
    None

    Post Condition
    --------------
    csv_fpath is .csv file containing sanitized data_df
    This file could be read by load_sanitized_df_from_csv()
    """
    if schema is not None:
        data_df = sanitize_df(data_df, schema, setup_index=False)
    has_non_numeric_index = (
        getattr(data_df.index, 'name', None) is not None
        or getattr(data_df.index, 'names', [None])[0] is not None)
    data_df.to_csv(csv_fpath, index=has_non_numeric_index)

def sanitize_df(data_df, schema, setup_index=True, missing_column_procedure='fill_zero'):
    """ Sanitize dataframe according to provided schema

    Returns
    -------
    data_df : pandas DataFrame
        Will have fields provided by schema
        Will have field types (categorical, datetime, etc) provided by schema.
    """
    data_df = data_df.reset_index()
    for ff, field_name in enumerate(schema.field_names):
        type_ff = schema.fields[ff].descriptor['type']
        if field_name not in data_df.columns:
            if missing_column_procedure == 'fill_zero':
                if type_ff == 'integer':
                    data_df[field_name] = 0
                elif type_ff == 'number':
                    data_df[field_name] = 0.0

    # Reorder columns to match schema
    data_df = data_df[schema.field_names]
    # Cast fields to required type (categorical / datetime)
    for ff, name in enumerate(schema.field_names):
        ff_spec = schema.descriptor['fields'][ff]
        if 'pandas_dtype' in ff_spec and ff_spec['pandas_dtype'] == 'category':
            data_df[name] = data_df[name].astype('category')
        elif 'type' in ff_spec and ff_spec['type'] == 'datetime':
            data_df[name] = pd.to_datetime(data_df[name])
    if hasattr(schema, 'primary_key'):
        data_df = data_df.sort_values(schema.primary_key)
        if setup_index:
            data_df = data_df.set_index(schema.primary_key)
    return data_df



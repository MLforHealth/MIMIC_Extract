import inspect

def retrieve_name_in_fn(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    out = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    assert len(out) == 1
    return out[0]

def blind_display(*dfs, blinded=True):
    for df in dfs:
        print(f"{retrieve_name_in_fn(df)}.shape: ", df.shape)
        if blinded:
            display(df.head(0))
        else:
            display(df.head())
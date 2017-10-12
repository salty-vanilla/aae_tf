import pandas as pd


def dump_config(dst_path, config):
    dict_ = vars(config)
    df = pd.DataFrame(list(dict_.items()), columns=['attr', 'status'])
    df.to_csv(dst_path, index=None)

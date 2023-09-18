import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def sku_id2name():
    skus = pd.read_csv('skus.csv')
    return {r['wms_sku_id']: r['name'] for _, r in skus.iterrows()}


def load_data():
    df = pd.read_csv('orders.csv')

    order_vals, order_ids = np.unique(df.order_id, return_inverse=True)
    sku_vals, sku_ids = np.unique(df.wms_sku_id, return_inverse=True)

    data = coo_matrix((np.full(len(order_ids), 1), (order_ids, sku_ids)))
    data = data.tocsr()

    return data, sku_vals


def filter_infrequent_skus(data, min_sku_freq):
    sku_counts = data.sum(0).A.flatten()
    mask = sku_counts > min_sku_freq
    data = data[:, mask]

    non_empty_orders_mask = data.sum(1).A.flatten() > 0
    data = data[non_empty_orders_mask]

    return data, mask


def sample_data(data, size=10):
    idx = np.random.choice(np.arange(data.shape[0]), size)
    return data[idx]

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class Evaluator:
    def __init__(self, cluster_map, sku_vals, sku_id2name, image_path, data):
        self.cluster_map = cluster_map
        self.sku_vals = sku_vals
        self.sku_id2name = sku_id2name
        self.image_path = image_path
        self.data = data

    def print_samples(self, sample_count=7):
        for i in range(self.cluster_map.shape[0]):
            print("CLUSTER:", i)
            skus = np.where(self.cluster_map[i])[0]
            np.random.shuffle(skus)
            for s in skus[:sample_count]:
                print(self.sku_id2name[self.sku_vals[s]])
            print("")

    def image_summary(self, output_folder: Path):
        output_folder.mkdir(exist_ok=True)

        for i in range(self.cluster_map.shape[0]):
            (output_folder / str(i)).mkdir(exist_ok=True)

        for i in range(len(self.sku_vals)):
            cluster_id = np.argmax(self.cluster_map[:, i])
            image_name = str(self.sku_vals[i]) + '.jpg'

            source_image = self.image_path / image_name
            if not source_image.exists():
                continue

            shutil.copy(
                source_image,
                output_folder / str(cluster_id) / image_name
            )

    def clearml_image_summary(self, logger):
        for i in range(len(self.sku_vals)):
            cluster_id = np.argmax(self.cluster_map[:, i])
            image_name = str(self.sku_vals[i]) + '.jpg'
            source_image = self.image_path / image_name
            if not source_image.exists():
                continue

            logger.report_image(
                str(cluster_id),
                str(self.sku_vals[i]),
                iteration=0,
                image=Image.open(source_image)
            )

    def data_summary(self, output_path=None):
        ret = []
        item_sum = self.data.sum(axis=0).A.flatten()

        for i in range(len(self.sku_vals)):
            ret.append({
                'sku': self.sku_vals[i],
                'name': self.sku_id2name[self.sku_vals[i]],
                'cluster': np.argmax(self.cluster_map[:, i]),
                'order_frequency': item_sum[i]
            })
        ret = pd.DataFrame(ret)
        if output_path is not None:
            ret.to_csv(output_path)
        return ret

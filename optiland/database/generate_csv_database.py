"""This script scrapes the refractiveindex.info database, retrieves all data
contained in yaml files, then records relevant material data in a csv file.
The csv file is used directly in Optiland to retrieve material data. See
the Material class in optiland.materials.py for more information.
"""

# pragma: no cover
from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def generate_database(output_file):
    """Scrape catalog-nk.yml and write the resulting glass data to a CSV file."""
    # load the catalog
    filename = "catalog-nk.yml"

    with open(filename, encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    # read data into pandas dataframe
    data = []
    for mat_group in yaml_data:
        group_name = mat_group["SHELF"]
        for full_mat_data in mat_group["content"]:
            try:
                mat_name = full_mat_data["BOOK"]
                mat_name_full = full_mat_data["name"]
                for mat_data in full_mat_data["content"]:
                    if (
                        mat_data.get("PAGE") is None
                        or mat_data.get("name") is None
                        or mat_data.get("data") is None
                    ):
                        continue
                    data.append(
                        [
                            group_name,
                            mat_name,
                            mat_name_full,
                            mat_data["PAGE"],
                            mat_data["name"],
                            mat_data["data"],
                        ],
                    )
            except KeyError:
                pass

    df = pd.DataFrame(
        data,
        columns=[
            "group",
            "category_name",
            "category_name_full",
            "reference",
            "name",
            "filename",
        ],
    )

    # read each file and record wavelength ranges
    for k, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = os.path.join("data-nk", row["filename"])
        with open(filename, encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
            try:
                wave_min = [np.nan]
                wave_max = [np.nan]
                for sub_data in yaml_data["DATA"]:
                    sub_data_type = sub_data["type"]
                    if sub_data_type.startswith("formula "):
                        values = sub_data["wavelength_range"]
                        wave_range = [float(x) for x in values.split()]
                        wave_min.append(wave_range[0])
                        wave_max.append(wave_range[1])

                    elif sub_data_type.startswith("tabulated "):
                        data_file = StringIO(sub_data["data"])
                        data = np.atleast_2d(np.loadtxt(data_file))
                        wavelength = data[:, 0]
                        wave_min.append(np.min(wavelength))
                        wave_max.append(np.max(wavelength))

                    df.loc[k, "min_wavelength"] = np.nanmax(wave_min)
                    df.loc[k, "max_wavelength"] = np.nanmin(wave_max)

            except KeyError:
                pass

            # Add filename without extension
            filename_no_ext = os.path.basename(row["filename"])
            filename_no_ext = os.path.splitext(filename_no_ext)[0]
            df.loc[k, "filename_no_ext"] = filename_no_ext

    # save the database
    df.to_csv(output_file, index=False)

    return output_file

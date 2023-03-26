# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: rad
#     language: python
#     name: python3
# ---

# %%

import os

# %%
import cartopy
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.cloud import bigquery

# %%
sns.set_theme(style="whitegrid")


# %%
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm


# %%
#eliminate messy areas
def messy_areas():
    argentina_chile = -87.98, -57.8, -52.54, -35.05  # argentina and chile
    south_atlantic = -45.38, -58.03, -27.77, -50.4  # southern atlantic island with ice
    se_australia = 136.9, -46.1, 151.4, -33.7  # southern Ausralia
    norway_s = -8.3, 60.6, 16.0, 71.5  # Norway south
    norway_n = -4.96, 65.96, 30.0, 90  # Norway north
    canada_ne = -118.2, 50.8, -48.6, 90  # Hudson
    alaska_n = -178.0, 65.2, -110.9, 90  # Northern alaska
    russia_n = 33.3, 62.4, 179.1, 90  # north russia
    maracaibo = -72.4642, 8.8698, -70.633, 10.6517 #lake maracaibo
    elimination_string = ""
    for region, name in zip(
        [
            argentina_chile,
            south_atlantic,
            se_australia,
            norway_s,
            norway_n,
            canada_ne,
            alaska_n,
            russia_n,
            maracaibo, 
        ],
        [
            "argentina_chile",
            "south_atlantic",
            "se_australia",
            "norway_s",
            "norway_n",
            "canada_ne",
            "alaska_n",
            "russia_n",
            "maracaibo"
        ],
    ):
        min_lon, min_lat, max_lon, max_lat = region
        s = (
            f"and not (detect_lat between {min_lat} and {max_lat} "
            f" and detect_lon between {min_lon} and {max_lon} ) # {name} \n"
        )
        elimination_string += s
    return elimination_string


# %%
def read_sql(filename):
    query = open(filename, "r")
    query_read = query.read()
    query.close()
    return query_read


# %%
def query_to_df(filename):
    query = open(filename, "r")
    query_read = query.read()
    query.close()
    return pd.read_gbq(query_read)


# %%
def create_table(df, full_table_name, table_description, if_exists = 'replace'):

        # Create table
        df.to_gbq(full_table_name, if_exists= if_exists)
        
        client = bigquery.Client()
        
        #get table
        table = client.get_table(full_table_name)

        # Set table description
        table.description = table_description
        
        #update table with description
        client.update_table(table, ["description"]) 


# %%
def query_to_table(query, table_id, max_retries=100, retry_delay=60):
    for _ in range(max_retries):

        config = bigquery.QueryJobConfig(
            destination=table_id, write_disposition="WRITE_TRUNCATE"
        )

        job = client.query(query, job_config=config)

        if job.error_result:
            err = job.error_result["reason"]
            msg = job.error_result["message"]
            if err == "rateLimitExceeded":
                print(f"retrying... {msg}")
                time.sleep(retry_delay)
                continue
            elif err == "notFound":
                print(f"skipping... {msg}")
                return
            else:
                raise RuntimeError(msg)

        job.result()  # wait to complete
        print(f"completed {table_id}")
        return

    raise RuntimeError("max_retries exceeded")


# %%
def execute_commands_in_parallel(commands):
    """This takes a list of commands and runs them in parallel
    Note that this assumes you can run 16 commands in parallel,
    your mileage may vary if your computer is old and slow.
    Requires having gnu parallel installed on your machine.
    """
    with open("commands.txt", "w") as f:
        f.write("\n".join(commands))
    os.system("parallel -j 16 < commands.txt")
    os.system("rm -f commands.txt")


# %%
def plot_raster_and_bivariate_in_bbox(df, extent, infrastructure_points, save = False):
    raster = psm.rasters.df2raster(
        df,
        "lon_index",
        "lat_index",
        "visit_hours_sum",
        xyscale=100,
        per_km2=True)

    raster_total = psm.rasters.df2raster(
        df, 
        "lon_index", 
        "lat_index", 
        "total_hours", 
        xyscale=100, 
        per_km2=True)

    grid_ratio_100 = np.divide(
        raster, raster_total, out=np.zeros_like(raster), where=raster_total != 0)

    fig = plt.figure(figsize=(12, 10))
    norm = mpcolors.LogNorm(vmin=0.001, vmax=raster.max()/4)
    with psm.context(psm.styles.dark):

        proj = psm.find_projection(df.lon_index/100, df.lat_index/100)
        extent = extent
        ax, im = psm.plot_raster(
            raster,
            cmap="presence",
            norm=norm,
            projection = proj.projection
            # zorder=2
        )
        ax.set_extent(extent)
        psm.add_countries()
        psm.add_eezs()

        cbax = psm.add_colorbar(im, label=r"hours per $\mathregular{km^2}$", width=0.5)
        cbax.tick_params(labelsize=16)

        ax.scatter(infrastructure_points.lon, infrastructure_points.lat, color = 'red', transform = psm.identity, alpha = .5)
        if save:
            plt.savefig(f'{b}_oil_vessel_activity.png', bbox_inches="tight", dpi = 300)
        plt.show()


    cmap = psm.cm.bivariate.TransparencyBivariateColormap(psm.cm.bivariate.orange_blue)
    with psm.context(psm.styles.dark):
        fig = plt.figure(figsize=(12, 10))
        ax = psm.create_map(projection=proj.projection)
        psm.add_land(ax)

        norm1 = mpcolors.Normalize(vmin=0.0, vmax=.05, clip=True)
        norm2 = norm
        
        psm.add_bivariate_raster(
        grid_ratio_100, np.clip(raster_total, 0.01, raster_total.max()/4), cmap, norm1, norm2
        )

        cb_ax = psm.add_bivariate_colorbox(
            cmap,
            norm1,
            norm2,
            xlabel="fraction of infra vessel hours",
            ylabel="total hours",
            yformat="{x:.2f}",
            aspect_ratio=2.0,
        )

        ax.set_extent(extent)

        ax.scatter(infrastructure_points.lon, infrastructure_points.lat, color = 'red', transform = psm.identity, alpha = .5)# zorder=1)

        if save:
            plt.savefig(f'{b}_oil_vessels_bivariate', dpi = 300)

        plt.show()

# %%

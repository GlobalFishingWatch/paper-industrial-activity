# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Upload NPP data
#
# This notebook processes NPP data from https://oceandata.sci.gsfc.nasa.gov and uploads the data to BigQuery for other analyses

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import netCDF4
import subprocess
# %matplotlib inline
from datetime import datetime, timedelta 

# +
from google.cloud import bigquery
from google.cloud.exceptions import NotFound



# Construct a BigQuery client object.
client = bigquery.Client()


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


# +
## Download 4km yearh NPP data from here:

# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20020012002365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20030012003365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20040012004366.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20050012005365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20060012006365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20070012007365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20080012008366.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20090012009365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20100012010365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20110012011365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20120012012366.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20130012013365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20140012014365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20150012015365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20160012016366.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20170012017365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20180012018365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20190012019365.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20200012020366.L3m_YR_CHL_chlor_a_4km.nc
# https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A20210012021365.L3m_YR_CHL_chlor_a_4km.nc

### https://oceancolor.gsfc.nasa.gov/
# -

# !ls chl_data

files = '''A20120012012366.L3m_YR_CHL_chlor_a_4km.nc
A20130012013365.L3m_YR_CHL_chlor_a_4km.nc
A20140012014365.L3m_YR_CHL_chlor_a_4km.nc
A20150012015365.L3m_YR_CHL_chlor_a_4km.nc
A20160012016366.L3m_YR_CHL_chlor_a_4km.nc
A20170012017365.L3m_YR_CHL_chlor_a_4km.nc
A20180012018365.L3m_YR_CHL_chlor_a_4km.nc
A20190012019365.L3m_YR_CHL_chlor_a_4km.nc
A20200012020366.L3m_YR_CHL_chlor_a_4km.nc
A20210012021365.L3m_YR_CHL_chlor_a_4km.nc'''.split("\n")


def upload_file(filename):
    year = filename[1:5]
    nc = netCDF4.Dataset(f'chl_data/{filename}')
    chla = nc['chlor_a']
    for i, row in enumerate(chla):
        raster[-i,:] = row.data # negative because 0 starts at bottom
    plt.imshow(np.flipud(raster))
    plt.title(year)
    plt.show()
    
    chl = raster.flatten()
    lat = np.arange(len(nc['lat']))
    lon = np.arange(len(nc['lon']))
    lons, lats = np.meshgrid(lon, lat)
    lons = lons.flatten()
    lats = lats.flatten()
    
    # Create DataFrame 
    data = {"lon":lons,"lat":lats,'chl':chl}
    df = pd.DataFrame(data) 
    df = df[df.chl>=0]
    df.to_csv('temp.csv',index=False)
    
    command = f'''bq load  --skip_leading_rows=1 scratch_david.npp_test_{year} temp.csv \
lon:integer,lat:integer,chl:float'''.format(YYYYMMDD=YYYYMMDD)
    subprocess.run(command.split())
    
    subprocess.run("rm -f temp.csv".split())


90-1/24/2

len(nc['lat'])

nc['lat'][0]

lat = np.arange(len(nc['lat']))
lon = np.arange(len(nc['lon']))

for filename in files:
    upload_file(filename)

nc

# +
q = '''select _table_suffix as year,
lon, lat, chl from `scratch_david.npp_test_*` '''

query_to_table(q, 
    'project-id.proj_global_sar.chlorophyll_24thdegree_yearly')

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import module_name_b
from module_name_a.utils import datetime

# Use _reload() while in development so that changes
# to your module code are recognized.
module_name_b._reload()

# %%
help(datetime.as_date_str)
# %%

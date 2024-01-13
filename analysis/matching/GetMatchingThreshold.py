# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # The idea is that the number of matches should equal the number of expected vessels in a scene
#
# We will select a score that gets this threshold

import pandas as pd
import matplotlib.pyplot as plt
import proplot
import numpy as np

# +
q = ''' select
round(sum(prob_inside*recall)) expected_matches
 from proj_global_sar.expected_recall
where date(_partitiontime) between "2017-01-01" and "2021-12-31" '''

expected_matches = pd.read_gbq(q)['expected_matches'].values[0]
expected_matches
# -

# # We expect 16.98 million vessels to appear in the scenes

np.arange(-6,1,.05)

# +
q = '''

select 
  sum(if(score_mult_recall_length > pow(10, threshold),1,0)) matches,
  pow(10, threshold) as threshold
from 
  proj_global_sar.detections_w_overpasses_v20230215
cross join unnest([-6.00000000e+00, -5.95000000e+00, -5.90000000e+00, -5.85000000e+00,
       -5.80000000e+00, -5.75000000e+00, -5.70000000e+00, -5.65000000e+00,
       -5.60000000e+00, -5.55000000e+00, -5.50000000e+00, -5.45000000e+00,
       -5.40000000e+00, -5.35000000e+00, -5.30000000e+00, -5.25000000e+00,
       -5.20000000e+00, -5.15000000e+00, -5.10000000e+00, -5.05000000e+00,
       -5.00000000e+00, -4.95000000e+00, -4.90000000e+00, -4.85000000e+00,
       -4.80000000e+00, -4.75000000e+00, -4.70000000e+00, -4.65000000e+00,
       -4.60000000e+00, -4.55000000e+00, -4.50000000e+00, -4.45000000e+00,
       -4.40000000e+00, -4.35000000e+00, -4.30000000e+00, -4.25000000e+00,
       -4.20000000e+00, -4.15000000e+00, -4.10000000e+00, -4.05000000e+00,
       -4.00000000e+00, -3.95000000e+00, -3.90000000e+00, -3.85000000e+00,
       -3.80000000e+00, -3.75000000e+00, -3.70000000e+00, -3.65000000e+00,
       -3.60000000e+00, -3.55000000e+00, -3.50000000e+00, -3.45000000e+00,
       -3.40000000e+00, -3.35000000e+00, -3.30000000e+00, -3.25000000e+00,
       -3.20000000e+00, -3.15000000e+00, -3.10000000e+00, -3.05000000e+00,
       -3.00000000e+00, -2.95000000e+00, -2.90000000e+00, -2.85000000e+00,
       -2.80000000e+00, -2.75000000e+00, -2.70000000e+00, -2.65000000e+00,
       -2.60000000e+00, -2.55000000e+00, -2.50000000e+00, -2.45000000e+00,
       -2.40000000e+00, -2.35000000e+00, -2.30000000e+00, -2.25000000e+00,
       -2.20000000e+00, -2.15000000e+00, -2.10000000e+00, -2.05000000e+00,
       -2.00000000e+00, -1.95000000e+00, -1.90000000e+00, -1.85000000e+00,
       -1.80000000e+00, -1.75000000e+00, -1.70000000e+00, -1.65000000e+00,
       -1.60000000e+00, -1.55000000e+00, -1.50000000e+00, -1.45000000e+00,
       -1.40000000e+00, -1.35000000e+00, -1.30000000e+00, -1.25000000e+00,
       -1.20000000e+00, -1.15000000e+00, -1.10000000e+00, -1.05000000e+00,
       -1.00000000e+00, -9.50000000e-01, -9.00000000e-01, -8.50000000e-01,
       -8.00000000e-01, -7.50000000e-01, -7.00000000e-01, -6.50000000e-01,
       -6.00000000e-01, -5.50000000e-01, -5.00000000e-01, -4.50000000e-01,
       -4.00000000e-01, -3.50000000e-01, -3.00000000e-01, -2.50000000e-01,
       -2.00000000e-01, -1.50000000e-01, -1.00000000e-01, -5.00000000e-02,
       -2.13162821e-14,  5.00000000e-02,  1.00000000e-01,  1.50000000e-01,
        2.00000000e-01,  2.50000000e-01,  3.00000000e-01,  3.50000000e-01,
        4.00000000e-01,  4.50000000e-01,  5.00000000e-01,  5.50000000e-01,
        6.00000000e-01,  6.50000000e-01,  7.00000000e-01,  7.50000000e-01,
        8.00000000e-01,  8.50000000e-01,  9.00000000e-01,  9.50000000e-01]) as threshold
      
        group by threshold
        order by threshold
        

'''

df = pd.read_gbq(q)
# -

df.head(20)

# +
# 7e-6 is close... can we get closer?
# -

q = '''

select sum(if(score_mult_recall_length > threshold,1,0)) matches,
threshold
from proj_global_sar.detections_w_overpasses_v20230215
cross join unnest([ 7.1e-6, 7.2e-6, 7.3e-6, 7.4e-6, 7.5e-6, 7.6e-6, 7.7e-6, 7.8e-6, 7.9e-6]) as threshold
      
        group by threshold
        order by threshold
        

'''
df = pd.read_gbq(q)


for index, row in df.iterrows():
    print(row.matches,row.matches/expected_matches, row.threshold)

# # 7.4e-6 for score_mult_recall_length

# ## how does it do at estimating number of vessels in scenes?

q = '''
select avg( safe_divide(expected_matches - matches,expected_matches )) frac_diff,
floor(frac_dark*10)/10 as frac_dark,

count(*) num
from

(select * from 
(select scene_id,
 round(sum(prob_inside*recall)) expected_matches
 from proj_global_sar.expected_recall
where date(_partitiontime) between "2017-01-01" and "2017-12-31"
group by scene_id
having expected_matches > 10


--  6e-4 mult_recall
--  7.4e-6 mult_recall_length
)
join
(select scene_id, sum(if(score > 7.4e-6,1,0)) matches,
sum(if(score< 7.4e-6,1,0))/count(*) frac_dark
from 
proj_global_sar.matched_mult_recall_length
-- proj_global_sar.matched_ave
where date(_partitiontime) between "2017-01-01" and "2017-12-31"
group by scene_id)
using(scene_id))

group by frac_dark order by frac_dark'''
pd.read_gbq(q)

# # for scenes with lots of dark vessels, this method over-matches by 2-4%
# # for scenes with no dark vessels, it underestimates matches by 2-4%
#
#





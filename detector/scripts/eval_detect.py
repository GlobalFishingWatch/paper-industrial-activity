"""
This script evaluates how well the matching script does
at identifying vessels in the SAR data

"""
import json
import sys
from textwrap import dedent
from pathlib import Path
from datetime import datetime
from subprocess import PIPE, Popen

import yaml

from cloud import table_exists
from utils import date_range


# FIXME: Use separate variables: project_id, dataset, table_id
def make_partition_table(table):
    """Make partition table (by day).

    table : project_id.dataset.table_id
    """
    table = replace_period_with_colon(table)
    cmd = f"bq mk --time_partitioning_type=DAY {table}"
    Popen(cmd, shell=True).wait()


def get_eval_table_name(dataset, detect_table):
    return f"{dataset}.matches_{detect_table}_4evaluated"


def replace_period_with_colon(table):
    """Convert period between project and dataset to colon."""
    t = table.split(".")
    if len(t) == 2:
        return table
    if len(t) == 3:
        return f"{t[0]}:{t[1]}.{t[2]}"


def get_query_cmd(matches_evaluated_YMD, query="query"):
    return f"""
        bq {query} --replace --max_rows=1  \
        --destination_table={matches_evaluated_YMD}   \
        --allow_large_results --use_legacy_sql=false
        """


def get_query_sql(date, project_id, dataset, vi_table, detect_table):
    return f"""
    CREATE TEMP FUNCTION
      the_DATE() AS ( DATE(TIMESTAMP("{date:%Y-%m-%d}")) );

    WITH
      interpolated AS (
      SELECT
         if(length(id)=92,SUBSTR(id, 26,67),id) as scene_id,
        ssvid,
        _partitiontime AS thedate,
      IF
        (timeto>timeto2,
          timeto2,
          timeto) AS seconds_to_nearest_ping,
        -- this is more complicated than need be because it can be useful
        -- to identify east versus north component of speed to troubleshoot
        -- the doppler shift effect
        (a.lat-a.lat2)*111/(timeto + timeto2)*60*60*1.852 n_knots,
        (a.lon-a.lon2)*111*COS(a.lat*3.1416/180)/(timeto + timeto2)*60*60*1.582 e_knots
      FROM
        `{project_id}.{dataset}.sentinel_1_ais_interpolated_positions_1km` a
      JOIN
        `{project_id}.pipe_static.distance_from_shore` b
      ON
        CAST( (a.lat*100) AS int64) = CAST( (b.lat*100) AS int64)
        AND CAST((a.lon*100) AS int64) =CAST(b.lon*100 AS int64)
      WHERE
        within_footprint_5km_in
        AND id IN (
        SELECT
          scene_id
        FROM
          `{dataset}.matches_{detect_table}_3top`)
        AND b.distance_from_shore_m > 1852*5 -- more than 5 nm from shore
        AND DATE(a._partitiontime) = the_DATE() ),
      vessel_size AS (
      SELECT
        ssvid,
        best.best_length_m
      FROM
        `{vi_table}`
      WHERE
        best.best_length_m IS NOT NULL
        AND best.best_vessel_class != "gear"),
      in_scenes AS (
      SELECT
        DISTINCT scene_id,
        ssvid,
        _partitiontime thedate
      FROM
        `{dataset}.matches_{detect_table}_1scored_ais`
      WHERE
        ssvid IS NOT NULL
        AND score > .001
        AND DATE(_partitiontime) = the_DATE() )

    SELECT
      FLOOR(best_length_m) length_m,
      subSTRING(a.scene_id,
        1,
        3) AS sat,
      thedate,
      SUM(
      IF
        (c.ssvid IS NOT NULL,
          1,
          0)) AS matched,
      AVG(POW(e_knots*e_knots + n_knots*n_knots, .5)) avg_speed,
      COUNT(*) total,
      ROUND(SUM(
        IF
          (c.ssvid IS NOT NULL,
            1,
            0))/COUNT(*),2) AS frac_matched

    FROM
      interpolated a
    LEFT JOIN
      vessel_size b
    USING
      (ssvid)
    LEFT JOIN
      in_scenes c
    USING
      (ssvid,
        scene_id,
        thedate)
    WHERE
      seconds_to_nearest_ping < 60*10
      -- and pow(e_knots*e_knots + n_knots*n_knots, .5) between 5 and 12
    GROUP BY
      length_m,
      thedate,
      sat
    ORDER BY
      length_m,
      sat
    """


def main(params):
    date = params["date"]
    project_id = params["project_id"]
    dataset = params["dataset"]
    detect_table = params["subbucket"]
    vi_table = params["vi_table"]

    date = datetime.strptime(date, "%Y-%m-%d").date()

    query = get_query_sql(
        date, project_id, dataset, vi_table, detect_table
    )
    matches_evaluated = get_eval_table_name(dataset, detect_table)

    if not table_exists(matches_evaluated):
        make_partition_table(matches_evaluated)

    matches_evaluated_YMD = f"{matches_evaluated}${date:%Y%m%d}"
    cmd = get_query_cmd(matches_evaluated_YMD)
    child_proccess = Popen(cmd.split(), stdin=PIPE, stdout=PIPE)
    out = child_proccess.communicate(bytes(query, "utf-8"))[0]
    print(out)


def help():
    usage = """
    Usage (2 options):

    # All params in a single dir
    evaluate.py [num_cpus] PARAMS_*.json

    # Params in multiple dirs/VMs (use single yaml)
    evaluate.py match.yaml
    """
    print(dedent(usage))
    sys.exit()


if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) == 0 or args[0] in ["-h", "--help"]:
        help()

    elif Path(args[0]).suffix in [".yml", ".yaml"]:
        # Read from yaml
        with open(args[0]) as f:
            p = yaml.load(f, Loader=yaml.FullLoader)
            num_cpus = p["num_cpus"]
            params_list = []
            # Add date to params -> list of params
            for date in date_range(*p['date_range']):
                params = p.copy()
                params["date"] = date
                params_list.append(params)

    elif Path(args[0]).suffix in [".json"]:
        # Read from json (PARAMS_*)
        num_cpus = int(args.pop(0)) if args[0].isnumeric() else 1
        params_list = [json.loads(Path(f).read_text()) for f in args]

    if num_cpus > 1:
        import ray

        ray.init(num_cpus=num_cpus)

        @ray.remote
        def _main(p):
            return main(p)

        futures = [_main.remote(p) for p in params_list]
        ray.get(futures)

    else:
        [main(p) for p in params_list]

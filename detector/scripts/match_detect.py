"""
Matches one day of SAR detections to AIS and VMS.


# TODO: Try using Python bindings instead of system calls

from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('../../sentinel-2/assets'))

template = env.get_template('messages_by_footprint.j2.sql')

query = template.render(
    source = 'proj_sentinel2.ais_interpolated_v20200423',
    max_scenes = 10000,
    min_cnt = 10,
    max_cnt = 1000,
    min_km_from_port = 10
)

# TODO: For match_day, I think the jinja2 stuff will work here.
I should take a look at the bq API. I’ve used it before, but not
recently but that is likely the cleaner approach then either system
calls or Pandas. Only other comment is main() is really long,
than should probably be broken down into smaller chunks.

"""

import json
import sys
from textwrap import dedent
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import yaml

from cloud import table_exists
from utils import date_range


def get_table_names(
    date, project_id, detect_dataset, detect_table, vms_tables
):
    """Define all BQ table names needed."""

    full_detect_table = (
        f"{project_id}.{detect_dataset}.{detect_table}"
    )

    matches_scored_ais_table = (
        f"{project_id}.{detect_dataset}."
        f"matches_{detect_table}_1scored_ais"
    )
    matches_scored_vms_tables = []

    for v in vms_tables:
        v = v.replace("pipe_", "").replace("_production", "")
        table = (
            f"{project_id}.{detect_dataset}"
            f".matches_{detect_table}_1scored_{v}"
        )
        matches_scored_vms_tables.append(table)

    matches_ranked_table = (
        f"{project_id}.{detect_dataset}.matches_{detect_table}_2ranked"
    )
    matches_top_table = (
        f"{project_id}.{detect_dataset}.matches_{detect_table}_3top"
    )
    s1_positions_table = (
            f"{project_id}.satellite_positions_v20190208."
            f"sentinel_1_positions{date:%Y%m%d}"
            )
    return (
        full_detect_table,
        matches_scored_ais_table,
        matches_scored_vms_tables,
        matches_ranked_table,
        matches_top_table,
        s1_positions_table,
    )


# FIXME: Use separate variables: project_id, dataset, table_id
def make_partition_table(table):
    """Make partition table (by day).

    table : project_id.dataset.table_id
    """
    table = replace_period_with_colon(table)
    cmd = f"bq mk --time_partitioning_type=DAY {table}"
    Popen(cmd, shell=True).wait()


def replace_period_with_colon(table):
    """Convert period between project and dataset to colon."""
    t = table.split(".")
    if len(t) == 2:
        return table
    if len(t) == 3:
        return f"{t[0]}:{t[1]}.{t[2]}"


# FIXME: Make this a separate/external process?
def update_s1_positions(date):
    """Update positions for Sentinel-1 if they don't exist.

    This function calls get_positions.py, which will update
    positions for Sentinel-1 if they don't exist for the respective day.
    """
    cmd = f"""python get_positions.py {date:%Y-%m-%d}"""
    Popen(cmd, shell=True).wait()


def get_score_ais_cmd(
    date, matches_scored_ais_table, full_detect_table, asset_dir
):
    matches_scored_YMD = matches_scored_ais_table + f"\${date:%Y%m%d}"
    matches_scored_YMD = replace_period_with_colon(matches_scored_YMD)

    ais_scoring_cmd = f"""
        jinja2  {asset_dir}/score_ais_match.sql.j2 \
        -D YYYY_MM_DD="{date:%Y-%m-%d}" \
        -D detect_table="{full_detect_table}" \
        | \
        bq query --replace  --allow_large_results --use_legacy_sql=false  \
        --destination_table={matches_scored_YMD}\
        """
    return ais_scoring_cmd


def get_score_vms_cmd(
    date,
    vms_table,
    matches_scored_vms_table,
    full_detect_table,
    asset_dir,
):
    matches_scored_YMD = matches_scored_vms_table + f"\${date:%Y%m%d}"
    matches_scored_YMD = replace_period_with_colon(matches_scored_YMD)

    vms_scoring_cmd = f"""
        jinja2  {asset_dir}/score_vms_match.sql.j2   \
        -D YYYY_MM_DD="{date:%Y-%m-%d}" \
        -D detect_table="{full_detect_table}"  \
        -D vms_dataset='{vms_table}' \
        | \
        bq query --replace \
        --destination_table={matches_scored_YMD}\
        --allow_large_results --use_legacy_sql=false
        """
    return vms_scoring_cmd


def get_matches_ranked_table_cmd(
    asset_dir,
    matches_scored_ais_table_YMD,
    matches_scored_vms_tables_YMD,
    matches_ranked_table_YMD,
):
    return f"""
        jinja2 {asset_dir}/ranked_matches.sql.j2 \
        -D matches_scored_ais_table="{matches_scored_ais_table_YMD}" \
        -D matches_scored_vms_tables="{matches_scored_vms_tables_YMD}" \
        | \
        bq query --replace \
        --destination_table={matches_ranked_table_YMD} \
        --allow_large_results --use_legacy_sql=false
        """


def get_matched_table_cmd(
    asset_dir,
    today_detect_table,
    matches_ranked_table_YMD,
    matched_table_YMD,
):
    return f"""
        jinja2 {asset_dir}/matched.sql.j2 \
        -D detect_table="{today_detect_table}" \
        -D matches_ranked_table="{matches_ranked_table_YMD}" \
        | \
        bq query --replace \
        --destination_table={matched_table_YMD} \
        --allow_large_results \
        --use_legacy_sql=false
        """


def get_ais_table_sql(date, matches_scored_ais_table):
    return f"""
        (select 'AIS' as source, * \
        from {matches_scored_ais_table} \
        where _partitiontime = timestamp('{date:%Y-%m-%d}'))
        """


def get_vms_tables_sql(date, matches_scored_vms_tables):
    tables = ""
    for table in matches_scored_vms_tables:
        tables += f"""
            union all \
            select concat(source,ssvid) as ssvid_source,*  \
            from {table} where date(_partitiontime) = '{date:%Y-%m-%d}' \
            and score > .00001
            """
    return tables


def get_ranked_table_sql(date, matches_ranked_table):
    return f"""
        (select * \
        from {matches_ranked_table} \
        where _partitiontime = timestamp('{date:%Y-%m-%d}'))
        """


def main(params):

    date = params["date"]
    vms_tables = params["vms_tables"]
    project_id = params["project_id"]
    detect_dataset = params["dataset"]
    detect_table = params["subbucket"]
    asset_dir = params["asset_dir"]

    date = datetime.strptime(date, "%Y-%m-%d").date()

    # Define names of BQ tables

    (
        full_detect_table,
        matches_scored_ais_table,
        matches_scored_vms_tables,
        matches_ranked_table,
        matches_top_table,
        s1_positions_table,
    ) = get_table_names(
        date, project_id, detect_dataset, detect_table, vms_tables
    )

    if not table_exists(s1_positions_table):
        print(f"Udating Sentinel-1 positions for {date:%Y-%m-%d} ...")
        update_s1_positions(date)

    for table in [
        matches_scored_ais_table,
        matches_ranked_table,
        matches_top_table,
    ] + matches_scored_vms_tables:

        if not table_exists(table):
            make_partition_table(table)

    # Get BQ commands using jinja2

    cmds = []

    cmds.append(
        get_score_ais_cmd(
            date,
            matches_scored_ais_table,
            full_detect_table,
            asset_dir,
        )
    )

    for vms_table, matches_scored_vms_table in zip(
        vms_tables, matches_scored_vms_tables
    ):
        cmd = get_score_vms_cmd(
            date,
            vms_table,
            matches_scored_vms_table,
            full_detect_table,
            asset_dir,
        )
        cmds.append(cmd)

    processes = [Popen(cmd, shell=True) for cmd in cmds]
    [p.wait() for p in processes]

    # Now create ranked tables with SQL

    matches_scored_ais_table_YMD = get_ais_table_sql(
        date, matches_scored_ais_table
    )
    matches_scored_vms_tables_YMD = get_vms_tables_sql(
        date, matches_scored_vms_tables
    )
    print(matches_scored_vms_tables_YMD)

    matches_ranked_table_YMD = replace_period_with_colon(
        matches_ranked_table + f"\${date:%Y%m%d}"
    )
    cmd = get_matches_ranked_table_cmd(
        asset_dir,
        matches_scored_ais_table_YMD,
        matches_scored_vms_tables_YMD,
        matches_ranked_table_YMD,
    )
    Popen(cmd, shell=True).wait()  # FIXME

    # Now Get Top Matches

    matches_ranked_table_YMD = get_ranked_table_sql(date, matches_ranked_table)

    matched_table_YMD = replace_period_with_colon(
        matches_top_table + f"\\${date:%Y%m%d}"
    )

    today_detect_table = full_detect_table + f"{date:%Y%m%d}"

    cmd = get_matched_table_cmd(
        asset_dir,
        today_detect_table,
        matches_ranked_table_YMD,
        matched_table_YMD,
    )
    Popen(cmd, shell=True).wait()  # FIXME


def help():
    usage = """
    Usage (2 options):

    # All params in a single dir
    match.py [num_cpus] PARAMS_*.json

    # Params in multiple dirs/VMs (use single yaml)
    match.py match.yaml
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

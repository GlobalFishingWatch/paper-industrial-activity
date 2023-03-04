'''
Utilities for use with bigquery
'''
import json
from pathlib import Path
from typing import ParamSpecKwargs

from google.cloud import bigquery
from jinja2 import Template
from module_name_a.utils.datetime import as_date_str


def format_query(template_file: Path, **params: ParamSpecKwargs) -> str:
    '''
    Format a jinja2 templated query with the given params.

    You may have extra params which are not used by the template, but all params
    required by the template must be provided.

    Args:
        template_file: path to query template
        params: key values for template population

    Returns:
        populated query statement

    '''

    # Open SQL query with jinja2
    with template_file.open() as f:
        sql_template: Template = Template(f.read())

    # Format query with jinja2 params
    formatted_template = sql_template.render(params)
    return formatted_template


def load_schema(schema_file):
    with open(schema_file) as file:
        return json.load(file)


class BigqueryHelper:
    def __init__(self):
        self.client = bigquery.Client()

    def format_query(self, template_file, **params):
        return format_query(template_file, **params)

    def create_table(
        self,
        full_table_name,
        schema_file,
        table_description,
        partition_field=None,
        exists_ok=True,
    ):
        '''
        Create a BigQuery table.

        Args:
            full_table_name: full_table_name: fully qualified table name
                'project_id.dataset.table'
            schema_file: path to schema json file
            table_description: text to include in the table's description field
            partition_field: name of field to use for time partitioning.
                Defaults to None.
            exists_ok: ignore “already exists” errors when creating the table.
                Defaults to True.

        Returns:
            A new google.cloud.bigquery.table.Table
        '''

        # load schema
        schema = load_schema(schema_file)

        # Create BQ table object
        bq_table = bigquery.Table(full_table_name, schema=schema)

        # Set table description
        bq_table.description = table_description

        # Set partitioning
        if partition_field:
            bq_table.time_partitioning = bigquery.TimePartitioning(
                field=partition_field
            )

        # Create table
        return self.client.create_table(bq_table, exists_ok=exists_ok)

    def clear_table_partition(self, full_table_name, partition_field, partition_date):
        '''
        Clear data from a specific table partition

        Args:
            full_table_name: name of table to query
            partition_field: name of date field to compare against
            partition_date: partition_date may be a datetime, date or a string
                formatted 'YYYY-MM-DD'

        Raises:
            RuntimeError: BigQuery Error dump
        '''

        partition_date = as_date_str(partition_date)

        # Clear data for query date before running query
        d_q = (
            f'DELETE FROM `{full_table_name}` WHERE DATE({partition_field}) '
            f"= '{partition_date}'"
        )

        config = bigquery.QueryJobConfig(priority='BATCH')
        job = self.client.query(d_q, job_config=config)

        if job.error_result:
            err = job.error_result['reason']
            msg = job.error_result['message']
            raise RuntimeError(f'{err}: {msg}')
        else:
            job.result()

    def run_query(self, query, dest_table=None):

        if dest_table:
            config = bigquery.QueryJobConfig(
                destination=dest_table,
                priority='BATCH',
                write_disposition='WRITE_APPEND',
            )
        else:
            config = bigquery.QueryJobConfig(
                priority='INTERACTIVE',
            )
        job = self.client.query(query, job_config=config)

        if job.error_result:
            err = job.error_result['reason']
            msg = job.error_result['message']
            raise RuntimeError(f'{err}: {msg}')
        else:
            return job.result()

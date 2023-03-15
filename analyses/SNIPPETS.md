# BigQuery Python API
---
## Using `jinja2` for queries

<details>
<summary>Formatting in `jinja2`</summary>
<br>

```jinja
SELECT *
FROM {{ pipeline_dataset }}.research_positions
WHERE DATE(timestamp) BETWEEN {{ start_date }} AND {{ end_date }}
AND hours >= {{ hours_threshold }}
```

A jinja-formatted SQL query can be placed inline in python code or in a separate file with the file ending of `.sql.j2` (encouraged for long queries).

Visit the [Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/) to learn more cool things you can do with Jinja templates. And check out the [AIS activity query](https://github.com/GlobalFishingWatch/vessel-info/blob/master/vessel_info_pipe/vi_ais_activity_identity_template.sql.j2) for an example of using logic blocks to swap out SQL code blocks based on an argument value.
</details>
<br>

<details>
<summary>Using a `jinja2` template</summary>
<br>

```python
from jinja2 import Template
with open('query.sql.j2') as f:
    query_template = Template(f.read())

# Format the query with the desired argument values
q = sql_template.render(
        pipeline_dataset="pipe_production_v20201001",
        start_date="2020-01-01",
        end_date="2020-01-31",
        hours_threshold=24,
)
```

While developing a query, it is often helpful to print the query out calling `render()` and then paste it into the BigQuery console to check it's validity and run size.
</details>

## Creating tables and uploading data

<details>
<summary>From a query</summary>
<br>

```python
from google.cloud import bigquery

config = bigquery.QueryJobConfig(
    destination="project.dataset.table-name",
    # Will overwrite any existing table.
    # Use "WRITE_APPEND" to add data to an existing table
    write_disposition="WRITE_TRUNCATE",
)

client = bigquery.Client()
job = client.query(q, job_config=config)
```

This form of table creation is meant for temporary or development tables as it does not readily support schemas or table descriptions. BQ schema/types will be inferred from dataframe.

For more details on parameters, see the [QueryJobConfig documentation](https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.QueryJobConfig).
</details>
<br>

<details>
<summary>From a dataframe</summary>
<br>

```python
from google.cloud import bigquery

table_id = "project.dataset.table-name"
config = bigquery.LoadJobConfig(
    # Will overwrite any existing table.
    # Use "WRITE_APPEND" to add data to an existing table
    write_disposition="WRITE_TRUNCATE",
)

client = bigquery.Client()
client.load_table_from_dataframe(
    df, table_id, job_config=config
)
```

This form of table creation is meant for temporary or development tables as it does not readily support schemas or table descriptions. BQ schema/types will be inferred from dataframe.

For more details on parameters, see the [LoadJobConfig documentation](https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.LoadJobConfig).
</details>
<br>

<details>
<summary>Advanced table creation and population</summary>
<br>

When generating persistent tables, first create the table with all relevant configurations (schema, table description, partitioning, and clustering) before populating it with data.

```python
from google.cloud import bigquery

# Create table with a schema
bq_table = bigquery.Table(reception_tbl,
                          schema=schema)

# Set table description
bq_table.description = "Table Description"

# Optional time-partitioning
bq_table.time_partitioning = bigquery.TimePartitioning(field="timestamp")

# Optional clustering
bq_table.clustering_fields = ['ssvid']

# Create table
client = bigquery.Client()
client.create_table(bq_table, exists_ok=True)

# Add data in "WRITE_APPEND" mode
config = bigquery.QueryJobConfig(
    destination="project.dataset.table-name",
    write_disposition="WRITE_APPEND"
)

job = client.query(q, job_config=config)
```

For example utility functions that can be incorporated into your code, see [this file](https://github.com/GlobalFishingWatch/pipe-python-prototype-template/blob/master/util/bigquery.py).
</details>


## Specifying a table schema

<details>
<summary>Inline in Python</summary>
<br>

```python
schema = [
    bigquery.SchemaField("shipname",
                         type="STRING",
                         mode="REQUIRED",
                         description="Shipname for the vessel"),
    bigquery.SchemaField("imo",
                         type="STRING",
                         mode="NULLABLE",
                         description="IMO number of vessel"),
    bigquery.SchemaField("start_date",
                         type="STRING",
                         mode="NULLABLE",
                         description="First date of vessel activity period"),
    bigquery.SchemaField("end_date",
                         type="DATE",
                         mode="REQUIRED",
                         description="Last date of vessel activity period"),
    bigquery.SchemaField("msg_count",
                         type="INTEGER",
                         mode="REQUIRED",
                         description="Number of AIS messages in activity period"),
]

client = bigquery.Client()
table = bigquery.Table(table_id, schema=schema)
table = client.create_table(table)
```
</details>
<br>

<details>
<summary>From a JSON file</summary>
<br>

```python
import json
from google.cloud import bigquery
with open(f"schema.json") as file:
    self.schema = json.load(file)

client = bigquery.Client()
table = bigquery.Table(table_id, schema=schema)
table = client.create_table(table)
```

Where the JSON file looks like:

```json
[
    {
        name="shipname",
        type="STRING",
        mode="REQUIRED",
        description="Shipname for the vessel"
    },
    {
        name="imo",
        type="STRING",
        mode="NULLABLE",
        description="IMO number of vessel"
    },
    {
        name="start_date",
        type="STRING",
        mode="NULLABLE",
        description="First date of vessel activity period")
    },
    {
        name="end_date",
        type="DATE",
        mode="REQUIRED",
        description="Last date of vessel activity period"
    },
    {
        name="msg_count",
        type="INTEGER",
        mode="REQUIRED",
        description="Number of AIS messages in activity period"
    }
]
```

</details>
<br>

<details>
<summary>Updating a table schema</summary>
<br>

```python
from google.cloud import bigquery

client = bigquery.Client()

table_id = "project.dataset.table-name"

# Add a new column to the schema.
# You could also specify an entirely new
# schema here inline or from a JSON file.
table = client.get_table(table_id)
original_schema = table.schema
new_schema = original_schema[:]  # Creates a copy of the schema.
new_schema.append(bigquery.SchemaField("n_shipname", "STRING"))

table.schema = new_schema
table = client.update_table(table, ["schema"])
```
</details>

## Formatting a table description
<details>
<summary>Adding a table description</summary>
<br>

See the advacend section of [Creating Tables and Uploading Data](#creating-tables-and-uploading-data).
</details>
<br>

<details>
<summary>Updating a table description</summary>
<br>

```python
from google.cloud import bigquery
client = bigquery.Client()

table_id = "project.dataset.table-name"
table = self.client.get_table(self.table_ref(table_id))

table.description = "New table description""
client.update_table(table, ["description"])
```
</details>
<br>

<details>
<summary>What to put in a table description</summary>
<br>

In order to help a user understand how a table was created, it is best to put in as much of the following information as possible. When a table is going into production, this will be even more formalized.

* Brief description of what the table contains and any caveats
* Link to GitHub repo that created it
* Command that created it (if possible)
* Source tables and parameters used to create it
</details>

## Deleting tables

<details>
<summary>Deleting a table</summary>
<br>

```python
from google.cloud import bigquery

bq = bigquery.Client()

table_id = "project.dataset.table-name"

job = bq.delete_table(table_id, not_found_ok=True)
print(f"Deleted {table_id}")
```
</details>
<br>

<details>
<summary>Deleting a partition in a table</summary>
<br>

```python
from google.cloud import bigquery

bq = bigquery.Client()

table_id = "project.dataset.table-name"
partition_field = "timestamp"
partition_date = "2020-01-01"

# Clear data for query date before running query
delete_query = f"DELETE FROM `{table_id}` WHERE DATE({partition_field}) = '{partition_date}'"

config = bigquery.QueryJobConfig(priority="BATCH")
job = client.query(delete_query, job_config=config)

if job.error_result:
    err = job.error_result["reason"]
    msg = job.error_result["message"]
    raise RuntimeError(f'{err}: {msg}')
else:
    job.result()
```
</details>

## Miscellaneous snippets

<details>
<summary>Listing tables in a dataset</summary>
<br>

```python
from google.cloud import bigquery

dataset = "pipe_production_v20201001"
for table in bq.list_tables(dataset=dataset):
    print("{}.{}.{}".format(table.project, table.dataset_id, table.table_id))
```
</details>
<br>

<details>
<summary>Recover deleted tables</summary>
<br>

```python
from subprocess import Popen

unixtime = "1604419200000"  # when the table existed
dataset_from = "dataset_from" # where it used to exist
dataset_to = "dataset_to" # where you want it to exist now
table = "table_name"

Popen(
    f"bq cp {dataset_from}.{table}@{unixtime} {dataset_to}.{table}".split()
).wait()
```

This is a rare instance when the command line BigQuery API needs to be used.

</details>
<br>

<details>
<summary>Copy tables</summary>
<br>

```python
from google.cloud import bigquery

# Construct a BigQuery client object.
client = bigquery.Client()


source_table_id = "project.source_dataset.source_table"
destination_table_id = "project.destination_dataset.destination_table"

job = client.copy_table(source_table_id, destination_table_id)
job.result()  # Wait for the job to complete.

print("A copy of the table created.")
```
</details>

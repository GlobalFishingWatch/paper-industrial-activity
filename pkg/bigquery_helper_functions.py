# bigquery helper fuctions


from google.cloud import bigquery
from google.cloud.exceptions import NotFound



# Construct a BigQuery client object.


def query_to_table(query, table_id, max_retries=100, retry_delay=60):
    client = bigquery.Client()

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


def update_table_description(table_id, table_description):

    client = bigquery.Client()
    
    #get table
    table = client.get_table(table_id)

    # Set table description
    table.description = table_description

    #update table with description
    client.update_table(table, ["description"]) 
'''
Pipeline operational logic.  All the real work gets done here
'''

import rtyaml
from module_name_a.utils.bigquery import BigqueryHelper


class Pipeline:
    def __init__(self, args):
        self.bigquery = BigqueryHelper()
        self.args = args

    @property
    def is_test_mode(self):
        return self.args.test

    @property
    def params(self):
        '''
        Gets a dict version of self.args

        Returns:
            self.args as dict
        '''
        return vars(self.args)

    def fishing_hours_table_description(self):
        items = {
            'pipeline': f'{self.args.PIPELINE_NAME} v{self.args.PIPELINE_VERSION}',
            'pipeline-description': self.args.PIPELINE_DESCRIPTION,
            'table-description': self.args.table_description,
            'arguments': self.params,
        }
        return rtyaml.dump(items)

    def fishing_hours_query(self):
        return self.bigquery.format_query('assets/fishing_hours.sql.j2', **self.params)

    def validation_query(self):
        return self.bigquery.format_query('assets/validate.sql.j2', **self.params)

    def run_fishing_hours(self):
        query = self.fishing_hours_query()
        partition_field = 'start_date'
        schema = 'assets/fishing_hours.schema.json'
        table_desc = self.fishing_hours_table_description()
        dest_table = self.args.dest_fishing_hours_flag_table
        start_date = self.args.start_date
        success = True

        if self.is_test_mode:
            print(query)
        else:
            # create table
            print(f'Creating table {dest_table} if not exists')
            self.bigquery.create_table(
                dest_table, schema, table_desc, partition_field=partition_field
            )

            # clear out the target partition so we don't get duplicates
            print(f'Clearing out partition where {partition_field} = {start_date}')
            self.bigquery.clear_table_partition(dest_table, partition_field, start_date)
            # run the query
            print(f'Writing fishing hours by flag to table {dest_table}')
            self.bigquery.run_query(query, dest_table)

            print('Completed successfully')

        return success

    def run_validation(self):
        query = self.validation_query()
        success = True

        if self.is_test_mode:
            print(query)
        else:
            result = self.bigquery.run_query(query)
            success = all(row['success'] for row in result)

            if success:
                print('Validation succeeded.')
            else:
                print('Validation failed.')
        return success

    def run(self):
        if self.args.operation == 'fishing_hours':
            return self.run_fishing_hours()
        elif self.args.operation == 'validate':
            return self.run_validation()
        else:
            raise RuntimeError(f'Invalid operation: {self.args.operation}')

        return False  # should not be able to get here, but just in case, return failure

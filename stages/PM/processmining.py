import pm4py
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

def importCSVToLog(filepath):
    event_log = pd.read_csv(filepath, sep=';')
    event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log, timest_format="%Y-%m-%d %H:%M:%S", timest_columns="Date")
    event_log = pm4py.format_dataframe(event_log, case_id='Case', activity_key='Action', timestamp_key='Date', timest_format="%Y-%m-%d %H:%M:%S")
    log = log_converter.apply(event_log)
    
    num_events = len(event_log)
    num_cases = len(event_log["Case"].unique())
    print(f"Imported {num_events} events with {num_cases} cases.")
    return log


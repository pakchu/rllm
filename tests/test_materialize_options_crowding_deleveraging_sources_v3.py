import pandas as pd
from training import materialize_options_crowding_deleveraging_sources_v3 as m

def test_off_grid_raw_observation_times_are_retained() -> None:
    frame=pd.DataFrame({'ts':pd.to_datetime(['2024-03-04T05:35:01Z']),'sum_open_interest':[100.0],'sum_open_interest_value':[1000.0]})
    m.validate_oi(frame)
    assert frame.loc[0,'ts']==pd.Timestamp('2024-03-04T05:35:01Z')

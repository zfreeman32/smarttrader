
import pandas as pd

# End of Month Effect Bitcoin Trading Strategy
def end_of_month_effect_signals(bitcoin_df, entry_days_before_month_end=10, exit_days_after_month_start=9):
    bitcoin_df = bitcoin_df.sort_index()
    signals = pd.DataFrame(index=bitcoin_df.index)
    signals['signal'] = 'neutral'

    grouping_index = bitcoin_df.index.tz_localize(None) if getattr(bitcoin_df.index, 'tz', None) is not None else bitcoin_df.index
    month_groups = list(bitcoin_df.groupby(grouping_index.to_period('M')))
    for idx, (_, current_month) in enumerate(month_groups):
        entry_slice = current_month.index[-entry_days_before_month_end:] if entry_days_before_month_end > 0 else current_month.index[:0]
        signals.loc[entry_slice, 'signal'] = 'long'

        if idx + 1 < len(month_groups):
            next_month = month_groups[idx + 1][1]
            hold_slice = next_month.index[:exit_days_after_month_start]
            signals.loc[hold_slice, 'signal'] = 'long'

    return signals

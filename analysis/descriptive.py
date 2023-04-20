import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DescriptiveStatistics:
    def __init__(self, dataframe):
        self.df = dataframe

    def summary_statistics(self):
        summary = self.df.groupby('event')['count'].describe()
        return summary

    def event_count(self):
        event_counts = self.df['event'].value_counts()
        return event_counts

    def plot_event_counts(self):
        event_counts = self.event_count()
        event_counts.plot(kind='bar', color='b', alpha=0.7)
        plt.xlabel('Event')
        plt.ylabel('Counts')
        plt.title('Event Counts')
        plt.show()

    def plot_mean_count_per_event(self):
        mean_counts = self.df.groupby('event')['count'].mean()
        mean_counts.plot(kind='bar', color='g', alpha=0.7)
        plt.xlabel('Event')
        plt.ylabel('Mean Count')
        plt.title('Mean Count per Event')
        plt.show()

    def plot_time_series(self, event):
        time_series_df = self.df[self.df['event'] == event]
        plt.plot(time_series_df['bin'], time_series_df['count'], marker='o', linestyle='-')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.title(f'Time Series for {event}')
        plt.show()
import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime, timedelta
import textwrap
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

userDir = os.path.expanduser('~')

# Add every font at the specified location
font_dir = [userDir + '/BirdNET-Pi/homepage/static']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set number of species to report
readings = 25

# Set Palette for graphics
pal = "Greens"

# current time
now = datetime.now()
current_hour = now.hour


def retrieve_data():
    conn = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/birds.db')
    # conn = sqlite3.connect("/Users/ford/Desktop/backup/birds.db")
    df = pd.read_sql_query("SELECT * from detections", conn)
    return df

def format_data(df):
    # Convert Date and Time Fields to Panda's format
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Add round hours to dataframe
    df['Hour of Day'] = [r.hour for r in df.Timestamp]

    return df

def get_top_n_today(df):
    # Get readings for past 24 hours
    now_rounded_up = now + timedelta(hours=1)
    now_rounded_up = now_rounded_up.replace(minute=0, second=0, microsecond=0)
    # Calculate the datetime for 24 hours ago
    one_day_ago = now_rounded_up - timedelta(days=1)

    # Filter the DataFrame for rows where the 'Timestamp' is within the past 24 hours
    df_last_24_hours = df[(df['Timestamp'] >= one_day_ago) & (df['Timestamp'] <= now)]

    plt_top_n_today = (df_last_24_hours['Com_Name'].value_counts()[:readings])
    df_plt_top_n_today = df_last_24_hours[df_last_24_hours.Com_Name.isin(plt_top_n_today.index)]
    
    if df_plt_top_n_today.empty:
        exit(0)

    return df_plt_top_n_today


df = retrieve_data()
df = format_data(df)
top_n_today = get_top_n_today(df)






# Set up plot axes and titles
f, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw=dict(width_ratios=[1, 6]), facecolor='#77C487')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

# generate y-axis order for all figures based on frequency
freq_order = pd.value_counts(top_n_today['Com_Name']).iloc[:readings].index

# make color for max confidence --> this groups by name and calculates max conf
confmax = top_n_today.groupby('Com_Name')['Confidence'].max()
# reorder confmax to detection frequency order
confmax = confmax.reindex(freq_order)

# norm values for color palette
norm = plt.Normalize(confmax.values.min(), confmax.values.max())
colors = plt.cm.Greens(norm(confmax))

# Generate frequency plot
plot = sns.countplot(y='Com_Name', data=top_n_today, palette=colors, order=freq_order, ax=axs[0])

# Try plot grid lines between bars - problem at the moment plots grid lines on bars - want between bars
z = plot.get_ymajorticklabels()
plot.set_yticklabels(['\n'.join(textwrap.wrap(ticklabel.get_text(), 15)) for ticklabel in plot.get_yticklabels()], fontsize=8)
plot.set(ylabel=None)
plot.set(xlabel="Detections")

# Generate crosstab matrix for heatmap plot
heat = pd.crosstab(top_n_today['Com_Name'], top_n_today['Hour of Day'])

# Order heatmap Birds by frequency of occurrance
heat.index = pd.CategoricalIndex(heat.index, categories=freq_order)
heat.sort_index(level=0, inplace=True)

hours_in_day = pd.Series(data=range(0, 24))
heat_frame = pd.DataFrame(data=0, index=heat.index, columns=hours_in_day)
heat = (heat + heat_frame).fillna(0)


# Reorder the columns to start from the hour after the current hour
columns_order = list(range(current_hour + 1, 24)) + list(range(0, current_hour + 1))
heat = heat[columns_order]

# Generate heatmap plot
plot = sns.heatmap(
    heat,
    norm=LogNorm(),
    annot=True,
    annot_kws={"fontsize": 7},
    fmt="g",
    cmap=pal,
    square=False,
    cbar=False,
    linewidths=0.5,
    linecolor="Grey",
    ax=axs[1],
    yticklabels=False
)


plot.set_xticklabels(plot.get_xticklabels(), rotation=0, size=7)

# Set heatmap border
for _, spine in plot.spines.items():
    spine.set_visible(True)

plot.set(ylabel=None)
plot.set(xlabel="Hour of Day")
# Set combined plot layout and titles
f.subplots_adjust(top=0.9)
plt.suptitle("Top 25 Last Updated: " + str(now.strftime("%Y-%m-%d %H:%M")))

# Save combined plot
savename = userDir + '/BirdSongs/Extracted/Charts/Combo-' + str(now.strftime("%Y-%m-%d")) + '.png'
# savename = userDir + '/Desktop/Combo-' + str(now.strftime("%Y-%m-%d")) + '.png'
plt.savefig(savename)
plt.show()
plt.close()
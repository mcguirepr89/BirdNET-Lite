#!/home/pi/BirdNET-Pi/birdnet/bin/python3
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
import textwrap

engine = sqlalchemy.create_engine("mariadb+mariadbconnector://birder:RGB123@localhost:3306/birds")

Session = sqlalchemy.orm.sessionmaker()
Session.configure(bind=engine)
Session = Session()

#Read SQL database into Pandas dataframe
df=pd.read_sql('detections', engine)

#Add round hours to dataframe
df['Hour of Day'] = [r.hour for r in df.Time]

#Create separate dataframes for separate locations
df_jhb=df #Default to use the whole Dbase

#Get todays readings
now = datetime.now()
df_jhb_today = df_jhb[df_jhb['Date']==now.strftime("%Y-%m-%d")]

#Set number of species to report
readings=10

jhb_top10_today = (df_jhb_today['Com_Name'].value_counts()[:readings])
df_jhb_top10_today = df_jhb_today[df_jhb_today.Com_Name.isin(jhb_top10_today.index)]

#Set Palette for graphics
pal = "Greens"

#Set up plot axes and titles
f, axs = plt.subplots(1, 3, figsize = (10, 4), gridspec_kw=dict(width_ratios=[3, 2, 5]))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)


#Generate frequency plot
plot=sns.countplot(y='Com_Name', data = df_jhb_top10_today, palette = pal+"_r",  order=pd.value_counts(df_jhb_top10_today['Com_Name']).iloc[:readings].index, ax=axs[0])

#Try plot grid lines between bars - problem at the moment plots grid lines on bars - want between bars
# plot.grid(True, axis='y')
z=plot.get_ymajorticklabels()
plot.set_yticklabels(['\n'.join(textwrap.wrap(ticklabel.get_text(),15)) for ticklabel in plot.get_yticklabels()], fontsize = 10)
plot.set(ylabel=None)
plot.set(xlabel="Detections")

huw=df_jhb_top10_today.groupby('Com_Name')['Confidence'].mean()
plot = sns.boxenplot(x=df_jhb_top10_today['Confidence']*100,color='Green',  y=df_jhb_top10_today['Com_Name'], ax=axs[1],order=pd.value_counts(df_jhb_top10_today['Com_Name']).iloc[:readings].index)
plot.set(xlabel="Confidence", ylabel=None,yticklabels=[])


#Generate crosstab matrix for heatmap plot

heat = pd.crosstab(df_jhb_top10_today['Com_Name'],df_jhb_top10_today['Hour of Day'])
#Order heatmap Birds by frequency of occurrance
heat.index = pd.CategoricalIndex(heat.index, categories = pd.value_counts(df_jhb_top10_today['Com_Name']).iloc[:readings].index)
heat.sort_index(level=0, inplace=True)


hours_in_day = pd.Series(data = range(0,24))
heat_frame = pd.DataFrame(data=0, index=heat.index, columns = hours_in_day)
heat=(heat+heat_frame).fillna(0)

#Generatie heatmap plot
plot = sns.heatmap(heat, norm=LogNorm(),  annot=True,  annot_kws={"fontsize":7}, cmap = pal , square = False, cbar=False, linewidths = 0.5, linecolor = "Grey", ax=axs[2], yticklabels = False)

# Set heatmap border
for _, spine in plot.spines.items():
    spine.set_visible(True)

plot.set(ylabel=None)
plot.set(xlabel="Hour of Day")
#Set combined plot layout and titles
# plt.tight_layout()
f.subplots_adjust(top=0.9)
plt.suptitle("Last Updated: "+ str(now.strftime("%d %m %Y %H:%M")))

#Save combined plot
savename='/home/pi/BirdSongs/Extracted/Charts/Combo-'+str(now.strftime("%d-%m-%Y"))+'.png'
plt.savefig(savename)



plt.close()

#Get bottom 10 today
jhb_bot10_today=(df_jhb_today['Com_Name'].value_counts()[-10:])
df_jhb_bot10_today = df_jhb_today[df_jhb_today.Com_Name.isin(jhb_bot10_today.index)]

#Set Palette for graphics
pal = "Reds"

#Set up plot axes and titles
f, axs = plt.subplots(1, 2, figsize = (8, 4), gridspec_kw=dict(width_ratios=[3, 5]))

#Generate frequency plot
plot=sns.countplot(y='Com_Name', data = df_jhb_bot10_today, palette = pal+"_r", order=pd.value_counts(df_jhb_bot10_today['Com_Name']).iloc[:10].index, ax=axs[0])
plot.set_yticklabels(['\n'.join(textwrap.wrap(ticklabel.get_text(),17)) for ticklabel in plot.get_yticklabels()])
plot.set(ylabel=None)
plot.set(xlabel="no. of detections")
#Generate crosstab matrix for heatmap plot
heat = pd.crosstab(df_jhb_bot10_today['Com_Name'],df_jhb_bot10_today['Hour of Day'])

#Order heatmap Birds by frequency of occurrance
heat.index = pd.CategoricalIndex(heat.index, categories = pd.value_counts(df_jhb_bot10_today['Com_Name']).iloc[:10].index)
heat.sort_index(level=0, inplace=True)
heat_frame = pd.DataFrame(data=0, index=heat.index, columns = hours_in_day)
heat=(heat+heat_frame).fillna(0)

#Generate heatmap plot
plot = sns.heatmap(heat, norm=LogNorm(), annot=True, annot_kws={"fontsize":7}, cmap = pal , square = False, cbar=False, linewidths = 0.5, linecolor = "Grey", ax=axs[1], yticklabels = False)

# Set heatmap border
for _, spine in plot.spines.items():
    spine.set_visible(True)
plot.set(ylabel=None)

#Set combined plot layout and titles
plt.tight_layout()
f.subplots_adjust(top=0.9)
plt.suptitle("Bottom 10 Detected: "+ str(now.strftime("%d-%h-%Y %H:%M")))
plot.set(xlabel="Hour of Day")
#Save combined plot
savename='/home/pi/BirdSongs/Extracted/Charts/Combo2-'+str(now.strftime("%d-%m-%Y"))+'.png'
plt.savefig(savename)

plt.close()


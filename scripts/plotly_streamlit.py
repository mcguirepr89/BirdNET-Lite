import os
import streamlit as st
import pandas as pd
import numpy as np
from numpy import ma
import plotly.graph_objects as go
from plotly.subplots import make_subplots
<<<<<<< Updated upstream
=======
import plotly.io as pio
>>>>>>> Stashed changes
from datetime import timedelta, datetime
from pathlib import Path
import sqlite3
from sqlite3 import Connection
import plotly.express as px
from sklearn.preprocessing import normalize

pio.templates.default = "plotly_white"

userDir = os.path.expanduser('~')
URI_SQLITE_DB = userDir + '/BirdNET-Pi/scripts/birds.db'

st.set_page_config(layout='wide')

# Remove whitespace from the top of the page
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2.5rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


<<<<<<< Updated upstream
@st.cache(hash_funcs={Connection: id})
def get_connection(path:str):
    return sqlite3.connect(path,check_same_thread=False)
=======

# @st.cache(hash_funcs={Connection: id})
@st.cache(allow_output_mutation=True)
def get_connection(path: str):
    return sqlite3.connect(path, check_same_thread=False)
>>>>>>> Stashed changes


def get_data(conn: Connection):
    df1=pd.read_sql("SELECT * FROM detections", con=conn)
    return df1

conn = get_connection(URI_SQLITE_DB)
<<<<<<< Updated upstream
# Read in the cereal data
# df = load_data()
df=get_data(conn)
df2=df.copy()
df2['DateTime']=pd.to_datetime(df2['Date'] + " " + df2['Time'])
df2=df2.set_index('DateTime')

=======
df = get_data(conn)
df2 = df.copy()
df2['DateTime'] = pd.to_datetime(df2['Date'] + " " + df2['Time'])
df2 = df2.set_index('DateTime')
>>>>>>> Stashed changes

daily = st.sidebar.checkbox('Single Day View', help= 'Select if you want single day view, unselect for multi-day views')

if daily:
# Date as slider
<<<<<<< Updated upstream
Start_Date = pd.to_datetime(df2.index.min()).date()
End_Date =   pd.to_datetime(df2.index.max()).date()
cols1,cols2= st.columns((1,1))
Date_Slider = cols1.slider('Date Range',
                                min_value = Start_Date-timedelta(days=1),
                                max_value = End_Date,
                                value=(Start_Date,
                                       End_Date)                                       
                                )



filt = (df2.index >= pd.Timestamp(Date_Slider[0])) &  (df2.index <= pd.Timestamp(Date_Slider[1]+timedelta(days=1)))
df2 = df2[filt]

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

resample_sel=cols2.radio("Select Resample Resolution - To downsample and make run faster select longer period, Daily provides a view on detections at 15 min intervals through the day", ('1 minute', '5 minutes', '10 minutes', 'Hourly', 'Daily'))

resample_times = {'1 minute':'1min',
                  '5 minutes':'5min',
                  '10 minutes':'10min',
                  'Hourly':'1H',
                  'Daily':'1D'
                  }
resample_time = resample_times[resample_sel]

df5=df2.resample(resample_time)['Com_Name'].aggregate('unique').explode()
=======
    Start_Date = pd.to_datetime(df2.index.min()).date()
    End_Date = pd.to_datetime(df2.index.max()).date()
#     cols1, cols2 = st.columns((1, 1))
    end_date = st.sidebar.slider('Date to View',
                            min_value = Start_Date,
                            max_value = End_Date,
                            value=(End_Date),
                            help= 'Select date for single day view'                                     
                            )
    start_date = end_date
else:
    Start_Date = pd.to_datetime(df2.index.min()).date()
    End_Date = pd.to_datetime(df2.index.max()).date()

#    cols1, cols2 = st.columns((1, 1))
    start_date, end_date = st.sidebar.slider('Date Range',
                                    min_value = Start_Date-timedelta(days=1),
                                    max_value = End_Date,
                                    value=(Start_Date, End_Date),
                                    help= 'Select start and end date, if same date get a clockplot for a single day'                                        
                                    )

# start_date, end_date = cols1.date_input(
#     "Date Input for Analysis - select Range for single specie analysis, select single date for daily view",
#     value=(Start_Date, End_Date),
#     min_value=Start_Date,
#     max_value=End_Date)

# start_date = datetime(2022 ,5 ,17).date()
# end_date = datetime(2022 ,5 ,17).date()

@st.cache()
def date_filter(df, start_date, end_date):
    filt = (df2.index >= pd.Timestamp(start_date)) & (df2.index <= pd.Timestamp(end_date + timedelta(days=1)))
    df = df[filt]
    return(df)

df2 = date_filter(df2, start_date, end_date)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
         unsafe_allow_html=True)
st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
         unsafe_allow_html=True)


# Select time period buttons
# Disallow "Daily time period" for "Daily Chart"
if start_date == end_date:
    resample_sel = st.sidebar.radio(
        "Resample Resolution",
        ('Raw', '15 minutes', 'Hourly'), index=1, help= 'Select resolution for single day - larger times run faster' )

    resample_times = {'Raw': 'Raw',
                      '1 minute': '1min',
                      '15 minutes': '15min',
                      'Hourly': '1H'
                      }
    resample_time = resample_times[resample_sel]

else:
    resample_sel = st.sidebar.radio(
        "Resample Resolution",
        ('Raw', '15 minutes', 'Hourly', 'DAILY'), index=1, help= 'Select resolution for species - DAILY provides time series')

    resample_times = {'Raw': 'Raw',
                      '1 minute': '1min',
                      '15 minutes': '15min',
                      'Hourly': '1H',
                      'DAILY': '1D'
                      }
    resample_time = resample_times[resample_sel]

@st.cache()
def time_resample(df, resample_time):
    if resample_time == 'Raw':
        df_resample = df['Com_Name']

    else:
        df_resample = df.resample(resample_time)['Com_Name'].aggregate('unique').explode()

    return(df_resample)
top_bird = df2['Com_Name'].mode()[0]
df5 = time_resample(df2, resample_time)
>>>>>>> Stashed changes

#Create species count for selected date range

Specie_Count=df5.value_counts()

<<<<<<< Updated upstream
#Create species treemap

=======
>>>>>>> Stashed changes
# Create Hourly Crosstab
hourly=pd.crosstab(df5,df5.index.hour, dropna=False)

# Filter on species
species = list(hourly.index)

<<<<<<< Updated upstream
cols1,cols2= st.columns((1,1))
top_N = cols1.slider(
    'Select Number of Birds to Show',
    min_value = 1,
    value=min(10,len(Specie_Count))
    )

top_N_species = (df5.value_counts()[:top_N])


specie = cols2.selectbox('Which bird would you like to explore for the dates '+str(Date_Slider[0])+' to '+str(Date_Slider[1])+'?', species,
                              index=species.index(list(top_N_species.index)[0]))


font_size=15


#specie filter
filt=df2['Com_Name']==specie

df_counts=sum(df5==specie)




if resample_time != '1D':
    fig = make_subplots(
                    rows=3, cols =2,
                    specs= [[{"type":"xy","rowspan":3}, {"type":"polar","rowspan":2}], [{"rowspan":1}, {"rowspan":1} ], [None, {"type":"xy","rowspan":1}]],
                    subplot_titles=('<b>Top '+ str(top_N) + ' Species in Date Range '+str(Date_Slider[0])+' to '+str(Date_Slider[1])+' for '+str(resample_sel)+' sampling interval.'+'</b>',
                                    'Total Detect:'+str('{:,}'.format(df_counts))+
                                    '   Confidence Max:'+str('{:.2f}%'.format(max(df2[df2['Com_Name']==specie]['Confidence'])*100))+
                                    '   '+'   Median:'+str('{:.2f}%'.format(np.median(df2[df2['Com_Name']==specie]['Confidence'])*100))
                                    )
                        )
    fig.layout.annotations[1].update(x=0.7,y=0.25, font_size=15)

    #Plot seen species for selected date range and number of species
    fig.add_trace(go.Bar(y=top_N_species.index, x=top_N_species, orientation='h'), row=1,col=1)

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis={'categoryorder':'total ascending'})


    # Set 360 degrees, 24 hours for polar plot
    theta = np.linspace(0.0, 360, 24, endpoint=False)

    specie_filt= df5==specie
    df3=df5[specie_filt]

    detections2= pd.crosstab(df3, df3.index.hour)




    d=pd.DataFrame(np.zeros((23,1))).squeeze()
    detections = hourly.loc[specie]
    detections=(d+detections).fillna(0)
    fig.add_trace(go.Barpolar(r = detections, theta=theta), row=1, col=2)
    fig.update_layout(
        autosize=False,
        width = 1000,
        height = 500,
        showlegend=False,
        polar = dict(
            radialaxis = dict(
                tickfont_size = font_size,
                showticklabels = False,
                hoverformat = "#%{theta}: <br>Popularity: %{percent} </br> %{r}"
                ),
            angularaxis = dict(
                tickfont_size= font_size,
                rotation = -90,
                direction = 'clockwise',
                tickmode='array',
                tickvals=[0,15,35,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345],
                ticktext=['12am','1am','2am','3am','4am','5am', '6am','7am','8am','9am','10am','11am','12pm','1pm','2pm','3pm','4pm','5pm', '6pm','7pm','8pm','9pm','10pm','11pm'],
                hoverformat = "#%{theta}: <br>Popularity: %{percent} </br> %{r}"
            ),
            ),
        )



    daily=pd.crosstab(df5,df5.index.date, dropna=False)

    fig.add_trace(go.Bar(x=daily.columns, y=daily.loc[specie]), row=3, col=2)

else:
    fig = make_subplots(
                    rows=1, cols =2,
                    specs= [[{"type":"xy","rowspan":1},{"type":"xy","rowspan":1}]],
                           

                    subplot_titles=('<b>Daily Top '+ str(top_N) + ' Species in Date Range '+str(Date_Slider[0])+' to '+str(Date_Slider[1])+'</b>',
                                    '<b>Daily ' + specie+ ' Detections on 15 minute intervals </b>'),
#                                     'Total Detect:'+str('{:,}'.format(df_counts))+
#                                     '   Confidence Max:'+str('{:.2f}%'.format(max(df2[df2['Com_Name']==specie]['Confidence'])*100))+
#                                     '   '+'   Median:'+str('{:.2f}%'.format(np.median(df2[df2['Com_Name']==specie]['Confidence'])*100))
#                                     )
                    )

    fig.add_trace(go.Bar(y=top_N_species.index, x=top_N_species, orientation='h'), row=1,col=1)
    df4=df2['Com_Name'][df2['Com_Name']==specie].resample('15min').count()
    df4.index=[df4.index.date, df4.index.time]
    day_hour_freq=df4.unstack().fillna(0)

    fig_x = [d.strftime('%d-%m-%Y') for d in day_hour_freq.index.tolist()]
    fig_y = [h.strftime('%H:%M') for h in day_hour_freq.columns.tolist()]
    fig_z = day_hour_freq.values.transpose()
    fig_heatmap = go.Figure(data=go.Heatmap(x=fig_x,y=fig_y,z=fig_z))

    fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    yaxis={'categoryorder':'total ascending'})
    color_pals= px.colors.named_colorscales()
    selected_pal = cols2.selectbox('Select Color Pallet for Daily Detections', color_pals)
    fig.add_trace(go.Heatmap(x=fig_x,y=fig_y,z=fig_z, autocolorscale = False, colorscale = selected_pal), row=1, col=2)
=======
#cols1, cols2 = st.columns((1, 1))
top_N = st.sidebar.slider(
    'Select Number of Birds to Show',
    min_value=1,
    max_value=len(Specie_Count),
    value=min(10, len(Specie_Count))
)

top_N_species = (df5.value_counts()[:top_N])

font_size = 15

if daily == False:
    specie = st.selectbox(
        'Which bird would you like to explore for the dates ' 
        + str(start_date) + ' to ' + str(end_date) + '?', 
        species,
        index = species.index(top_bird))

    filt = df2['Com_Name'] == specie

    df_counts = sum(df5 == specie)

    if resample_time != '1D':

        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{"type": "xy", "rowspan": 3}, {"type": "polar", "rowspan": 2}], [{"rowspan": 1}, {"rowspan": 1}],
                   [None, {"type": "xy", "rowspan": 1}]],
            subplot_titles=('<b>Top ' + str(top_N) + ' Species in Date Range ' + str(start_date) + ' to ' + str(
                end_date) + ' for ' + str(resample_sel) + ' sampling interval.' + '</b>',
                            'Total Detect:' + str('{:,}'.format(df_counts)) +
                            '   Confidence Max:' + str(
                                '{:.2f}%'.format(max(df2[df2['Com_Name'] == specie]['Confidence']) * 100)) +
                            '   ' + '   Median:' + str(
                                '{:.2f}%'.format(np.median(df2[df2['Com_Name'] == specie]['Confidence']) * 100))
                            )
        )
        fig.layout.annotations[1].update(x=0.7, y=0.25, font_size=15)

        # Plot seen species for selected date range and number of species
        fig.add_trace(go.Bar(y=top_N_species.index, x=top_N_species, orientation='h', marker_color='seagreen'), row=1, col=1)

        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            yaxis={'categoryorder': 'total ascending'})

        # Set 360 degrees, 24 hours for polar plot
        theta = np.linspace(0.0, 360, 24, endpoint=False)

        specie_filt = df5 == specie
        df3 = df5[specie_filt]

        detections2 = pd.crosstab(df3, df3.index.hour)

        d = pd.DataFrame(np.zeros((23, 1))).squeeze()
        detections = hourly.loc[specie]
        detections = (d + detections).fillna(0)
        fig.add_trace(go.Barpolar(r=detections, theta=theta, marker_color='seagreen'), row=1, col=2)
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500,
            showlegend=False,
            polar=dict(
                radialaxis=dict(
                    tickfont_size=font_size,
                    showticklabels=False,
                    hoverformat="#%{theta}: <br>Popularity: %{percent} </br> %{r}"
                ),
                angularaxis=dict(
                    tickfont_size=font_size,
                    rotation=-90,
                    direction='clockwise',
                    tickmode='array',
                    tickvals=[0, 15, 35, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270,
                              285, 300, 315, 330, 345],
                    ticktext=['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am',
                              '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'],
                    hoverformat="#%{theta}: <br>Popularity: %{percent} </br> %{r}"
                ),
            ),
        )

        daily = pd.crosstab(df5, df5.index.date, dropna=False)

        fig.add_trace(go.Bar(x=daily.columns, y=daily.loc[specie], marker_color='seagreen'), row=3, col=2)
    else:
        fig = st.container()
        fig = make_subplots(
                        rows=1, cols =1)
    #                     specs= [[{"type":"xy","rowspan":1},{"type":"heatmap","rowspan":1}]],
                            

    #                     subplot_titles=('<b>Daily Top '+ str(top_N) + ' Species in Date Range '+ str(start_date) +' to '+ str(end_date) +'</b>',
    #                                     '<b>Daily ' + specie+ ' Detections on 15 minute intervals </b>'),
    # #                                     'Total Detect:'+str('{:,}'.format(df_counts))+
    # #                                     '   Confidence Max:'+str('{:.2f}%'.format(max(df2[df2['Com_Name']==specie]['Confidence'])*100))+
    # #                                     '   '+'   Median:'+str('{:.2f}%'.format(np.median(df2[df2['Com_Name']==specie]['Confidence'])*100))
    # #                                     )
    #                     )

#        fig.add_trace(go.Bar(y=top_N_species.index, x=top_N_species, orientation='h'), row=1,col=1)
        df4=df2['Com_Name'][df2['Com_Name']==specie].resample('15min').count()
        df4.index=[df4.index.date, df4.index.time]
        day_hour_freq=df4.unstack().fillna(0)

        fig_x = [d.strftime('%d-%m-%Y') for d in day_hour_freq.index.tolist()]
        fig_y = [h.strftime('%H:%M') for h in day_hour_freq.columns.tolist()]
        fig_z = day_hour_freq.values.transpose()
#        fig_heatmap = go.Figure(data=go.Heatmap(x=fig_x,y=fig_y,z=fig_z))

        # fig.update_layout(
        # margin=dict(l=0, r=0, t=50, b=0),
        # yaxis={'categoryorder':'total ascending'})
        color_pals= px.colors.named_colorscales()
        selected_pal = st.sidebar.selectbox('Select Color Pallet for Daily Detections', color_pals)
        fig.add_trace(go.Heatmap(x=fig_x,y=fig_y,z=fig_z, autocolorscale = False, colorscale = selected_pal), row=1, col=1)

else:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy", "rowspan": 1}, {"type": "xy", "rowspan": 1}]],
        subplot_titles=('<b>Top ' + str(top_N) + ' Species For ' + str(start_date) + '</b>',
                        '<b>Daily ' + str(start_date) + ' Detections on ' + resample_sel + ' interval</b>'),
        shared_yaxes='all',
        horizontal_spacing=0
    )

    df6 = df5.to_frame(name='Com_Name')
    readings = top_N

    plt_topN_today = (df6['Com_Name'].value_counts()[:readings])
    freq_order = pd.value_counts(df6['Com_Name']).iloc[:readings].index
    #        confmax = df6.groupby('Com_Name')['Confidence'].max()
    # reorder confmax to detection frequency order
    #        confmax = confmax.reindex(freq_order)
    #         norm = plt.Normalize(confmax.values.min(), confmax.values.max())
    #
    #         colors = plt.cm.Greens(norm(confmax))
    fig.add_trace(go.Bar(y=plt_topN_today.index, x=plt_topN_today, marker_color='seagreen', orientation='h'), row=1,
                  col=1)

    #        plot=sns.countplot(y='Com_Name', data = df_plt_topN_today, palette = colors,  order=freq_order, ax=axs[0])

    df6['Hour of Day'] = [r.hour for r in df6.index.time]
    heat = pd.crosstab(df6['Com_Name'], df6['Hour of Day'])
    # Order heatmap Birds by frequency of occurrance
    heat.index = pd.CategoricalIndex(heat.index, categories=freq_order)
    heat.sort_index(level=0, inplace=True)

    heat_plot_values = ma.log(heat.values).filled(0)

    hours_in_day = pd.Series(data=range(0, 24))
    heat_frame = pd.DataFrame(data=0, index=heat.index, columns=hours_in_day)

    heat = (heat + heat_frame).fillna(0)
    heat_values_normalized = normalize(heat.values, axis=1, norm='l1')

    labels = heat.values.astype(int).astype('str')
    labels[labels == '0'] = ""
    fig.add_trace(go.Heatmap(x=heat.columns, y=heat.index, z=heat_values_normalized,  # heat.values,
                             showscale=False,
                             text=labels, texttemplate="%{text}", colorscale='Blugrn'
                             ), row=1, col=2)
    fig.update_yaxes(visible=True, autorange="reversed", ticks="inside", tickson="boundaries", ticklen=10000,
                     showgrid=True)
    fig.update_layout(xaxis_ticks="inside",
                      margin=dict(l=0, r=0, t=50, b=0))
>>>>>>> Stashed changes
# container=st.container()
# config={'displayModelBar': False}
st.plotly_chart(fig, use_container_width=True) #, config=config)

# cols3,cols4=st.columns((1,1))
# 
# extract_date=Date_Slider
# 
# audio_file = open('/home/*/BirdSongs/Extracted/By_Date/2022-03-22/Yellow-streaked_Greenbul/Yellow-streaked_Greenbul-77-2022-03-22-birdnet-15:04:28.mp3', 'rb')
# audio_bytes = audio_file.read()
# cols4.audio(audio_bytes, format='audio/mp3')

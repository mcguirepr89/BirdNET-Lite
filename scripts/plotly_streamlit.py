import os
import streamlit as st
import pandas as pd
import numpy as np
from numpy import ma
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import timedelta
import sqlite3
from sqlite3 import Connection
import plotly.express as px
from sklearn.preprocessing import normalize
from suntime import Sun
from datetime import datetime

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


@st.cache_resource()
def get_connection(path: str):
    return sqlite3.connect(path, check_same_thread=False)


@st.cache_resource()
def get_data(_conn: Connection):
    df1 = pd.read_sql("SELECT * FROM detections", con=conn)
    return df1


conn = get_connection(URI_SQLITE_DB)
df = get_data(conn)
df2 = df.copy()
df2['DateTime'] = pd.to_datetime(df2['Date'] + " " + df2['Time'])
df2 = df2.set_index('DateTime')

daily = st.sidebar.checkbox('Single Day View', help='Select if you want single day view, unselect for multi-day views')

if daily:
    # Date as slider
    Start_Date = pd.to_datetime(df2.index.min()).date()
    End_Date = pd.to_datetime(df2.index.max()).date()
#     cols1, cols2 = st.columns((1, 1))
    end_date = st.sidebar.date_input('Date to View',
                                     min_value=Start_Date,
                                     max_value=End_Date,
                                     value=(End_Date),
                                     help='Select date for single day view')
    start_date = end_date
else:
    Start_Date = pd.to_datetime(df2.index.min()).date()
    End_Date = pd.to_datetime(df2.index.max()).date()

#    cols1, cols2 = st.columns((1, 1))
    start_date, end_date = st.sidebar.slider('Date Range',
                                             min_value=Start_Date-timedelta(days=1),
                                             max_value=End_Date,
                                             value=(Start_Date, End_Date),
                                             help='Select start and end date, if same date get a clockplot for a single day')

# start_date, end_date = cols1.date_input(
#     "Date Input for Analysis - select Range for single specie analysis, select single date for daily view",
#     value=(Start_Date, End_Date),
#     min_value=Start_Date,
#     max_value=End_Date)

# start_date = datetime(2022 ,5 ,17).date()
# end_date = datetime(2022 ,5 ,17).date()


@st.cache_data()
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
        ('Raw', '15 minutes', 'Hourly'), index=1, help='Select resolution for single day - larger times run faster')

    resample_times = {'Raw': 'Raw',
                      '1 minute': '1min',
                      '15 minutes': '15min',
                      'Hourly': '1H'
                      }
    resample_time = resample_times[resample_sel]

else:
    resample_sel = st.sidebar.radio(
        "Resample Resolution",
        ('Raw', '15 minutes', 'Hourly', 'DAILY'), index=1, help='Select resolution for species - DAILY provides time series')

    resample_times = {'Raw': 'Raw',
                      '1 minute': '1min',
                      '15 minutes': '15min',
                      'Hourly': '1H',
                      'DAILY': '1D'
                      }
    resample_time = resample_times[resample_sel]


@st.cache_data()
def time_resample(df, resample_time):
    if resample_time == 'Raw':
        df_resample = df['Com_Name']

    else:
        df_resample = df.resample(resample_time)['Com_Name'].aggregate('unique').explode()

    return(df_resample)


top_bird = df2['Com_Name'].mode()[0]
df5 = time_resample(df2, resample_time)

# Create species count for selected date range

Specie_Count = df5.value_counts()

# Create Hourly Crosstab
hourly = pd.crosstab(df5, df5.index.hour, dropna=True, margins=True)

# Filter on species
species = list(hourly.sort_values("All", ascending=False).index)

# cols1, cols2 = st.columns((1, 1))
top_N = st.sidebar.slider(
    'Select Number of Birds to Show',
    min_value=1,
    max_value=len(Specie_Count),
    value=min(10, len(Specie_Count))
)

top_N_species = (df5.value_counts()[:top_N])

font_size = 15

def sunrise_sunset_scatter(num_days_to_display):
    latitude = df['Lat'][0]
    longitude = df['Lon'][0]

    sun = Sun(latitude, longitude)

    sunrise_list = []
    sunset_list = []
    sunrise_week_list = []
    sunset_week_list = []
    sunrise_text_list = []
    sunset_text_list = []

    now = datetime.now()

    for past_day in range(num_days_to_display):
        d = timedelta(days=num_days_to_display - past_day - 1)

        current_date = now - d
        # current_date = datetime.fromisocalendar(2022, week + 1, 5)
        # time_zone = datetime.now()
        sun_rise = sun.get_local_sunrise_time(current_date)
        sun_dusk = sun.get_local_sunset_time(current_date)

        sun_rise_time = float(sun_rise.hour) + float(sun_rise.minute) / 60.0
        sun_dusk_time = float(sun_dusk.hour) + float(sun_dusk.minute) / 60.0

        temp_time = str(sun_rise)[-14:-9] + " Sunrise"
        sunrise_text_list.append(temp_time)
        temp_time = str(sun_dusk)[-14:-9] + " Sunset"
        sunset_text_list.append(temp_time)
        sunrise_list.append(sun_rise_time)
        sunset_list.append(sun_dusk_time)
        sunrise_week_list.append(past_day)
        sunset_week_list.append(past_day)

    sunrise_week_list.append(None)
    sunrise_list.append(None)
    sunrise_text_list.append(None)
    sun_list = sunrise_list.extend(sunset_list)
    sun_week_list = sunrise_week_list.extend(sunset_week_list)
    sunrise_text_list.extend(sunset_text_list)

    return sunrise_week_list, sunrise_list, sunrise_text_list

def hms_to_dec(t):
    # (h, m, s) = t.split(':')
    h = t.hour
    m = t.minute / 60
    s = t.second / 3600
    result = h + m + s
    return result

def hms_to_str(t):
    # (h, m, s) = t.split(':')
    h = t.hour
    m = t.minute
    # s = t.second / 3600
    # result = h + m + s
    return "%02d:%02d" % (h, m)


if daily is False:

    if resample_time != '1D':
        specie = st.selectbox(
                'Which bird would you like to explore for the dates '
                + str(start_date) + ' to ' + str(end_date) + '?',
                species,
                index=0)

    # filt = df2['Com_Name'] == specie
        if specie == 'All':
            df_counts = int(hourly[hourly.index == specie]['All'])
            fig = make_subplots(
                rows=3, cols=2,
                specs=[[{"type": "xy", "rowspan": 3}, {"type": "polar", "rowspan": 2}],
                       [{"rowspan": 1}, {"rowspan": 1}],
                       [None, {"type": "xy", "rowspan": 1}]],
                subplot_titles=('<b>Top ' + str(top_N) + ' Species in Date Range ' + str(start_date) + ' to ' + str(
                    end_date) + '<br>for ' + str(resample_sel) + ' sampling interval.' + '</b>',
                                'Total Detect:' + str('{:,}'.format(df_counts))
                                # +   '   Confidence Max:' + str(
                                #     '{:.2f}%'.format(max(df2[df2['Com_Name'] == specie]['Confidence']) * 100)) +
                                # '   ' + '   Median:' + str(
                                #     '{:.2f}%'.format(np.median(df2[df2['Com_Name'] == specie]['Confidence']) * 100))
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
                        tickvals=[0, 15, 35, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210,
                                  225, 240, 255, 270, 285, 300, 315, 330, 345],
                        ticktext=['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am',
                                  '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm',
                                  '7pm', '8pm', '9pm', '10pm', '11pm'],
                        hoverformat="#%{theta}: <br>Popularity: %{percent} </br> %{r}"
                    ),
                ),
            )

            daily = pd.crosstab(df5, df5.index.date, dropna=True, margins=True)
            fig.add_trace(go.Bar(x=daily.columns[:-1], y=daily.loc[specie][:-1], marker_color='seagreen'), row=3, col=2)
            st.plotly_chart(fig, use_container_width=True)  # , config=config)

        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = make_subplots(
                    rows=3, cols=1,
                    specs=[[{"type": "polar", "rowspan": 2}], [{"rowspan": 1}], [{"type": "xy", "rowspan": 1}]]
                            )
                # Set 360 degrees, 24 hours for polar plot
                theta = np.linspace(0.0, 360, 24, endpoint=False)

                specie_filt = df5 == specie
                df3 = df5[specie_filt]

                detections2 = pd.crosstab(df3, df3.index.hour)

                d = pd.DataFrame(np.zeros((23, 1))).squeeze()
                detections = hourly.loc[specie]
                detections = (d + detections).fillna(0)
                fig.add_trace(go.Barpolar(r=detections, theta=theta, marker_color='seagreen'), row=1, col=1)
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
                            tickvals=[0, 15, 35, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195,
                                      210, 225, 240, 255, 270, 285, 300, 315, 330, 345],
                            ticktext=['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am',
                                      '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm',
                                      '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'],
                            hoverformat="#%{theta}: <br>Popularity: %{percent} </br> %{r}"
                        ),
                    ),
                )

                daily = pd.crosstab(df5, df5.index.date, dropna=True, margins=True)
                fig.add_trace(go.Bar(x=daily.columns[:-1], y=daily.loc[specie][:-1], marker_color='seagreen'), row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)  # , config=config)
                df_counts = int(hourly[hourly.index == specie]['All'])
                st.subheader('Total Detect:' + str('{:,}'.format(df_counts))
                             + '   Confidence Max:' +
                             str('{:.2f}%'.format(max(df2[df2['Com_Name'] == specie]['Confidence']) * 100))
                             + '   ' + '   Median:' +
                             str('{:.2f}%'.format(np.median(df2[df2['Com_Name'] == specie]['Confidence']) * 100)))

            recordings = df2[df2['Com_Name'] == specie]['File_Name']

            with col2:
                try:
                    recording = st.selectbox('Available recordings', recordings.sort_index(ascending=False))
                    date_specie = df2.loc[df2['File_Name'] == recording, ['Date', 'Com_Name']]
                    date_dir = date_specie['Date'].values[0]
                    specie_dir = date_specie['Com_Name'].values[0].replace(" ", "_")
                    st.image(userDir + '/BirdSongs/Extracted/By_Date/' + date_dir + '/' + specie_dir + '/' + recording + '.png')
                    st.audio(userDir + '/BirdSongs/Extracted/By_Date/' + date_dir + '/' + specie_dir + '/' + recording)
                except Exception:
                    st.title('RECORDING NOT AVAILABLE :(')
            # try:
            #     con = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/birds.db')
            #     cur = con.cursor()
            cola, colb, colc, cold = st.columns((3, 1, 1, 1))
            with colb:
                seen = st.checkbox('Reviewed')
            if seen:
                with colc:
                    verified = st.radio("Verification", ['True Positive', 'False Positive'])

                    if verified == "False Positive":
                        df_names = pd.read_csv(userDir + '/BirdNET-Pi/model/labels.txt', delimiter='_', names=['Sci_Name', 'Com_Name'])
                        df_unknown = pd.DataFrame({"Sci_Name": ["UNKNOWN"], "Com_Name": ["UNKNOWN"]})
                        df_names = pd.concat([df_unknown, df_names], ignore_index=True)
                        with cold:
                            corrected = st.selectbox('What species?', df_names['Com_Name'])
            #     cur.execute("UPDATE detections SET  Seen = seen WHERE File_Name = recording")
            #     con.commit()
            #     con.close()

            # except BaseException:
            #     print("Database busy")
            #     time.sleep(2)

    else:

        specie = st.selectbox('Which bird would you like to explore for the dates '
                              + str(start_date) + ' to ' + str(end_date) + '?',
                              species[1:],
                              index=0)

    # filt = df2[df2['Com_Name'] == specie]

        df_counts = int(hourly[hourly.index == specie]['All'])
        fig = st.container()
        fig = make_subplots(rows=1, cols=1)
    #                     specs= [[{"type":"xy","rowspan":1},{"type":"heatmap","rowspan":1}]],

    #                     subplot_titles=('<b>Daily Top '+ str(top_N) + ' Species in Date Range '+ str(start_date) +' to '+ str(end_date) +'</b>',
    #                                     '<b>Daily ' + specie+ ' Detections on 15 minute intervals </b>'),
    # #                                     'Total Detect:'+str('{:,}'.format(df_counts))+
    # #                                     '   Confidence Max:'+str('{:.2f}%'.format(max(df2[df2['Com_Name']==specie]['Confidence'])*100))+
    # #                                     '   '+'   Median:'+str('{:.2f}%'.format(np.median(df2[df2['Com_Name']==specie]['Confidence'])*100))
    # #                                     )
    #                     )

#        fig.add_trace(go.Bar(y=top_N_species.index, x=top_N_species, orientation='h'), row=1,col=1)
        df4 = df2['Com_Name'][df2['Com_Name'] == specie].resample('15min').count()
        df4.index = [df4.index.date, df4.index.time]
        day_hour_freq = df4.unstack().fillna(0)

        saved_time_labels = [hms_to_str(h) for h in day_hour_freq.columns.tolist()]
        fig_dec_y = [hms_to_dec(h) for h in day_hour_freq.columns.tolist()]
        fig_x = [d.strftime('%d-%m-%Y') for d in day_hour_freq.index.tolist()]
        fig_y = [h.strftime('%H:%M') for h in day_hour_freq.columns.tolist()]
        day_hour_freq.columns = fig_dec_y
        fig_z = day_hour_freq.values.transpose()
#        fig_heatmap = go.Figure(data=go.Heatmap(x=fig_x,y=fig_y,z=fig_z))

        # fig.update_layout(
        # margin=dict(l=0, r=0, t=50, b=0),
        # yaxis={'categoryorder':'total ascending'})
        color_pals = px.colors.named_colorscales()
        selected_pal = st.sidebar.selectbox('Select Color Pallet for Daily Detections', color_pals)

        heatmap = go.Heatmap(
            # x=fig_x, y=fig_y,
            x=fig_x,
            y=day_hour_freq.columns,
            z=fig_z,  # heat.values,
            showscale=False,
            # text=labels,
            texttemplate="%{text}", autocolorscale=False, colorscale=selected_pal
        )
        num_days_to_display = len(fig_x)
        sunrise_week_list, sunrise_list, sunrise_text_list = sunrise_sunset_scatter(num_days_to_display)
        daysback_range = fig_x
        daysback_range.append(None)
        daysback_range.extend(daysback_range)
        daysback_range = daysback_range[:-1]

        sunrise_sunset = go.Scatter(x=daysback_range,
                                    y=sunrise_list,
                                    mode='lines',
                                    hoverinfo='text',
                                    text=sunrise_text_list,
                                    line_color='orange', line_width=1, name=' ')

        fig = go.Figure(data=[heatmap, sunrise_sunset])
        number_of_y_ticks = 12
        y_downscale_factor = int(len(saved_time_labels) / number_of_y_ticks)
        fig.update_layout(
            yaxis=dict(
                tickmode = 'array',
                tickvals = day_hour_freq.columns[::y_downscale_factor],
                ticktext = saved_time_labels[::y_downscale_factor],
                nticks = 6
            )
        )
        st.plotly_chart(fig, use_container_width=True)  # , config=config)
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
# container=st.container()
# config={'displayModelBar': False}
    st.plotly_chart(fig, use_container_width=True)  # , config=config)

# cols3,cols4=st.columns((1,1))
# extract_date=Date_Slider
# audio_file = open('/home/*/BirdSongs/Extracted/By_Date/2022-03-22/Yellow-streaked_Greenbul/Yellow-streaked_Greenbul-77-2022-03-22-birdnet-15:04:28.mp3', 'rb')
# audio_bytes = audio_file.read()
# cols4.audio(audio_bytes, format='audio/mp3')

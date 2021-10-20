import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
import seaborn as sns
import statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess
import streamlit as st
import pydeck as pdk


_lock = RendererAgg.lock
plt.style.use('default')

##########Page Details###############
st.set_page_config(page_title='Who wins IPL - Indian Premier League',
					page_icon = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1920px-Indian_Premier_League_Official_Logo.svg.png',
                   layout="wide")

##########Load the datasets##########
@st.cache(allow_output_mutation=True)
def ball_by_ball():
	# ball_by_ball statsitics
    balls_data_ = pd.read_csv('https://github.com/nikhilYadala/kaggle_ipl_dataset/blob/main/IPL%20Ball-by-Ball%202008-2020.csv?raw=true')
    return balls_data_

balls_data = ball_by_ball()

#--------
@st.cache(allow_output_mutation=True)
def matches():
    # matches data
    matches_data_ = pd.read_csv('https://github.com/nikhilYadala/kaggle_ipl_dataset/blob/main/IPL%20Matches%202008-2020.csv?raw=true')

    return matches_data_

matches_data = matches()

####Data releated variables and pre-processing###############

city=['Mumbai','Kolkata','Delhi','Bangalore','Hyderabad','Chennai','Chandigarh','Jaipur','Pune']
teams=['Royal Challengers Bangalore','Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Kings XI Punjab','Delhi Capitals','Rajasthan Royals','Sunrisers Hyderabad']


# print(len(matches_data[matches_data['city'].isin(city)]))
replace_dict={'Delhi Daredevils':'Delhi Capitals','Deccan Chargers':'Sunrisers Hyderabad','Bengaluru':'Bangalore'}
matches_data=matches_data.replace(replace_dict)
balls_data=balls_data.replace(replace_dict)
matches_data=matches_data.query('city in @city and team1 in @teams and team2 in @teams')
matchids=matches_data['id'].to_list()
balls_data=balls_data.query('id in @matchids')


matches_data['date']=pd.to_datetime(matches_data['date'])
matches_data['year']=matches_data['date'].dt.year



# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 2, .1)
    )

row1_1.title('IPL Aggregated analysis across matches')

with row1_2:
    st.write('')
    row1_2.subheader(
    'What does a team do differently when it wins vs loses?')
    # balls_data
    # matches_data
    # matches_data.columns

# # ROW 2 ------------------------------------------------------------------------

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns(
    (.1, 1.6, .1, 1.6, .1)
    )

with row2_1:
	matches = matches_data.to_dict('records')
	matches = list(set(matches_data['team1']))
	matches = ['Kings XI Punjab', 'Royal Challengers Bangalore','Mumbai Indians','Kolkata Knight Riders','Chennai Super Kings','Delhi Capitals','Rajasthan Royals','Sunrisers Hyderabad']
	team_name = st.selectbox('Select a Team', options =matches)
								# format_func = lambda match: f'{match["team1"]}')


def extract_year(date):
    return int(date.split("-")[0])
    

with row2_2:
	# matches_data["year"] = matches_data.apply(lambda row:extract_year(row["date"]), axis =1)
	years = list(set(matches_data["year"]))
	start_year, end_year = st.select_slider('Select the year range for analysis',
												options = list(range(2008,2020)),
												value = (2008,2019))



##################### Data wranging for the analytics#######################
df_t=matches_data.query('year>=@start_year and year<=@end_year')

toss_bat = []
bat_win = []
city = []
lon = []
lat= []
lon2 = []
lat2=[]

loc={'Mumbai':[19.0760,72.8777],
    'Kolkata':[22.5726,88.3639],
    'Delhi':[28.704,77.1025],
    'Bangalore':[12.9716,77.5946],
    'Hyderabad':[17.3850,78.4867],
     'Chennai':[13.0827,80.2707],
     'Chandigarh':[30.7333,76.7794],
     'Jaipur':[26.9124,75.7873],
     'Pune':[18.5204,73.8567]
    }

offset = 0.05
loc2={'Mumbai':[19.0760+offset,72.8777],
    'Kolkata':[22.5726+offset,88.3639],
    'Delhi':[28.704+offset,77.1025],
    'Bangalore':[12.9716+offset,77.5946],
    'Hyderabad':[17.3850+offset,78.4867],
     'Chennai':[13.0827+offset,80.2707],
     'Chandigarh':[30.7333+offset,76.7794],
     'Jaipur':[26.9124+offset,75.7873],
     'Pune':[18.5204+offset,73.8567]
    }    

for name,group in df_t.groupby('city'):
    choose_bat=len(group[group['toss_decision']=='bat'])
    choose_field=len(group[group['toss_decision']=='field'])
    field_win=len(group[group['result']=='wickets'])
    toss_win=len(group[group['toss_winner']==group['winner']])
    toss_bat.append([len(group[group['toss_decision']=='bat'])/len(group)])
    bat_win.append([len(group[group['result']=='runs'])/len(group)])
    city.append([name])
    lon.append([float(loc[name][1])])
    lat.append([float(loc[name][0])])

    lon2.append([float(loc2[name][1])])
    lat2.append([float(loc2[name][0])])

df_toss_win_bat = pd.DataFrame(np.hstack((city,toss_bat, bat_win, lon, lat)), columns = ["city","toss_bat","bat_win", "lon", "lat"])
df_toss_win_bat["lon"] = pd.to_numeric(df_toss_win_bat["lon"])
df_toss_win_bat["lat"] = pd.to_numeric(df_toss_win_bat["lat"])

# df_toss_win_bat = pd.concat([df_toss_win_bat]*1000)
df_toss_win_bat = df_toss_win_bat.sample(n=30000, weights = df_toss_win_bat["toss_bat"], replace = True)



df_toss_win_bat2 = pd.DataFrame(np.hstack((city,toss_bat, bat_win, lon2, lat2)), columns = ["city","toss_bat","bat_win", "lon2", "lat2"])
df_toss_win_bat2["lon2"] = pd.to_numeric(df_toss_win_bat2["lon2"])
df_toss_win_bat2["lat2"] = pd.to_numeric(df_toss_win_bat2["lat2"])

# df_toss_win_bat2 = pd.concat([df_toss_win_bat2]*1000)
df_toss_win_bat2 = df_toss_win_bat2.sample(n=30000, weights = df_toss_win_bat2["bat_win"], replace = True)






df_join=matches_data[['id','year','winner']]
balls_data=balls_data.merge(df_join)



df_s=balls_data.query('batting_team==@team_name or bowling_team==@team_name').query('winner==@team_name')
df_s=df_s.query('year>=@start_year and year<=@end_year')

# Team runs,wickets in each over
df_batting=df_s.query('batting_team==@team_name')
no_matches=df_batting.groupby('id').ngroups
df_batting_res=(df_batting.groupby('over').sum()/no_matches)[['total_runs','is_wicket']]

# Opponent runs,wickets in each over
df_bowling=df_s.query('bowling_team==@team_name')
no_matches=df_bowling.groupby('id').ngroups
df_bowling_res=(df_bowling.groupby('over').sum()/no_matches)[['total_runs','is_wicket']]

#data for scatterplot of runs vs wickets
df_runs_wickets = df_batting.groupby("id").sum()[["total_runs","is_wicket"]]



#Does location, toss and who  bat first affect the winner#

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 3, .1)
    )

row1_1.subheader("Influence of Location and the Impact of Toss on Wins")

with row1_2:
	st.markdown('<span style="color:blue"> Height of **Blue** bar </span> -> probability that a team decides to bat after winning the toss.', unsafe_allow_html=True)
	st.markdown('<span style="color:Green"> Height of **Green** bar </span>  -> probability of the team batting first, winning the match', unsafe_allow_html=True)



row1_spacer1, row1, row1_spacer2= st.columns(
    (.1, 1, 0.1)
    )



def map(data, data2, lat, lon, zoom):
	    st.write(pdk.Deck(
	        map_style="mapbox://styles/mapbox/light-v9",
	        # initial_view_state = pdk.data_utils.compute_view(data),
	        initial_view_state={
	            "latitude": lat,
	            "longitude": lon,
	            "zoom": 5,
	            "pitch": 200,
	        },
	        layers=[
	            pdk.Layer(
	                "HexagonLayer",
	                id = 'my-hexagon-layer',
	                layerName = "myname",
	                data=data,
	                get_position=["lon", "lat"],
	                radius=10000,
	                elevation_scale=100,
	                elevation_range=[1000, 5000],
	                pickable=True,
	                extruded=True,
	                # colorDomain = [5,5],
	                colorRange = [[10,0,255]]
	                # getElevationValue=df_toss_win_bat["count"]
	            ),

	            pdk.Layer(
	                "HexagonLayer",
	                id = 'my-hexagon-layer',
	                layerName = "myname",
	                data=data2,
	                get_position=["lon2", "lat2"],
	                radius=10000,
	                elevation_scale=100,
	                elevation_range=[1000, 5000],
	                pickable=True,
	                extruded=True,
	                # colorDomain = [5,5],
	                colorRange = [[10,255,0]]
	                # getElevationValue=df_toss_win_bat["count"]
	            )
	        ]
	    ))

with row1:
	midpoint = (df_toss_win_bat["lat"].mean(), df_toss_win_bat["lon"].mean())
	# df_toss_win_bat["count"] = np.random.uniform(low=1000, high=100000, size=(df_toss_win_bat["lat"].shape[0],))
	map(df_toss_win_bat[["lon","lat"]],df_toss_win_bat2[["lon2","lat2"]], midpoint[0], midpoint[1], 11)





########winner analysis################
# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )

row1_1.subheader('Winning Matches Analysis')


####################Row 2 for graphs###################################
st.write('')
row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))


def runs_overs_plot():

  fig1 = Figure()
  ax = fig1.subplots()
  sns.lineplot(data= df_batting_res["total_runs"], color='#0085CA',
                 label=team_name,ax=ax)
  sns.lineplot(data=df_bowling_res['total_runs'], color = '#CA5800',
  				 # color=COLORS.get(team_name),
                 label='Opponent',ax=ax)
  ax.legend()
  ax.set_xlabel('Overs', fontsize=12)
  ax.set_ylabel('Runs scored', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([0,20])
  st.pyplot(fig1)

with row2_1, _lock:
    st.subheader('Runs per Over')
    st.write('')
    runs_overs_plot()
    # df_batting_res["total_runs"]
    # df_bowling_res['total_runs']
    # df_s["winner"]


def runs_wickets_plot():

  fig1 = Figure()
  ax = fig1.subplots()
  sns.lineplot(data= df_batting_res["is_wicket"], color='#0085CA',
                 label=team_name,ax=ax)
  sns.lineplot(data=df_bowling_res['is_wicket'], color = '#CA5800',
  				 # color=COLORS.get(team_name),
                 label='Opponent',ax=ax)
  ax.legend()
  ax.set_xlabel('Overs', fontsize=12)
  ax.set_ylabel('wicket Rate', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([0,20])
  st.pyplot(fig1)

with row2_2, _lock:
    st.subheader('Wickets per Over')
    st.write('')

    runs_wickets_plot()
    # df_batting_res["is_wicket"]
    # df_bowling_res['is_wicket']

def run_vs_wickets():
  fig1 = Figure()
  ax = fig1.subplots()

  sns.scatterplot(x="total_runs", y="is_wicket",data=df_runs_wickets,
    color='#0085CA',ax=ax)

  sns.scatterplot(x="total_runs", y="is_wicket",data=pd.DataFrame([[df_runs_wickets["total_runs"].mean(), df_runs_wickets["is_wicket"].mean()]], columns = ["total_runs","is_wicket"]),
    color='#FF0000',ax=ax, s=100, label = "Mean Point")


  ax.legend()
  ax.set_xlabel('Runs', fontsize=12)
  ax.set_ylabel('Wickets', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  # ax.set_xlim([0,20])
  st.pyplot(fig1)	


with row2_3, _lock:
    st.subheader('Runs vs Wickets of '+team_name)
    run_vs_wickets()
    # df_runs_wickets






########################Analysis on losing team########################

########winner analysis################
# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )

row1_1.subheader('Losing Matches Analysis')



df_s=balls_data.query('batting_team==@team_name or bowling_team==@team_name').query('winner!=@team_name')
df_s=df_s.query('year>=@start_year and year<=@end_year')

# Team runs,wickets in each over
df_batting=df_s.query('batting_team==@team_name')
no_matches=df_batting.groupby('id').ngroups
df_batting_res=(df_batting.groupby('over').sum()/no_matches)[['total_runs','is_wicket']]

# Opponent runs,wickets in each over
df_bowling=df_s.query('bowling_team==@team_name')
no_matches=df_bowling.groupby('id').ngroups
df_bowling_res=(df_bowling.groupby('over').sum()/no_matches)[['total_runs','is_wicket']]

#data for scatterplot of runs vs wickets
df_runs_wickets = df_batting.groupby("id").sum()[["total_runs","is_wicket"]]


####################Row 2 for graphs###################################
st.write('')
row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))


def runs_overs_plot():

  fig1 = Figure()
  ax = fig1.subplots()
  sns.lineplot(data= df_batting_res["total_runs"], color='#0085CA',
                 label=team_name,ax=ax)
  sns.lineplot(data=df_bowling_res['total_runs'], color = '#CA5800',
  				 # color=COLORS.get(team_name),
                 label='Opponent',ax=ax)
  ax.legend()
  ax.set_xlabel('Overs', fontsize=12)
  ax.set_ylabel('Runs scored', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([0,20])
  st.pyplot(fig1)

with row2_1, _lock:
    st.subheader('Runs per Over')
    st.write('')

    runs_overs_plot()
    # df_batting_res["total_runs"]
    # df_bowling_res['total_runs']


def runs_wickets_plot():

  fig1 = Figure()
  ax = fig1.subplots()
  sns.lineplot(data= df_batting_res["is_wicket"], color='#0085CA',
                 label=team_name,ax=ax)
  sns.lineplot(data=df_bowling_res['is_wicket'], color = '#CA5800',
  				 # color=COLORS.get(team_name),
                 label='Opponent',ax=ax)
  ax.legend()
  ax.set_xlabel('Overs', fontsize=12)
  ax.set_ylabel('wicket Rate', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([0,20])
  st.pyplot(fig1)

with row2_2, _lock:
    st.subheader('Wickets per Over')
    st.write('')

    runs_wickets_plot()
    # df_batting_res["is_wicket"]
    # df_bowling_res['is_wicket']

def run_vs_wickets():
  fig1 = Figure()
  ax = fig1.subplots()

  sns.scatterplot(x="total_runs", y="is_wicket",data=df_runs_wickets,
    color='#0085CA',ax=ax)

  sns.scatterplot(x="total_runs", y="is_wicket",data=pd.DataFrame([[df_runs_wickets["total_runs"].mean(), df_runs_wickets["is_wicket"].mean()]], columns = ["total_runs","is_wicket"]),
    color='#FF0000',ax=ax, s=100, label = "Mean Point")


  ax.legend()
  ax.set_xlabel('Runs', fontsize=12)
  ax.set_ylabel('Wickets', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  # ax.set_xlim([0,20])
  st.pyplot(fig1)	


with row2_3, _lock:
    st.subheader('Runs vs Wickets of '+team_name)
    run_vs_wickets()
    # df_runs_wickets




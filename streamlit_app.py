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


_lock = RendererAgg.lock
plt.style.use('default')

# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='Who wins IPL - Indian Premier League',
					page_icon = 'https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1920px-Indian_Premier_League_Official_Logo.svg.png',
                   layout="wide")
# READ DATA --------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def ball_by_ball():
    balls_data_ = pd.read_csv('https://github.com/nikhilYadala/kaggle_ipl_dataset/blob/main/IPL%20Ball-by-Ball%202008-2020.csv?raw=true')
    return balls_data_

balls_data = ball_by_ball()

#--------
@st.cache(allow_output_mutation=True)
def matches():

    matches_data_ = pd.read_csv('https://github.com/nikhilYadala/kaggle_ipl_dataset/blob/main/IPL%20Matches%202008-2020.csv?raw=true')

    return matches_data_

matches_data = matches()

# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )

row1_1.title('IPL Winner analysis')

with row1_2:
    st.write('')
    row1_2.subheader(
    'what affects the cricket matches?')
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
	selected_team = st.selectbox('Select a Team', options =matches)
								# format_func = lambda match: f'{match["team1"]}')
	selected_team

def extract_year(date):
    return int(date.split("-")[0])
    

with row2_2:
	matches_data["year"] = matches_data.apply(lambda row:extract_year(row["date"]), axis =1)
	years = list(set(matches_data["year"]))
	start_year = st.select_slider('Select the year range',
												options = list(range(2008,2021)),
												value = (2012))

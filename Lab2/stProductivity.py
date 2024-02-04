
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/597/productivity+prediction+of+garment+employees.zip"
)

"""
# Garment Workers Productivity

Abstract: This dataset includes important attributes of the garment manufacturing process and the productivity of the employees which had been collected manually and also been validated by the industry experts. (https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)).


"""

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    return data

data = load_data(100000)


"## Summary"    
st.dataframe(data.describe())


"""
## Raw Data

We can see all the data here by pressing check button.
"""

if st.checkbox("Show Raw Data"):
    data

"## Filtering Data by the Actual Productivity"
#TODO 9.3: add slider to the sidebar
min_prod, max_prod = st.sidebar.slider(
    "Select a range of actual productivity values",
    float(data['actual_productivity'].min()),
    float(data['actual_productivity'].max()),
    (0.0, 1.0)
)


#TODO 10.1
filtered_data = data[(data['actual_productivity'] >= min_prod) & (data['actual_productivity'] <= max_prod)]

st.write("The number of filtered data samples: ", filtered_data.shape[0])

   
fig, axes = plt.subplots(2,2)

# TODO 10.2

# TODO: Using plot.hist in pandas, plot histogram of actual_productivity data in axes[0][0] (top-right subplot area)
axes[0][0].hist(data['actual_productivity'], bins=20, color='skyblue', alpha=0.7)
axes[0][0].set_xlabel('Actual Productivity')
axes[0][0].set_ylabel('Frequency')
axes[0][0].set_title('Productivity Histogram')

# TODO: Using plot.hist in pandas, plot actual_productivity vs target_productivity data in axes[0][1] (top-right subplot area)
axes[0][1].scatter(data['actual_productivity'], data['targeted_productivity'], alpha=0.5)
axes[0][1].set_xlabel('Actual Productivity')
axes[0][1].set_ylabel('Targeted Productivity')
axes[0][1].set_title('Actual vs Targeted Productivity')

# TODO: Using plot in pandas, plot actual_productivity vs no_of_workers in axes[1][0] (bottom-left subplot area)
axes[1][0].scatter(data['actual_productivity'], data['no_of_workers'], alpha=0.5, color='green')
axes[1][0].set_xlabel('Actual Productivity')
axes[1][0].set_ylabel('Number of Workers')
axes[1][0].set_title('Actual Productivity vs Number of Workers')

# TODO: Using plot in pandas, plot actual_productivity vs smv in axes[1][1] (bottom-right subplot area)
axes[1][1].scatter(data['actual_productivity'], data['smv'], alpha=0.5, color='orange')
axes[1][1].set_xlabel('Actual Productivity')
axes[1][1].set_ylabel('SMV')
axes[1][1].set_title('Actual Productivity vs SMV')

plt.tight_layout()
st.pyplot(fig)

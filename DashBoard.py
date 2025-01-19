#!/usr/bin/env python
# coding: utf-8

# # Dasboard

# ### Importing Required Libraries

# In[34]:


import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


# ### Load and Clean the Dataset

# In[37]:


# Load your dataset
data = pd.read_csv('C:/Users/chira/Downloads/archive (1)/shopping_trends.csv')

# Ensure your data is clean
data = data.dropna()  # Drop rows with missing values


# ### Prepare Correlation Heatmap

# In[40]:


# Prepare numeric data for the heatmap
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Generate the correlation matrix
correlation_matrix = numeric_data.corr()

# Plotly heatmap
correlation_heatmap = px.imshow(
    correlation_matrix,
    text_auto=True,
    title="Correlation Heatmap",
    color_continuous_scale="Viridis"
)


# ### Initialize the Dash App

# In[43]:


# Initialize the Dash app
app = Dash(__name__, use_pages=False)


# ### Define Layout

# In[46]:


# Update the app layout to include the corrected heatmap and seasons dashboard
app.layout = html.Div([
    html.H1("Shopping Trends Dashboard", style={'textAlign': 'center'}),

    # Dropdown for selecting segmentation category
    html.Label("Select Segmentation Category:"),
    dcc.Dropdown(
        id='segmentation-dropdown',
        options=[
            {'label': 'Age', 'value': 'Age'},
            {'label': 'Category', 'value': 'Category'},
            {'label': 'Gender', 'value': 'Gender'}
        ],
        value='Age'
    ),
    dcc.Graph(id='segmentation-graph'),

    html.Br(),

    # Heatmap for correlation
    html.H2("Correlation Heatmap", style={'textAlign': 'center'}),
    dcc.Graph(
        id='correlation-heatmap',
        figure=correlation_heatmap
    ),

    html.Br(),

    # Seasons analysis section
    html.H2("Seasonal Analysis", style={'textAlign': 'center'}),
    dcc.Graph(
        id='seasonal-graph'
    )
])


# ### Adding Callbacks

# In[49]:


@app.callback(
    Output('segmentation-graph', 'figure'),
    [Input('segmentation-dropdown', 'value')]
)
def update_segmentation_chart(selected_category):
    fig = px.histogram(
        data,
        x=selected_category,
        title=f"Distribution of {selected_category}",
        color=selected_category
    )
    return fig


# ### Seasonal Analysis Callback

# In[52]:


@app.callback(
    Output('seasonal-graph', 'figure'),
    [Input('segmentation-dropdown', 'value')]  # Optional: Can use any input related to seasons
)
def update_seasonal_chart(selected_category):
    fig = px.bar(
        data,
        x='Season',  
        y='Purchase Amount (USD)',  # Replace with appropriate numeric column
        title="Seasonal Analysis of Purchases",
        color='Season',
        barmode='group'
    )
    return fig


# ### Running the App

# In[30]:


# Run the Dash app
app.run_server(mode="inline", port=8060) # Can change the port number if its already in use


# In[ ]:





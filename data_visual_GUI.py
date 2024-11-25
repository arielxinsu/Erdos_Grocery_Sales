import pandas as pd
import os
import numpy as np

# warnings 
import warnings
warnings.filterwarnings("ignore")

# Plot:
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go


# Get the current folder path
current_folder = os.getcwd()

#Import data: 
datapath = current_folder  + '/Data/'


df = pd.read_csv(datapath + 'merged_train_alt.csv')
df_test = pd.read_csv(datapath + 'test_sales.csv')

df.date = pd.to_datetime(df.date)


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout for the GUI
app.layout = html.Div([
    html.H1("Sales Visualization"),
    
    # Dropdown for store selection
    html.Label("Select Store:"),
    dcc.Dropdown(
        id='store-dropdown',
        options=[{'label': f"Store {store}", 'value': store} for store in df['store_nbr'].unique()],
        value=df['store_nbr'].unique()[0]
    ),
    
    # Dropdown for family selection
    html.Label("Select Family:"),
    dcc.Dropdown(
        id='family-dropdown',
        options=[{'label': family, 'value': family} for family in df['family'].unique()],
        value=df['family'].unique()[0]
    ),
    
    # Graph for displaying the sales data
    dcc.Graph(id='sales-graph')
])

# Callback for updating the graph
@app.callback(
    Output('sales-graph', 'figure'),
    [Input('store-dropdown', 'value'),
     Input('family-dropdown', 'value')]
)
def update_graph(storeID, family):
    # Filter the data based on the selected store and family
    df_plot = df[(df['store_nbr'] == storeID) & (df['family'] == family)]
    df_test_plot = df_test[(df_test['store_nbr'] == storeID) & (df_test['family'] == family)]
    grouped_sales = df_plot.groupby('date')['sales'].sum().reset_index()
    grouped_sales_test = df_test_plot.groupby('date')['sales'].sum().reset_index()

    # Create the figure
    fig = go.Figure()

    # Sales data
    fig.add_trace(go.Scatter(
        x=grouped_sales['date'],
        y=grouped_sales['sales'],
        mode='lines',
        name='Actual Sales',
        line=dict(color='blue'),
        showlegend=False  # Disable legend for the entire figure
    ))

    # Test sales data
    fig.add_trace(go.Scatter(
        x=grouped_sales_test['date'],
        y=grouped_sales_test['sales'],
        mode='lines',
        name='Test Sales',
        line=dict(color='orange'),
        showlegend=False  # Disable legend for the entire figure
    ))

    # Add vertical lines for events
    event_data = df_plot[df_plot['event'] == 1][['date', 'hol_event_name']].drop_duplicates()
    for _, row in event_data.iterrows():
        holiday_date = row['date']
        holiday_name = row['hol_event_name']
        fig.add_trace(go.Scatter(
            x=[holiday_date, holiday_date],
            y=[0, grouped_sales['sales'].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            hoverinfo='text',
            text=f"{holiday_name} ({holiday_date.date()})",
            showlegend=False  # Disable legend for the entire figure
        ))

    # Add transparent vertical lines for holidays with day of the week
    week_data = df_plot[['date', 'day_of_week']].drop_duplicates()
    for _, row in week_data.iterrows():
        holiday_date = row['date']
        weekname = row['day_of_week']

        fig.add_trace(go.Scatter(
            x=[holiday_date, holiday_date],
            y=[0, grouped_sales['sales'].max()],
            mode='lines',
            line=dict(color='green', dash='dash'),
            hoverinfo='text',
            text=f"{weekname} ({holiday_date.date()})",
            opacity=0.0,  # Set transparency to 50%
            showlegend=False  # Disable legend for this trace
        ))

    # Add semi-transparent vertical lines for holidays with holiday names
    holiday_data = df_plot[df_plot['hol_Nat'] == 1][['date', 'hol_Nat_name']].drop_duplicates()
    for _, row in holiday_data.iterrows():
        holiday_date = row['date']
        holiday_name = row['hol_Nat_name']

        fig.add_trace(go.Scatter(
            x=[holiday_date, holiday_date],
            y=[0, grouped_sales['sales'].max()],
            mode='lines',
            line=dict(color='green', dash='dash'),
            hoverinfo='text',
            text=f"{holiday_name} ({holiday_date.date()})",
            opacity=0.3,  # Set transparency to 50%
            showlegend=False  # Disable legend for this trace
        ))

    # Add semi-transparent vertical lines for local holidays with holiday names
    holiday_data = df_plot[df_plot['hol_Reg'] == 1][['date', 'hol_Reg_name']].drop_duplicates()
    for _, row in holiday_data.iterrows():
        holiday_date = row['date']
        holiday_name = row['hol_Reg_name']

        fig.add_trace(go.Scatter(
            x=[holiday_date, holiday_date],
            y=[0, grouped_sales['sales'].max()],
            mode='lines',
            line=dict(color='purple', dash='dash'),
            hoverinfo='text',
            text=f"{holiday_name} ({holiday_date.date()})",
            opacity=0.3,  # Set transparency to 50%
            showlegend=False  # Disable legend for this trace
        ))

    # Add local holidays 
        holiday_data = df_plot[df_plot['hol_Loc'] == 1][['date', 'hol_Loc_name']].drop_duplicates()
    for _, row in holiday_data.iterrows():
        holiday_date = row['date']
        holiday_name = row['hol_Loc_name']

        fig.add_trace(go.Scatter(
            x=[holiday_date, holiday_date],
            y=[0, grouped_sales['sales'].max()],
            mode='lines',
            line=dict(color='pink', dash='dash'),
            hoverinfo='text',
            text=f"{holiday_name} ({holiday_date.date()})",
            opacity=0.3,  # Set transparency to 50%
            showlegend=False  # Disable legend for this trace
        ))    


    # Update layout
    fig.update_layout(
        title=f"Sales for Store {storeID}, Family {family}",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        showlegend=False  # Disable legend for the entire figure
    )

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
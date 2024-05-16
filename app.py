from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

app = Dash()

# Read the dataset from a CSV file
df = pd.read_csv('dataset.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Convert 'Latitude' and 'Longitude' columns to numeric data types
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Create a dictionary to map day of the year to the corresponding date
dates_2015 = pd.date_range(start='2015-01-01', end='2015-12-31')
date_to_day_map = {date.strftime('%d/%m/%Y'): i + 1 for i, date in enumerate(dates_2015)}

# Define the app layout
app.layout = html.Div(className='container', children=[
    html.Div(className='row', children=[
        html.Div(className='column map-column', children=[
            dcc.Graph(id='scatterplot', className='scatterplot')
        ]),
        html.Div(className='column settings-column', children=[
            # Date range slider
            html.Div(className='setting-container', children=[
                html.Label('Date Range:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.RangeSlider(
                        id='date-slider',
                        min=1,
                        max=len(dates_2015),
                        step=1,
                        value=[1, 31],
                        marks={i + 1: str(i + 1) for i in range(len(dates_2015)) if i % 30 == 0},
                        tooltip={'always_visible': True, 'placement': 'bottom'}
                    )
                ]),
                html.Div(id='selected-date-range', className='slider-label'),
            ]),
            # Time of day slider
            html.Div(className='setting-container', children=[
                html.Label('Time of Day:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.RangeSlider(
                        id='time-slider',
                        min=0,
                        max=23.99,
                        step=0,
                        value=[0, 23.99],
                        marks={i: f'{i:02d}:00' for i in range(0, 25, 3)}
                    )
                ])
            ]),
            # Sex of driver dropdown
            html.Div(className='setting-container', children=[
                html.Label('Sex of Driver:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.Dropdown(
                        id='sex-dropdown',
                        options=[{'label': ["Male","Female","Unknown"][sex-1], 'value': sex} for sex in sorted(df['Sex_of_Driver'].unique())],
                        value=df['Sex_of_Driver'].unique().tolist(),
                        multi=True
                    )
                ])
            ]),
            # Age of driver slider
            html.Div(className='setting-container', children=[
                html.Label('Age of Driver:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.RangeSlider(
                        id='age-driver-slider',
                        min=0,
                        max=df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int),
                        step=1,
                        value=[0, df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int)],
                        marks={i: str(i) for i in range(0, df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int) + 1, 10)},
                        tooltip={'always_visible': True, 'placement': 'bottom'}
                    )
                ])
            ]),
            # Age of vehicle slider
            html.Div(className='setting-container', children=[
                html.Label('Age of Vehicle:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.RangeSlider(
                        id='age-vehicle-slider',
                        min=0,
                        max=df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int),
                        step=1,
                        value=[0, df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int)],
                        marks={i: str(i) for i in range(0, df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int) + 1, 10)},
                        tooltip={'always_visible': True, 'placement': 'bottom'}
                    )
                ])
            ]),
            # Light conditions slider
            html.Div(className='setting-container', children=[
                html.Label('Light Conditions (Good to Bad):', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.RangeSlider(
                        id='light-conditions-slider',
                        min=df['Light_Conditions'].min(),
                        max=df['Light_Conditions'].astype(str).str.extract('(\d+)', expand=False).astype(float).max().astype(int),
                        step=1,
                        value=[1, 7],
                    )
                ])
            ]),
            # Accident severity dropdown
            html.Div(className='setting-container', children=[
                html.Label('Severity of Accident:', className='slider-label'),
                html.Div(className='slider', children=[
                    dcc.Dropdown(
                        id='severity-dropdown',
                        options=[{'label': ["Fatal","Serious","Slight"][severity-1], 'value': severity} for severity in df['Accident_Severity'].unique()],
                        value=df['Accident_Severity'].unique().tolist(),
                        multi=True
                    )
                ])
            ]),
            # Info buttons
            html.Div(className='setting-container', id='info-button-container', children=[
                html.A(className='setting-container info-button', href="https://dash.plotly.com/dash-html-components/link", id="github", children=["View on GitHub"]),
                html.A(className='setting-container info-button', id="youtube", children=["Watch Demo"]),
            ])
        ]),
        # Charts and graphs
        html.Div(className='column chart-column', children=[
            dcc.Graph(id='severity-gender-plot', className='pie-chart'),
            dcc.Graph(id='age-driver-plot', className='age-plot'),
            dcc.Graph(id='age-vehicle-plot', className='age-plot'),
            dcc.Graph(id='impact-severity-heatmap', className='heatmap')
        ])
    ])
])

# Callback to update the selected date range text
@app.callback(
    Output('selected-date-range', 'children'),
    [Input('date-slider', 'value')]
)
def update_selected_date_range(selected_range):
    start_day, end_day = selected_range
    start_date = dates_2015[start_day - 1].strftime('%d/%m/%Y')
    end_date = dates_2015[end_day - 1].strftime('%d/%m/%Y')
    return f'Selected: {start_date} - {end_date}'

# Callback to update the scatterplot based on the selected filters
@app.callback(
    Output('scatterplot', 'figure'),
    [Input('date-slider', 'value'),
     Input('time-slider', 'value'),
     Input('age-driver-slider', 'value'),
     Input('age-vehicle-slider', 'value'),
     Input('light-conditions-slider', 'value'),
     Input('severity-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_scatterplot(selected_range, time_range, age_driver_range, age_vehicle_range, light_conditions_range, selected_severities, selected_sex):
    start_day, end_day = selected_range
    start_date = dates_2015[start_day - 1]
    end_date = dates_2015[end_day - 1]
    start_time, end_time = time_range
    min_age_driver, max_age_driver = age_driver_range
    min_age_vehicle, max_age_vehicle = age_vehicle_range

    min_light_cond_index, max_light_cond_index = light_conditions_range
    light_conditions = sorted(df['Light_Conditions'].unique())
    selected_light_conditions = light_conditions[min_light_cond_index:max_light_cond_index+1]

    # Extract hour from the 'Time' column, skipping invalid time values
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

    # Filter the dataframe based on the selected filters
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) &
                    (df['Hour'] >= start_time) & (df['Hour'] <= end_time) &
                    (df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_driver, max_age_driver, inclusive='both')) &
                    (df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_vehicle, max_age_vehicle, inclusive='both')) &
                    (df['Light_Conditions'].isin(selected_light_conditions)) &
                    (df['Accident_Severity'].isin(selected_severities)) &
                    (df['Sex_of_Driver'].isin(selected_sex))]

    # Create a dictionary for the hover data
    hover_data = {
        'Date': filtered_df['Date'],
        'Time': filtered_df['Time'],
        'Sex_of_Driver': filtered_df['Sex_of_Driver'].map({1: 'Male', 2: 'Female', 3: 'Unknown'}),
        'Age_of_Driver': filtered_df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(int),
        'Age_of_Vehicle': filtered_df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(int),
        'Light_Conditions': filtered_df['Light_Conditions'],
        'Accident_Severity': filtered_df['Accident_Severity'].map({1: 'Fatal', 2: 'Serious', 3: 'Slight'}),
        'Vehicle_Type': filtered_df['Vehicle_Type'].map({1: "Pedal cycle", 2: "Motorcycle 50cc and under", 3: "Motorcycle 125cc and under", 4: "Motorcycle over 125cc and up to 500cc", 5: "Motorcycle over 500cc", 8: "Taxi/Private hire car", 9: "Car", 10: "Minibus (8 - 16 passenger seats)", 11: "Bus or coach (17 or more pass seats)", 16: "Ridden horse", 17: "Agricultural vehicle", 18: "Tram", 19: "Van / Goods 3.5 tonnes mgw or under", 20: "Goods over 3.5t. and under 7.5t", 21: "Goods 7.5 tonnes mgw and over", 22: "Mobility scooter", 23: "Electric motorcycle", 90: "Other vehicle", 97: "Motorcycle - unknown cc", 98: "Goods vehicle - unknown weight", 99: "Unknown vehicle type (self rep only)", 103: "Motorcycle - Scooter (1979-1998)", 104: "Motorcycle (1979-1998)", 105: "Motorcycle - Combination (1979-1998)", 106: "Motorcycle over 125cc (1999-2004)", 108: "Taxi (excluding private hire cars) (1979-2004)", 109: "Car (including private hire cars) (1979-2004)", 110: "Minibus/Motor caravan (1979-1998)", 113: "Goods over 3.5 tonnes (1979-1998)", -1: "Data missing or out of range" }).fillna("Unknown")
    }

    # Remove the columns used in hover_data from filtered_df
    filtered_df = filtered_df.drop(columns=list(hover_data.keys()))

    # Create a color mapping dictionary for accident severity
    color_map = {
        'Fatal': '#d81b60',
        'Serious': '#ffc107',
        'Slight': '#1e88e5'
    }

    # Create a scatter mapbox plot
    fig = px.scatter_mapbox(
        filtered_df,
        lat='Latitude',
        lon='Longitude',
        hover_data=hover_data,
        zoom=4.5,
        color=hover_data['Accident_Severity'],
        color_discrete_map=color_map
    )

    # Customize the hover template
    fig.update_traces(hovertemplate=
        "<b>Date:</b> %{customdata[0]|%Y-%m-%d}<br>" +
        "<b>Time:</b> %{customdata[1]}<br>" +
        "<b>Sex:</b> %{customdata[2]}<br>" +
        "<b>Severity:</b> %{customdata[6]}<br>" +
        "<b>Age:</b> %{customdata[3]}<br>" +
        "<b>Vehicle Age:</b> %{customdata[4]}<br>" +
        "<b>Light Conditions:</b> %{customdata[5]}<br>" +
        "<b>Vehicle Type:</b> %{customdata[7]}<br>"  # Add this line
    )

    # Update the layout of the scatter mapbox plot
    fig.update_layout(
        # mapbox_style="open-street-map",
        mapbox_style="carto-positron",
        mapbox_zoom=4.5,
        mapbox_center={"lat": 54.5, "lon": -4},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=800,
        showlegend=False
    )

    return fig

# Callback to update the severity-gender plot based on the selected filters and selected data points
@app.callback(
    Output('severity-gender-plot', 'figure'),
    [Input('scatterplot', 'selectedData'),
     Input('date-slider', 'value'),
     Input('time-slider', 'value'),
     Input('age-driver-slider', 'value'),
     Input('age-vehicle-slider', 'value'),
     Input('light-conditions-slider', 'value'),
     Input('severity-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_severity_gender_plot(selectedData, selected_range, time_range, age_driver_range, age_vehicle_range, light_conditions_range, selected_severities, selected_sex):
    start_day, end_day = selected_range
    start_date = dates_2015[start_day - 1]
    end_date = dates_2015[end_day - 1]
    start_time, end_time = time_range
    min_age_driver, max_age_driver = age_driver_range
    min_age_vehicle, max_age_vehicle = age_vehicle_range
    min_light_cond_index, max_light_cond_index = light_conditions_range
    light_conditions = sorted(df['Light_Conditions'].unique(), key=lambda x: df['Light_Conditions'].unique().tolist().index(x))
    selected_light_conditions = light_conditions[min_light_cond_index:max_light_cond_index+1]

    # Extract hour from the 'Time' column, skipping invalid time values
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

    # Filter the dataframe based on the selected filters
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) &
                    (df['Hour'] >= start_time) & (df['Hour'] <= end_time) &
                    (df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_driver, max_age_driver, inclusive='both')) &
                    (df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_vehicle, max_age_vehicle, inclusive='both')) &
                    (df['Light_Conditions'].isin(selected_light_conditions)) &
                    (df['Accident_Severity'].isin(selected_severities)) &
                    (df['Sex_of_Driver'].isin(selected_sex))]

    # If data points are selected on the scatterplot, filter the dataframe further
    if selectedData:
        selected_points = selectedData['points']
        selected_indices = [point['pointIndex'] for point in selected_points]
        filtered_df = filtered_df.iloc[selected_indices]

    # Calculate the relative percentages of severities for each gender
    severity_ratios = filtered_df.groupby(['Sex_of_Driver', 'Accident_Severity']).size().unstack(fill_value=0)
    severity_ratios = severity_ratios.div(severity_ratios.sum(axis=1), axis=0) * 100

    # Map severity numbers to labels
    severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    severity_ratios.columns = [severity_labels[col] for col in severity_ratios.columns]

    # Map gender numbers to labels
    gender_labels = {1: 'Male', 2: 'Female', 3: 'Unknown'}
    severity_ratios.index = [gender_labels[idx] for idx in severity_ratios.index]

    fig = go.Figure()

    color_map = {
        'Fatal': '#d81b60',
        'Serious': '#ffc107',
        'Slight': '#1e88e5'
    }
    # Add traces for each severity level
    for severity in severity_ratios.columns:
        fig.add_trace(go.Bar(
            x=severity_ratios.index,
            y=severity_ratios[severity],
            name=severity,
            marker_color=color_map[severity],
            customdata=filtered_df[filtered_df['Accident_Severity'] == list(severity_labels.keys())[list(severity_labels.values()).index(severity)]].groupby('Sex_of_Driver').size(),
            hovertemplate='Percentage: %{y:.2f}%<br>Total Count: %{customdata}<extra></extra>'
        ))

    # Update the layout of the severity-gender plot
    fig.update_layout(
        title='Accident Severity Ratio by Gender',
        xaxis_title='Gender',
        barmode='stack',
        yaxis=dict(ticksuffix='%'),
        legend_title_text='Severity'
    )

    return fig

# Callback to update the age distribution plots based on the selected filters and selected data points
@app.callback(
    [Output('age-driver-plot', 'figure'),
     Output('age-vehicle-plot', 'figure')],
    [Input('scatterplot', 'selectedData'),
     Input('date-slider', 'value'),
     Input('time-slider', 'value'),
     Input('age-driver-slider', 'value'),
     Input('age-vehicle-slider', 'value'),
     Input('light-conditions-slider', 'value'),
     Input('severity-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_age_distribution_plots(selectedData, selected_range, time_range, age_driver_range, age_vehicle_range, light_conditions_range, selected_severities, selected_sex):
    start_day, end_day = selected_range
    start_date = dates_2015[start_day - 1]
    end_date = dates_2015[end_day - 1]
    start_time, end_time = time_range
    min_age_driver, max_age_driver = age_driver_range
    min_age_vehicle, max_age_vehicle = age_vehicle_range

    min_light_cond_index, max_light_cond_index = light_conditions_range
    light_conditions = sorted(df['Light_Conditions'].unique())
    selected_light_conditions = light_conditions[min_light_cond_index:max_light_cond_index+1]

    # Extract hour from the 'Time' column, skipping invalid time values
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

    # Filter the dataframe based on the selected filters
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) &
                    (df['Hour'] >= start_time) & (df['Hour'] <= end_time) &
                    (df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_driver, max_age_driver, inclusive='both')) &
                    (df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_vehicle, max_age_vehicle, inclusive='both')) &
                    (df['Light_Conditions'].isin(selected_light_conditions)) &
                    (df['Accident_Severity'].isin(selected_severities)) &
                    (df['Sex_of_Driver'].isin(selected_sex))]

    # If data points are selected on the scatterplot, filter the dataframe further
    if selectedData:
        selected_points = selectedData['points']
        selected_indices = [point['pointIndex'] for point in selected_points]
        filtered_df = filtered_df.iloc[selected_indices]

    # Age of Driver plot
    driver_age_data = filtered_df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(int)

    driver_age_bins = list(range(driver_age_data.min(), driver_age_data.max() + 2, 1))  # Adjust the bin size as needed
    driver_age_counts, _ = np.histogram(driver_age_data, bins=driver_age_bins)

    driver_fig = go.Figure()

    # Add a bar trace for the age distribution
    driver_fig.add_trace(go.Bar(
        x=driver_age_bins[:-1],
        y=driver_age_counts,
        name='Count',
        hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Add a KDE trace for the age distribution
    driver_kde = gaussian_kde(driver_age_data)
    driver_x = np.linspace(driver_age_data.min(), driver_age_data.max(), 200)
    driver_y = driver_kde(driver_x)
    driver_y_normalized = driver_y / driver_y.max() * driver_age_counts.max()
    driver_fig.add_trace(go.Scatter(
        x=driver_x,
        y=driver_y_normalized,
        mode='lines',
        name='KDE',
        line=dict(color='red', width=2),
        hoverinfo='skip'
    ))

    # Update the layout of the age of driver plot
    driver_fig.update_layout(
        title='Age of Driver Distribution',
        xaxis_title='Age',
        yaxis_title='Count',
        bargap=0.1,
    )

    # Age of Vehicle plot
    vehicle_age_data = filtered_df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(int)

    vehicle_age_bins = list(range(vehicle_age_data.min(), vehicle_age_data.max() + 2, 1))  # Adjust the bin size as needed
    vehicle_age_counts, _ = np.histogram(vehicle_age_data, bins=vehicle_age_bins)

    vehicle_fig = go.Figure()

    # Add a bar trace for the age distribution
    vehicle_fig.add_trace(go.Bar(
        x=vehicle_age_bins[:-1],
        y=vehicle_age_counts,
        name='Count',
        hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Add a KDE trace for the age distribution
    vehicle_kde = gaussian_kde(vehicle_age_data)
    vehicle_x = np.linspace(vehicle_age_data.min(), vehicle_age_data.max(), 200)
    vehicle_y = vehicle_kde(vehicle_x)
    vehicle_y_normalized = vehicle_y / vehicle_y.max() * vehicle_age_counts.max()
    vehicle_fig.add_trace(go.Scatter(
        x=vehicle_x,
        y=vehicle_y_normalized,
        mode='lines',
        name='KDE',
        line=dict(color='red', width=2),
        hoverinfo='skip'
    ))

    # Update the layout of the age of vehicle plot
    vehicle_fig.update_layout(
        title='Age of Vehicle Distribution',
        xaxis_title='Age',
        yaxis_title='Count',
        bargap=0.1,
    )

    return driver_fig, vehicle_fig

# Callback to update the impact-severity heatmap based on the selected filters and selected data points
@app.callback(
    Output('impact-severity-heatmap', 'figure'),
    [Input('scatterplot', 'selectedData'),
     Input('date-slider', 'value'),
     Input('time-slider', 'value'),
     Input('age-driver-slider', 'value'),
     Input('age-vehicle-slider', 'value'),
     Input('light-conditions-slider', 'value'),
     Input('severity-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_impact_severity_heatmap(selectedData, selected_range, time_range, age_driver_range, age_vehicle_range, light_conditions_range, selected_severities, selected_sex):
    start_day, end_day = selected_range
    start_date = dates_2015[start_day - 1]
    end_date = dates_2015[end_day - 1]
    start_time, end_time = time_range
    min_age_driver, max_age_driver = age_driver_range
    min_age_vehicle, max_age_vehicle = age_vehicle_range
    min_light_cond_index, max_light_cond_index = light_conditions_range
    light_conditions = sorted(df['Light_Conditions'].unique(), key=lambda x: df['Light_Conditions'].unique().tolist().index(x))
    selected_light_conditions = light_conditions[min_light_cond_index:max_light_cond_index+1]

    # Extract hour from the 'Time' column, skipping invalid time values
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

    # Filter the dataframe based on the selected filters
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) &
                        (df['Hour'] >= start_time) & (df['Hour'] <= end_time) &
                        (df['Age_of_Driver'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_driver, max_age_driver, inclusive='both')) &
                        (df['Age_of_Vehicle'].astype(str).str.extract('(\d+)', expand=False).astype(float).between(min_age_vehicle, max_age_vehicle, inclusive='both')) &
                        (df['Light_Conditions'].isin(selected_light_conditions)) &
                        (df['Accident_Severity'].isin(selected_severities)) &
                        (df['Sex_of_Driver'].isin(selected_sex))]

    # If data points are selected on the scatterplot, filter the dataframe further
    if selectedData:
        selected_points = selectedData['points']
        selected_indices = [point['pointIndex'] for point in selected_points]
        filtered_df = filtered_df.iloc[selected_indices]

    # Create a heatmap based on the first point of impact and accident severity
    heatmap_data = filtered_df.groupby(['1st_Point_of_Impact', 'Accident_Severity']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(columns=[1, 2, 3])  # Reorder columns to match severity levels

    # Normalize the data within each row
    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

    # Update x and y labels using the mapping dictionaries
    impact_labels = {
        0: 'No impact',
        1: 'Front',
        2: 'Back',
        3: 'Offside',
        4: 'Nearside',
        9: 'Unknown',
        -1: '?'
    }

    severity_labels = {
        1: 'Fatal',
        2: 'Serious',
        3: 'Slight'
    }

    # Update x and y labels using the mapping dictionaries
    heatmap_data.columns = heatmap_data.columns.map(severity_labels)
    heatmap_data.index = heatmap_data.index.map(impact_labels)

    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='First Point of Impact: %{y}<br>Accident Severity: %{x}<br>Relative Frequency: %{z:.2%}<extra></extra>',
        showscale=False
    ))

    # Update the layout of the heatmap
    fig.update_layout(
        title='First Point of Impact vs Accident Severity',
        xaxis_title='Accident Severity',
        # yaxis_title='First Point of Impact'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import warnings
import chardet
import base64

def create_professional_svg_logo():
    """Create a professional SVG logo for the dashboard."""
    svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 100">
        <defs>
            <linearGradient id="brandGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#FF5A5F;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#FF385C;stop-opacity:0.8" />
            </linearGradient>
        </defs>
        <rect width="300" height="100" fill="url(#brandGradient)" rx="15" ry="15"/>
        <text x="150" y="45" font-family="Arial, sans-serif" font-size="40" 
              font-weight="bold" text-anchor="middle" fill="white">AirBnB</text>
        <text x="150" y="75" font-family="Arial, sans-serif" font-size="20" 
              text-anchor="middle" fill="rgba(255,255,255,0.8)">Performance Analytics</text>
    </svg>
    '''
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

def detect_file_encoding(file_path):
    """Detect the file encoding using chardet."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_csv_with_encoding(file_path):
    """Read CSV file with detected or fallback encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']

    try:
        detected_encoding = detect_file_encoding(file_path)
        df = pd.read_csv(file_path, encoding=detected_encoding, low_memory=False)
        print(f"Successfully read file with detected encoding: {detected_encoding}")
        return df
    except Exception as e:
        print(f"Failed to read with detected encoding: {e}")

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"Successfully read file with {encoding} encoding")
            return df
        except Exception as e:
            print(f"Failed to read with {encoding} encoding: {e}")

    raise ValueError("Could not read the file with any known encoding")

def preprocess_data(df):
    """Comprehensive data preprocessing with robust error handling."""
    warnings.filterwarnings('ignore', category=UserWarning)

    # Validate columns
    required_columns = ['Start date', 'End date', '# of adults', '# of children', '# of infants', 'Earnings', 'Location', 'Unit Type', 'Status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Ensure date columns are parsed correctly
    date_columns = ['Start date', 'End date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Remove rows with invalid data
    df = df.dropna(subset=['Start date', 'Location', 'Earnings'])

    # Convert numeric columns safely
    numeric_columns = ['# of adults', '# of children', '# of infants']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Clean earnings column
    def clean_earnings(x):
        try:
            cleaned = str(x).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
            return float(cleaned) if cleaned else 0
        except (ValueError, TypeError):
            return 0

    df['Earnings'] = df['Earnings'].apply(clean_earnings)

    # Remove records with zero earnings
    df = df[df['Earnings'] > 0]

    # Calculate stay duration
    df['Stay Duration'] = (df['End date'] - df['Start date']).dt.days
    df['Stay Duration'] = df['Stay Duration'].fillna(0)

    # Calculate total guests
    df['Total Guests'] = df[numeric_columns].sum(axis=1)
    df['Total Guests'] = df['Total Guests'].replace(0, 1)

    # Ensure Listing column exists
    if 'Listing' not in df.columns:
        df['Listing'] = df.index + 1

    # Additional preprocessing for new visualizations
    df['Booking Month'] = df['Start date'].dt.month_name()
    df['Weekday'] = df['Start date'].dt.day_name()
    
    # Additional derived features
    df['Revenue per Guest'] = df['Earnings'] / df['Total Guests']
    df['Seasonal Period'] = pd.cut(
        df['Start date'].dt.month, 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall']
    )

    return df

def create_dash_app(df):
    """Create an advanced Dash application with dynamic filtering and visualizations."""
    app = dash.Dash(__name__, external_stylesheets=[
        dbc.themes.FLATLY, 
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
    ])

    # Professional color palette
    color_palette = [
        '#FF5A5F',  # Airbnb Red
        '#007A87',  # Teal
        '#00D1C1',  # Turquoise
        '#7B0051',  # Deep Purple
        '#00A699',  # Sea Green
        '#FC642D',  # Vibrant Orange
        '#484848',  # Dark Gray
    ]

    # Prepare dynamic filter components
    location_options = [{'label': loc, 'value': loc} for loc in df['Location'].unique()]
    unit_type_options = [{'label': unit, 'value': unit} for unit in df['Unit Type'].unique()]
    
    # Custom styling for filter cards
    filter_card_style = {
        'backgroundColor': 'white', 
        'border': f'1px solid {color_palette[1]}', 
        'borderRadius': '10px', 
        'padding': '15px', 
        'marginBottom': '20px'
    }

    # Apply professional layout to figures
    def apply_professional_layout(fig, title):
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': color_palette[6]},
                'x': 0.5,
                'xanchor': 'center'
            },
            plot_bgcolor='rgba(0,0,0,0.05)',
            paper_bgcolor='white',
            font={'family': 'Arial, sans-serif'},
            margin={'t': 50, 'b': 20, 'l': 20, 'r': 20}
        )
        return fig

    # Filter function to be used with callbacks
    def filter_dataframe(
        location_filter, 
        unit_type_filter, 
        min_earnings, 
        max_earnings, 
        min_stay_duration, 
        max_stay_duration, 
        min_total_guests, 
        max_total_guests
    ):
        filtered_df = df.copy()
        
        if location_filter:
            filtered_df = filtered_df[filtered_df['Location'].isin(location_filter)]
        
        if unit_type_filter:
            filtered_df = filtered_df[filtered_df['Unit Type'].isin(unit_type_filter)]
        
        filtered_df = filtered_df[
            (filtered_df['Earnings'] >= min_earnings) & 
            (filtered_df['Earnings'] <= max_earnings) &
            (filtered_df['Stay Duration'] >= min_stay_duration) & 
            (filtered_df['Stay Duration'] <= max_stay_duration) &
            (filtered_df['Total Guests'] >= min_total_guests) & 
            (filtered_df['Total Guests'] <= max_total_guests)
        ]
        
        return filtered_df

    # Visualizations Generation Function
    def generate_visualizations(filtered_df):
        # 1. Earnings by Location (Box Plot)
        earnings_boxplot = go.Figure(go.Box(
            x=filtered_df['Location'], 
            y=filtered_df['Earnings'], 
            name="Earnings", 
            marker_color=color_palette[0],
            boxmean=True
        ))
        earnings_boxplot = apply_professional_layout(earnings_boxplot, "Earnings Distribution by Location")

        # 2. Guest Composition by Location (Stacked Bar Chart)
        guest_comp = filtered_df.groupby('Location')[['# of adults', '# of children', '# of infants']].mean().reset_index()
        guest_comp_fig = go.Figure()
        guest_comp_fig.add_trace(go.Bar(x=guest_comp['Location'], y=guest_comp['# of adults'], name='Adults', marker_color=color_palette[1]))
        guest_comp_fig.add_trace(go.Bar(x=guest_comp['Location'], y=guest_comp['# of children'], name='Children', marker_color=color_palette[2]))
        guest_comp_fig.add_trace(go.Bar(x=guest_comp['Location'], y=guest_comp['# of infants'], name='Infants', marker_color=color_palette[3]))
        guest_comp_fig.update_layout(barmode='stack')
        guest_comp_fig = apply_professional_layout(guest_comp_fig, "Average Guest Composition by Location")

        # 3. Monthly Earnings Trend
        df_monthly = filtered_df.groupby(filtered_df['Start date'].dt.to_period('M'))['Earnings'].sum().reset_index()
        df_monthly['Start date'] = df_monthly['Start date'].astype(str)
        monthly_earnings_line = go.Figure(go.Scatter(
            x=df_monthly['Start date'], 
            y=df_monthly['Earnings'], 
            mode='lines+markers', 
            name="Monthly Earnings",
            line=dict(color=color_palette[4], width=3),
            marker=dict(size=8)
        ))
        monthly_earnings_line = apply_professional_layout(monthly_earnings_line, "Monthly Earnings Trend")

        # 4. Earnings Heatmap by Booking Month and Weekday
        earnings_heatmap = filtered_df.pivot_table(
            values='Earnings', 
            index='Booking Month', 
            columns='Weekday', 
            aggfunc='mean'
        )
        # Specific order for months and days
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        earnings_heatmap = earnings_heatmap.reindex(index=month_order, columns=day_order)

        earnings_heatmap_fig = go.Figure(data=go.Heatmap(
            z=earnings_heatmap.values,
            x=earnings_heatmap.columns,
            y=earnings_heatmap.index,
            colorscale='Viridis',
            hoverongaps = False,
            text=np.round(earnings_heatmap.values, 2),
            texttemplate='%{text}',
            textfont={"size":8}
        ))
        earnings_heatmap_fig = apply_professional_layout(earnings_heatmap_fig, "Average Earnings by Month and Weekday")

        # 5. Scatter Plot: Stay Duration vs Total Guests vs Earnings
        scatter_fig = px.scatter(
            filtered_df, 
            x='Stay Duration', 
            y='Earnings', 
            color='Total Guests',
            size='Total Guests',
            hover_data=['Location', 'Unit Type'],
            color_continuous_scale='Viridis',
            title='Earnings vs Stay Duration (Color and Size: Total Guests)'
        )
        scatter_fig = apply_professional_layout(scatter_fig, "Earnings Relationship with Stay Duration")

        # 6. Pie Chart: Unit Type Distribution
        unit_type_pie = px.pie(
            filtered_df, 
            names='Unit Type', 
            values='Earnings', 
            title='Earnings Distribution by Unit Type',
            color_discrete_sequence=color_palette
        )
        unit_type_pie = apply_professional_layout(unit_type_pie, "Earnings Distribution by Unit Type")

        # 7. Seasonal Performance Analysis
        seasonal_performance = filtered_df.groupby('Seasonal Period')['Earnings'].agg(['mean', 'sum', 'count']).reset_index()
        seasonal_fig = go.Figure()
        seasonal_fig.add_trace(go.Bar(
            x=seasonal_performance['Seasonal Period'], 
            y=seasonal_performance['mean'], 
            name='Average Earnings',
            marker_color=color_palette[5]
        ))
        seasonal_fig.add_trace(go.Scatter(
            x=seasonal_performance['Seasonal Period'], 
            y=seasonal_performance['count'], 
            name='Number of Bookings', 
            yaxis='y2',
            mode='lines+markers',
            marker_color=color_palette[2]
        ))
        seasonal_fig.update_layout(
            title='Seasonal Performance Analysis',
            yaxis=dict(title='Average Earnings'),
            yaxis2=dict(title='Number of Bookings', overlaying='y', side='right')
        )
        seasonal_fig = apply_professional_layout(seasonal_fig, "Seasonal Performance Analysis")

        # 8. Correlation Heatmap
        correlation_columns = ['Earnings', 'Stay Duration', 'Total Guests', 'Revenue per Guest', '# of adults', '# of children', '# of infants']
        correlation_matrix = filtered_df[correlation_columns].corr()
        correlation_fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size":8}
        ))
        correlation_fig = apply_professional_layout(correlation_fig, "Correlation Heatmap of Key Metrics")

        # Visualizations Layout
        return [
            # First Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Earnings Overview", className="text-center", style={'color': color_palette[0]})),
                        dbc.CardBody([dcc.Graph(figure=earnings_boxplot)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Guest Composition", className="text-center", style={'color': color_palette[1]})),
                        dbc.CardBody([dcc.Graph(figure=guest_comp_fig)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6)
            ], className="mb-4"),

            # Second Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Monthly Performance", className="text-center", style={'color': color_palette[4]})),
                        dbc.CardBody([dcc.Graph(figure=monthly_earnings_line)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Earnings by Month & Day", className="text-center", style={'color': color_palette[2]})),
                        dbc.CardBody([dcc.Graph(figure=earnings_heatmap_fig)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6)
            ], className="mb-4"),

            # Third Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Earnings vs Stay Duration", className="text-center", style={'color': color_palette[3]})),
                        dbc.CardBody([dcc.Graph(figure=scatter_fig)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Unit Type Performance", className="text-center", style={'color': color_palette[5]})),
                        dbc.CardBody([dcc.Graph(figure=unit_type_pie)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6)
            ], className="mb-4"),

            # Fourth Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Seasonal Performance", className="text-center", style={'color': color_palette[0]})),
                        dbc.CardBody([dcc.Graph(figure=seasonal_fig)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Metrics Correlation", className="text-center", style={'color': color_palette[3]})),
                        dbc.CardBody([dcc.Graph(figure=correlation_fig)])
                    ], className="mb-4 shadow-sm")
                ], width=12, lg=6)
            ])
        ]

    # Dashboard Layout
    app.layout = dbc.Container([
        # Header with Logo
        dbc.Row([
            dbc.Col([
                html.Img(src=create_professional_svg_logo(), style={
                    'height': '120px', 
                    'margin-bottom': '20px', 
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'border-radius': '15px'
                }),
            ], width=12, className="text-center")
        ]),

        # Filters Row
        dbc.Row([
            # Location Filter
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Location Filter"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='location-dropdown',
                            options=location_options,
                            multi=True,
                            placeholder="Select Locations"
                        )
                    ])
                ], style=filter_card_style)
            ], width=12, md=4),

            # Unit Type Filter
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Unit Type Filter"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='unit-type-dropdown',
                            options=unit_type_options,
                            multi=True,
                            placeholder="Select Unit Types"
                        )
                    ])
                ], style=filter_card_style)
            ], width=12, md=4),

            # Earnings Slider
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Earnings Range"),
                    dbc.CardBody([
                        dcc.RangeSlider(
                            id='earnings-slider',
                            min=df['Earnings'].min(),
                            max=df['Earnings'].max(),
                            step=(df['Earnings'].max() - df['Earnings'].min()) / 100,
                            marks={
                                df['Earnings'].min(): f'${df["Earnings"].min():.0f}',
                                df['Earnings'].max(): f'${df["Earnings"].max():.0f}'
                            },
                            value=[df['Earnings'].min(), df['Earnings'].max()]
                        )
                    ])
                ], style=filter_card_style)
            ], width=12, md=4)
        ], className="mb-4"),

        # Stay Duration Slider
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stay Duration"),
                    dbc.CardBody([
                        dcc.RangeSlider(
                            id='stay-duration-slider',
                            min=df['Stay Duration'].min(),
                            max=df['Stay Duration'].max(),
                            step=(df['Stay Duration'].max() - df['Stay Duration'].min()) / 100,
                            marks={
                                df['Stay Duration'].min(): f'{df["Stay Duration"].min():.0f} days',
                                df['Stay Duration'].max(): f'{df["Stay Duration"].max():.0f} days'
                            },
                            value=[df['Stay Duration'].min(), df['Stay Duration'].max()]
                        )
                    ])
                ], style=filter_card_style)
            ], width=12, md=4),

            # Total Guests Slider
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Total Guests"),
                    dbc.CardBody([
                        dcc.RangeSlider(
                            id='total-guests-slider',
                            min=df['Total Guests'].min(),
                            max=df['Total Guests'].max(),
                            step=1,
                            marks={
                                df['Total Guests'].min(): str(df['Total Guests'].min()),
                                df['Total Guests'].max(): str(df['Total Guests'].max())
                            },
                            value=[df['Total Guests'].min(), df['Total Guests'].max()]
                        )
                    ])
                ], style=filter_card_style)
            ], width=12, md=4),

            # Reset Filters Button
            dbc.Col([
                dbc.Button(
                    "Reset All Filters", 
                    id="reset-filters-btn", 
                    color="primary", 
                    className="mt-4 w-100",
                    style={'backgroundColor': color_palette[0], 'borderColor': color_palette[0]}
                )
            ], width=12, md=4)
        ], className="mb-4"),

        # Dynamic Visualizations Placeholder
        html.Div(id='visualizations-container')
    ], fluid=True)

    # Callback for filtering and updating visualizations
    @app.callback(
        Output('visualizations-container', 'children'),
        [
            Input('location-dropdown', 'value'),
            Input('unit-type-dropdown', 'value'),
            Input('earnings-slider', 'value'),
            Input('stay-duration-slider', 'value'),
            Input('total-guests-slider', 'value'),
            Input('reset-filters-btn', 'n_clicks')
        ]
    )
    def update_visualizations(
        location_filter, 
        unit_type_filter, 
        earnings_range, 
        stay_duration_range, 
        total_guests_range,
        reset_clicks
    ):
        # Handle reset button
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters-btn.n_clicks':
            # Reset all filters to their default values
            location_filter = None
            unit_type_filter = None
            earnings_range = [df['Earnings'].min(), df['Earnings'].max()]
            stay_duration_range = [df['Stay Duration'].min(), df['Stay Duration'].max()]
            total_guests_range = [df['Total Guests'].min(), df['Total Guests'].max()]

        # Filter DataFrame
        filtered_df = filter_dataframe(
            location_filter, 
            unit_type_filter, 
            earnings_range[0], 
            earnings_range[1],
            stay_duration_range[0], 
            stay_duration_range[1],
            total_guests_range[0], 
            total_guests_range[1]
        )

        # Generate visualizations based on filtered data
        return generate_visualizations(filtered_df)

    return app

def main():
    warnings.filterwarnings('ignore')
    try:
        df = read_csv_with_encoding('airbnb_open_data.csv')
        df = preprocess_data(df)
        
        if df.empty:
            print("No valid data found after preprocessing.")
            return
        
        app = create_dash_app(df)
        app.run_server(debug=True, port=8050)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
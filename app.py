################################################################################################################################################
 # Imports
################################################################################################################################################
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import plotly.express as px

################################################################################################################################################
 # Data Preprocessing
################################################################################################################################################
df = pd.read_csv('stars.csv')

numerical_columns = ['Temperature', 'L', 'R', 'A_M']
categorical_columns = ['Color', 'Spectral_Class']
mapping = {0: 'Red Dwarf', 
           1: 'Brown Dwarf',
           2: 'White Dwarf',
           3: 'Main Sequence', 
           4: 'Super Giants', 
           5: 'Hyper Giants'}

X = df[numerical_columns]
y = df['Type']
smote = SMOTE(random_state=42) # for imbalanced data
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest with class weights
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

################################################################################################################################################
# App Layout
################################################################################################################################################
app = dash.Dash(__name__)
app.title = "NASA Star Classification"

app.layout = html.Div(
    style={
        "backgroundColor": "#0a0a47", # dark blue
        "backgroundImage": "url('https://www.publicdomainpictures.net/pictures/320000/velka/starry-sky.jpg')",
        "backgroundSize": "cover",
        "color": "#f7f75c",  # light yellow
        "padding": "20px",
    },
    children=[
        html.H1("NASA Star Classification", style={'textAlign': 'center'}),
        
        # Scatterplot 
        html.Div([
            html.H2("Scatterplot (Numerical Variables)"),
            html.Label("Select X-axis:", style={'color': '#f7f75c'}),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value='Temperature',  # default
                style={'marginBottom': '10px', 'color': 'black'}  
            ),
            html.Label("Select Y-axis:", style={'color': '#f7f75c'}),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in numerical_columns],
                value='L',  # default
                style={'marginBottom': '10px', 'color': 'black'}  
            ),
            dcc.Graph(id='scatter-plot'),
        ]),

        # Bar Plot 
        html.Div([
            html.H2("Bar Chart (Categorical Variables)"),
            html.Label("Select Categorical Variable:", style={'color': '#f7f75c'}),
            dcc.Dropdown(
                id='categorical-variable',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value='Color',  # default
                style={'marginBottom': '10px', 'color': 'black'} 
            ),
            dcc.Graph(id='bar-chart'),
        ]),

        # User Input 
        html.Div([
            html.H2("Make Predictions"),
            html.P("Input the features of a star:"),
            html.Div([
                html.Div([
                    html.Label("Temperature (K):"),
                    dcc.Input(id='input-temperature', type='number', value=5000, style={'width': '100%'}),
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Label("Luminosity (log10(L/Lo)):", style={'marginTop': '10px'}),
                    dcc.Input(id='input-luminosity', type='number', value=0.5, style={'width': '100%'}),
                ], style={'marginBottom': '10px'}),

                html.Div([
                    html.Label("Radius (log10(R/Ro)):", style={'marginTop': '10px'}),
                    dcc.Input(id='input-radius', type='number', value=0.5, style={'width': '100%'}),
                ], style={'marginBottom': '10px'}),

                html.Div([
                    html.Label("Absolute Magnitude:", style={'marginTop': '10px'}),
                    dcc.Input(id='input-magnitude', type='number', value=5.0, style={'width': '100%'}),
                ], style={'marginBottom': '10px'}),
            ]),
            html.Button('Predict', id='predict-button', n_clicks=0, style={'marginTop': '20px'}),
            html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px'}),
        ]),
    ]
)

################################################################################################################################################
# Callbacks
################################################################################################################################################

# Scatterplot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'),
     Input('y-axis', 'value')]
)
def update_scatter_plot(x_axis, y_axis):
    figure = px.scatter(
        df, x=x_axis, y=y_axis, color=df['Type'].map(mapping),
        title=f"{x_axis} vs. {y_axis}",
        labels={x_axis: x_axis, y_axis: y_axis},
        template="plotly_dark",
        symbol_sequence=["star"]
    )
    figure.update_traces(marker=dict(size=10))  
    return figure

# Bar Plot
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('categorical-variable', 'value')]
)
def update_bar_chart(categorical_variable):
    counts = df[categorical_variable].value_counts()
    figure = px.bar(
        counts, x=counts.index, y=counts.values,
        labels={'x': categorical_variable, 'y': 'Count'},
        title=f"Distribution of {categorical_variable}",
        template="plotly_dark",
        color_discrete_sequence=["#c478f0"] # light purple
    )
    return figure

# Predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-temperature', 'value'),
     State('input-luminosity', 'value'),
     State('input-radius', 'value'),
     State('input-magnitude', 'value')]
)
def predict_star_type(n_clicks, temperature, luminosity, radius, magnitude):
    if n_clicks > 0:
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'L': [luminosity],
            'R': [radius],
            'A_M': [magnitude],
        })
        
        input_scaled = scaler.transform(input_data)
        prediction_numeric = rf.predict(input_scaled)[0]
        prediction_label = mapping[prediction_numeric]
        return f"Predicted Star Type: {prediction_label}"
    return "Awaiting input..."

################################################################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
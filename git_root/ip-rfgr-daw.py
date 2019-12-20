#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.display import display, IFrame, HTML
import os

def show_app(app, port=9999, width=900, height=700):
    host = 'localhost'
    url = f'http://{host}:{port}'

    display(HTML(f"<a href='{url}' target='_blank'>Open in new tab</a>"))
    display(IFrame(url, width=width, height=height))
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    return app.run_server(debug=False, host=host, port=port)

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle

from scipy.spatial.distance import euclidean, cityblock, cosine
from dash.dependencies import Input, Output, State

app = dash.Dash(assets_folder='web/data', assets_url_path='web/data')

with open('data/cluster0_agg_re.pickle', 'rb') as fp:
    cluster0_agg_re = pickle.load(fp) # Offlane - Disabler / Healer - Position 3/4 - Utility
    
with open('data/cluster1_agg_re.pickle', 'rb') as fp:
    cluster1_agg_re = pickle.load(fp) # Support - Position 5
    
with open('data/cluster2_agg_re.pickle', 'rb') as fp:
    cluster2_agg_re = pickle.load(fp) # Carry/Mid - Position 1/2

# List of hero names
hero_names = ['antimage',
 'axe',
 'bane',
 'bloodseeker',
 'crystal_maiden',
 'drow_ranger',
 'earthshaker',
 'juggernaut',
 'mirana',
 'morphling',
 'nevermore',
 'phantom_lancer',
 'puck',
 'pudge',
 'razor',
 'sand_king',
 'storm_spirit',
 'sven',
 'tiny',
 'vengefulspirit',
 'windrunner',
 'zuus',
 'kunkka',
 'lina',
 'lion',
 'shadow_shaman',
 'slardar',
 'tidehunter',
 'witch_doctor',
 'lich',
 'riki',
 'enigma',
 'tinker',
 'sniper',
 'necrolyte',
 'warlock',
 'beastmaster',
 'queenofpain',
 'venomancer',
 'faceless_void',
 'skeleton_king',
 'death_prophet',
 'phantom_assassin',
 'pugna',
 'templar_assassin',
 'viper',
 'luna',
 'dragon_knight',
 'dazzle',
 'rattletrap',
 'leshrac',
 'furion',
 'life_stealer',
 'dark_seer',
 'clinkz',
 'omniknight',
 'enchantress',
 'huskar',
 'night_stalker',
 'broodmother',
 'bounty_hunter',
 'weaver',
 'jakiro',
 'batrider',
 'chen',
 'spectre',
 'doom_bringer',
 'ancient_apparition',
 'ursa',
 'spirit_breaker',
 'gyrocopter',
 'alchemist',
 'invoker',
 'silencer',
 'obsidian_destroyer',
 'lycan',
 'brewmaster',
 'shadow_demon',
 'lone_druid',
 'chaos_knight',
 'meepo',
 'treant',
 'ogre_magi',
 'undying',
 'rubick',
 'disruptor',
 'nyx_assassin',
 'naga_siren',
 'keeper_of_the_light',
 'wisp',
 'visage',
 'slark',
 'medusa',
 'troll_warlord',
 'centaur',
 'magnataur',
 'shredder',
 'bristleback',
 'tusk',
 'skywrath_mage',
 'abaddon',
 'elder_titan',
 'legion_commander',
 'techies',
 'ember_spirit',
 'earth_spirit',
 'abyssal_underlord',
 'terrorblade',
 'phoenix',
 'oracle',
 'winter_wyvern',
 'arc_warden',
 'monkey_king',
 'dark_willow',
 'pangolier',
 'grimstroke',
 'mars']

# Object retrieval function
def nearest_k(df, query, objects, k, dist):
    """Return the indices to objects most similar to query
    
    Parameters
    ----------
    df : dataframe
        Dataframe to retrieve target names from
    query : ndarray
        query object represented in the same form vector representation as the
        objects
    objects : ndarray
        vector-represented objects in the database; rows correspond to 
        objects, columns correspond to features
    k : int
        number of most similar objects to return
    dist : function
        accepts two ndarrays as parameters then returns their distance
    
    Returns
    -------
    most_similar : ndarray
        Indices to the most similar objects in the database
    """
    indices = np.argsort([dist(i, query) for i in objects])[1:k+1]
    
    return df.iloc[indices, :].index.tolist()

app.callback_map = {}
app.layout = html.Div([
    html.Div([
    html.Div('DOTA2: Banned Hero Alternative Recommender', 
             style={'font-family': 'Arial', 'text-align': 'center', 
                    'font-weight': 'bold', 'font-size': '48px'}),
    
    # Intro
    html.Br(),
    html.Div('''In every sport, it is the goal of every team to win. 
                The most important factor in winning a DOTA2 game is to 
                properly execute your team’s game strategy by drafting the 
                right heroes. However, in professional games, each team 
                is only allowed to pick five (5) heroes for their team 
                and ban six (6) heroes to hamper the opposing team’s 
                strategy.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''By banning heroes, teams are forced to make 
                adjustments to their strategy that may cause their defeat. 
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''The objective of this website is to minimize the deviation to 
                the team’s overall strategy when a key hero is banned by 
                recommending the most similar hero based on their game impact.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''Whether or not you're playing professional or public games,
                you can now try the hero recommender system below!
            ''', style={'text-align': 'justify'}),
    html.Br(), 
    html.Hr(),
    html.Br(),
    
    # Input
    html.Div([
        html.Div([
            dcc.Dropdown(id='banned_hero',
            options=[{'label': i, 'value': i} for i in hero_names],
            placeholder='Heroes'
        )], style={'width': '49%', 'display': 'inline-block', 'color': 'black'}),
        html.Div([
            dcc.Dropdown(id='banned_hero_role',
            options=[{'label': 'Utility', 'value': 'Utility'}, 
                     {'label': 'Support', 'value': 'Support'}, 
                     {'label': 'Core', 'value': 'Core'}],
            placeholder='Roles'
        )], style={'width': '49%', 'display': 'inline-block', 'color': 'black'}),
    ], style={'font-family': 'Arial', 'text-align': 'center'}),
    
    html.Br(),
    
    # Input Images
    html.Div([
        html.Div(id='banned_hero_image', 
                 style={'width': '49%', 'display': 'inline-block',
                       'text-align': 'center', 'vertical-align': 'middle',
                       'font-family': 'Arial'}),
        html.Div(id='banned_hero_role_image', 
                 style={'width': '49%', 'display': 'inline-block',
                       'text-align': 'center', 'vertical-align': 'middle',
                       'font-family': 'Arial'})
    ]),
    
    html.Br(),
    
    # Alternatives
    
    html.Div(id='alternative_heroes', 
             style={'text-align': 'center', 'vertical-align': 'middle',
                   'font-family': 'Arial'}),
    html.Br(), 
    html.Hr(),
    html.Br(),
    
    # FAQ
    html.Div('Frequently Asked Questions (FAQ)', 
             style={'font-family': 'Arial', 'text-align': 'center', 
                    'font-weight': 'bold', 'font-size': '24px'}),
    html.Br(),
    
    # What is DOTA2?
    html.Div('What is DOTA2?', 
             style={'font-family': 'Arial', 'text-align': 'left', 
                    'font-weight': 'bold', 'font-size': '21px'}),
    html.Br(),
    html.Div('''DotA2 is a team-oriented game pitting two teams of five 
                players against each other. A game is won by destroying 
                the enemy's Ancient building before they destroy yours. 
                The Ancient is the largest building and is centrally located
                in each team's base. The teams are often referred to as the 
                Radiant and the Dire.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div(html.Img(src=app.get_asset_url('radiantvsdire.jpg'),
                     style={'max-width': '100%', 'max-height': '100%'}),
             style={'text-align': 'center'}),
    html.Div('Radiant vs Dire', 
             style={'text-align': 'center', 'font-weight': 'bold'}),
    html.Br(),
    html.Div('''The goal of each team is to spend time gaining resources 
                such as experience and gold while limiting and reducing 
                the opposing team's resources. The team with greater 
                resources will have a bigger advantage, enabling them to 
                destroy important objectives and eventually, the enemy's 
                Ancient.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    
    # What is my motivation?
    html.Div('What is my motivation?', 
             style={'font-family': 'Arial', 'text-align': 'left', 
                    'font-weight': 'bold', 'font-size': '21px'}),
    html.Br(),
    html.Div('''The e-Sports industry is currently a multi-million-dollar 
                industry that is poised to reach new heights this year. 
                According to recent studies, the total revenue for the 
                e-Sports market is projected to hit $1.1 billion dollars 
                this year -- 26.7% higher than the previous year.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''Due to high tournament prize pools with amounts reaching 
                to at most $30 million dollars, traditional sports teams 
                have been investing on e-Sports teams to get their start 
                in the industry.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''For every competitive sport, it is obvious that winning is 
                crucial. DotA 2 is not any different. A big determinant for 
                winning DotA 2 games is how a team drafts to execute their 
                game strategy.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    
    # How was the Recommender System created?
    html.Div('How was the Recommender System created?', 
             style={'font-family': 'Arial', 'text-align': 'left', 
                    'font-weight': 'bold', 'font-size': '21px'}),
    html.Br(),
    html.Div('''To create the recommender system, we employed the use of
                unsupervised machine learning. We clustered 
                heroes based on their game impact using the data scraped 
                from OpenDota's API. The data consists of premium matches 
                starting from August 27, 2018 to June 16, 2019 amounting to 
                15260 hero observations with 14 features.
            ''', style={'text-align': 'justify'}),
    html.Br(),
    html.Div('''K-Means algorithm was used in performing the clustering.
                Once clustered, similar heroes to the banned key hero can 
                be retrieved from a specific role/cluster that a team wants 
                it to be played, using a distance metric called 
                Euclidean distance.
            ''', style={'text-align': 'justify'}),
    html.Br(),
], style={'font-family': 'Arial', 'text-align': 'center', 'color': 'white',
          'padding': '50px',
          'margin-left': '28%', 'margin-right': '28%'})
], style={'background-color': 'black',
          'background-image': 'url(web/data/dota2bg.jpg)',
          'background-position': 'top',
          'background-repeat': 'no-repeat',
          'background-attachment': 'scroll',
          'background-size': 'contain'})

@app.callback(Output('banned_hero_image', 'children'),
             [Input('banned_hero', 'value')])
def display_banned_hero(banned_hero):
    if banned_hero == None:
        return html.Div('Choose a banned hero!')
    else:
        return html.Img(src=app.get_asset_url(str(banned_hero) + '.png'),
                        style={'max-width': '100%', 'max-height': '100%'})

@app.callback(Output('banned_hero_role_image', 'children'),
             [Input('banned_hero_role', 'value')])
def display_banned_hero_role(banned_hero_role):
    if banned_hero_role == None:
        return html.Div("Choose the banned hero's role!")
    else:
        if banned_hero_role == 'Utility':
           role_choice = html.Div(str(banned_hero_role),
                                 style={'font-weight': 'bold',
                                        'font-size': '18px'})
           role_desc = html.Div('''Heroes that are versatile and can adapt to 
                                   the pace of the game. A mixture of core and
                                   support.''',
                     style={'max-width': '100%', 'max-height': '100%'})
        elif banned_hero_role == 'Support':
           role_choice = html.Div(str(banned_hero_role),
                                 style={'font-weight': 'bold',
                                        'font-size': '18px'})
           role_desc = html.Div('''Heroes that are primarily played to support
                                   cores by providing healing and purchasing 
                                   items that benefit the whole team.''',
                     style={'max-width': '100%', 'max-height': '100%'})
        elif banned_hero_role == 'Core':
           role_choice = html.Div(str(banned_hero_role),
                                 style={'font-weight': 'bold',
                                        'font-size': '18px'})
           role_desc = html.Div('''Heroes that are designed to be the primary 
                                   damage dealers throughout the game and are 
                                   usually allocated the highest amount of 
                                   resources.''',
                     style={'max-width': '100%', 'max-height': '100%'})
        
        return [html.Img(src=app.get_asset_url(str(banned_hero_role) + '.png'),
                         style={'max-width': '100%', 'max-height': '100%'}),
                html.Br(),
                role_choice, 
                role_desc]

@app.callback(Output('alternative_heroes', 'children'),
         [Input('banned_hero', 'value'),
          Input('banned_hero_role', 'value')])
def findsimilarhero(banned_hero, banned_hero_role, k=3):
    '''Returns a dataframe of k most similar hero from the query using
    euclidean distance
    
    Parameters
    ----------
    banned_hero : str
        hero name to query
    banned_hero_role : str
        cluster to query in
    k : int
        number of similar heroes to return
        
    Returns
    -------
    df : dataframe
        dataframe of k most similar hero from the query
    '''
    try:
        if banned_hero_role == 'Utility':
            banned_hero_cluster = cluster0_agg_re
        elif banned_hero_role == 'Support':
            banned_hero_cluster = cluster1_agg_re
        elif banned_hero_role == 'Core':
            banned_hero_cluster = cluster2_agg_re

        alts = nearest_k(banned_hero_cluster, 
                         banned_hero_cluster.loc[banned_hero, :].values, 
                         banned_hero_cluster.values, 
                         k, 
                         euclidean)
        return [html.Div('Top 3 Alternatives',
                         style={'font-family': 'Arial', 'text-align': 'center', 
                                'font-weight': 'bold', 'font-size': '24px'}),
                html.Br(),
                html.Div([html.Img(src=app.get_asset_url(str(alts[0]) + '.png'),
                     style={'max-width': '100%', 'max-height': '100%'}),
                          html.Div(alts[0])], style={'width': '33%', 
                                         'display': 'inline-block'}),
                html.Div([html.Img(src=app.get_asset_url(str(alts[1]) + '.png'),
                     style={'max-width': '100%', 'max-height': '100%'}),
                          html.Div(alts[1])], style={'width': '33%', 
                                         'display': 'inline-block'}),
                html.Div([html.Img(src=app.get_asset_url(str(alts[2]) + '.png'),
                     style={'max-width': '100%', 'max-height': '100%'}),
                          html.Div(alts[2])], style={'width': '33%', 
                                         'display': 'inline-block'})]
    except:
        if banned_hero == None:
            return html.Div('',
                             style={'font-family': 'Arial', 
                                    'text-align': 'center', 
                                    'font-weight': 'bold', 
                                    'font-size': '24px'})
        elif banned_hero_role == None:
            return html.Div('',
                             style={'font-family': 'Arial', 
                                    'text-align': 'center', 
                                    'font-weight': 'bold', 
                                    'font-size': '24px'})
        else:
            return html.Div('There are no alternatives for ' + 
                            str(banned_hero) + 
                            ' as a ' + str(banned_hero_role) + '.',
                            style={'font-family': 'Arial', 
                                   'text-align': 'center', 
                                   'font-weight': 'bold', 
                                   'font-size': '24px'})
    
if __name__ == '__main__':
    app.run_server(debug=True)
    
show_app(app)


# In[ ]:





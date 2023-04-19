import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import pickle
from utils import get_encoder


Server_win = pickle.load(open('models/classifier.pkl','rb'))    
encoder = get_encoder(col='player')
df_info = pd.read_csv("data/data.csv", index_col=0)

def serve_winner(test_point):
    i = random.random()
    a = test_point
    if i < Server_win.predict_proba(a)[0][1]:
        return 1
    else:
        return 0

def simulate_points(server,receiver,GN,serve_number,server_points_total):
    
    test_point = np.array([server,receiver,GN,serve_number,server_points_total]).reshape(1,-1)
    point_winner = serve_winner(test_point)
    server_win_prob = Server_win.predict_proba(test_point)[0][1]
    return point_winner, server_win_prob


def simulate_game(server, receiver):  
    player_score, player_sets, player_games, final_score = dict(), dict(), dict(), dict()
    
    i = 0
    
    # csv_columns = ['server','Receiver','Serv_prob']
    #server = np.random.choice(df_info['server']) 
    #receiver = np.random.choice(df_info['Receiver'])

    receiver_name  = encoder.inverse_transform(receiver.reshape(-1,1))[0]
    server_name =  encoder.inverse_transform(server.reshape(-1,1))[0]
    result_df = pd.DataFrame(columns=['server', 'server_name', 'receiver', 'receiver_name', 'winner', 'winner_name'])
    tot_player_points = {server_name:0, receiver_name:0}
    player1, player2 = server_name, receiver_name
    win_prob = {player1: "NA", player2: "NA"}
    
    game_counter = 0
    serve_counter = 0
    
    player_sets[f'{server_name}_sets'], player_sets[f'{receiver_name}_sets'] = 0, 0
    final_score[server], final_score[receiver] = 0, 0

    while player_sets[f'{server_name}_sets'] != 2 and player_sets[f'{receiver_name}_sets'] != 2:
        player_games[f'{server_name}_games'],player_games[f'{receiver_name}_games'] = 0, 0
        games_to_win = 6
        while player_games[f'{server_name}_games'] != games_to_win and player_games[f'{receiver_name}_games'] != games_to_win:
            player_score[f'{server_name}_score'], player_score[f'{receiver_name}_score'] = 0, 0
            deuce = False
            server_name, receiver_name = receiver_name, server_name
            server, receiver = receiver, server
            while deuce == True or (player_score[f'{server_name}_score'] < 4 and player_score[f'{receiver_name}_score'] < 4):
                point_winner, server_win_prob = simulate_points(server,receiver,game_counter,serve_counter,
                                               tot_player_points[server_name])
                if point_winner == 1:
                    player_score[f'{server_name}_score'] += 1
                    tot_player_points[server_name] += 1
                else :
                    player_score[f'{receiver_name}_score'] += 1
                    tot_player_points[receiver_name] += 1
                if player_score[f'{server_name}_score'] >= 3 and player_score[f'{receiver_name}_score'] >= 3:
                    deuce = True
                    if abs(player_score[f'{server_name}_score'] - player_score[f'{receiver_name}_score']) == 2:
                        deuce = False
                        
                server_win_prob = round(server_win_prob, 2)
                win_prob[server_name] = server_win_prob
                win_prob[receiver_name] = 1 - server_win_prob
                result_df = result_df.append(pd.DataFrame([[server, server_name, receiver, receiver_name, point_winner, point_winner]], 
                                              columns=['server', 'server_name', 'receiver', 'receiver_name', 'winner', 'winner_name']))
                
                i += 1
                serve_counter += 1
                
            if player_score[f'{server_name}_score'] > player_score[f'{receiver_name}_score']:
                player_games[f'{server_name}_games'] += 1
            else:
                player_games[f'{receiver_name}_games'] += 1
            if player_games[f'{server_name}_games'] == 5 and player_games[f'{receiver_name}_games'] == 5:
                games_to_win = 7
                
            game_counter += 1
             
        if player_games[f'{server_name}_games'] > player_games[f'{receiver_name}_games']:
            player_sets[f'{server_name}_sets'] += 1
            final_score[server] += 1
        else:
            player_sets[f'{receiver_name}_sets'] += 1
            final_score[receiver] += 1

    game_winner = sorted(final_score.items(), key=lambda x: x[1])[-1][0]
    
    return result_df, game_winner
    

def tennis_match():
    player_score, player_sets, player_games = dict(), dict(), dict()
    
    i = 0
    
    # csv_columns = ['server','Receiver','Serv_prob']
    server = np.random.choice(df_info['server']) 
    receiver = np.random.choice(df_info['Receiver'])
    p1, p2 = server, receiver

    receiver_name  = encoder.inverse_transform(receiver.reshape(-1,1))[0]
    server_name =  encoder.inverse_transform(server.reshape(-1,1))[0]
    display_df = pd.DataFrame(columns=[server_name, receiver_name])
    tot_player_points = {server_name:0, receiver_name:0}
    player1, player2 = server_name, receiver_name
    win_prob = {player1: "NA", player2: "NA"}
    
    game_counter = 0
    serve_counter = 0
    
    player_sets[f'{server_name}_sets'], player_sets[f'{receiver_name}_sets'] = 0, 0
    
    place_holder = st.empty()
    place_holder1 = st.empty()
    while player_sets[f'{server_name}_sets'] != 2 and player_sets[f'{receiver_name}_sets'] != 2:
        player_games[f'{server_name}_games'],player_games[f'{receiver_name}_games'] = 0, 0
        games_to_win = 6
        while player_games[f'{server_name}_games'] != games_to_win and player_games[f'{receiver_name}_games'] != games_to_win:
            player_score[f'{server_name}_score'], player_score[f'{receiver_name}_score'] = 0, 0
            deuce = False
            server_name, receiver_name = receiver_name, server_name
            server, receiver = receiver, server
            while deuce == True or (player_score[f'{server_name}_score'] < 4 and player_score[f'{receiver_name}_score'] < 4):
                point_winner, server_win_prob = simulate_points(server,receiver,game_counter,serve_counter,
                                               tot_player_points[server_name])
                if point_winner == 1:
                    player_score[f'{server_name}_score'] += 1
                    tot_player_points[server_name] += 1
                else :
                    player_score[f'{receiver_name}_score'] += 1
                    tot_player_points[receiver_name] += 1
                if player_score[f'{server_name}_score'] >= 3 and player_score[f'{receiver_name}_score'] >= 3:
                    deuce = True
                    if abs(player_score[f'{server_name}_score'] - player_score[f'{receiver_name}_score']) == 2:
                        deuce = False
                print(f'\033[92m{i}\033[0m', server, player_score, tot_player_points, '\n') 

                server_win_prob = round(server_win_prob, 2)
                winners_list = []
                for _ in range(100):
                    result_df, game_winner = simulate_game(p1, p2)
                    winners_list.append(game_winner)
                
                p1_odds = sum([1 for i in winners_list if i==p1])
                p2_odds = 100 - p1_odds
                win_prob[player1] = p1_odds
                win_prob[player2] = p2_odds
                # win_prob[server_name] = server_win_prob
                # win_prob[receiver_name] = 1 - server_win_prob

                # with place_holder.container():
                #     st.write('Server is -',server_name,player_score,'\n')
                #     st.write(player_games, '\n')
                #     st.write(player_sets, '\n')
                #     st.write('Server winning probablity is -',server_win_prob,'\n')
                
                with place_holder1.container():
                    display_df.loc["Score", server_name] = player_score[f'{server_name}_score']
                    display_df.loc["Score", receiver_name] = player_score[f'{receiver_name}_score']
                    display_df.loc["Games", server_name] = player_games[f'{server_name}_games']
                    display_df.loc["Games", receiver_name] = player_games[f'{receiver_name}_games']
                    display_df.loc["Sets", server_name] = player_sets[f'{server_name}_sets']
                    display_df.loc["Sets", receiver_name] = player_sets[f'{receiver_name}_sets']
                    st.write(display_df)
                    
                    odds1, odds2 = st.columns(2)
                    with odds1:
                        st.write(str(player1) + "_prob : ", win_prob[player1])
                    with odds2:
                        st.write(str(player2) + "_prob : ", win_prob[player2])
                    
                
                #time.sleep(2)   
                i += 1
                serve_counter += 1
                
            if player_score[f'{server_name}_score'] > player_score[f'{receiver_name}_score']:
                player_games[f'{server_name}_games'] += 1
            else:
                player_games[f'{receiver_name}_games'] += 1
            if player_games[f'{server_name}_games'] == 5 and player_games[f'{receiver_name}_games'] == 5:
                games_to_win = 7
            
            print(f'\033[95m{i}\033[0m', player_games, tot_player_points, '\n')
            # with place_holder.container():
            #     st.write(player_score, '\n')
            #     st.write(player_games, '\n')
                
            #time.sleep(2)    
            game_counter += 1
             
        if player_games[f'{server_name}_games'] > player_games[f'{receiver_name}_games']:
            player_sets[f'{server_name}_sets'] += 1
        else:
            player_sets[f'{receiver_name}_sets'] += 1
        
        with place_holder1.container():
            display_df.loc["Sets", server_name] = player_sets[f'{server_name}_sets']
            display_df.loc["Sets", receiver_name] = player_sets[f'{receiver_name}_sets']
            st.write(display_df)
        
        print(f'\033[93m{i}\033[0m', player_sets, tot_player_points, '\n')
        # with place_holder.container():
        #     st.write( player_games,'\n')
        #     st.write( player_sets, '\n')
        #time.sleep(2)

    return player_sets


st.title("Tennis Match Simulation")


if st.button("Simulate Match"):
    winner = tennis_match()
    st.write("The Final score:", winner)







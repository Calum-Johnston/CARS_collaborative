import pandas as pd
import numpy as np
from os import system
from scipy.spatial.distance import cosine
import math

contextList = ['sunny', 'cloudy', 'rainy', 'snowing']
contextScore = [8, 6, 4, 2]
#[0.32653061224489793, 0.046052631578947366, 0.021875, 0.014184397163120567]

def generateRecommendations(userID, context):

    # Get the predicted item scores for each context
    itemScores = []
    alphaValues = []
    for c in contextList:
        itemScore, userAverageRating = generateItemScores(userID, c)
        itemScores.append(itemScore)
        alphaValues.append(1/userAverageRating)
    
    # Predict item scores based on all context
    contextWeights = []
    currentContextScore = contextScore[contextList.index(context)]
    for c_score in contextScore:
        score_Diff = abs(c_score - currentContextScore)
        if(score_Diff == 0):
            score_Diff = 1
        inv_score_Diff = 1 / score_Diff
        alpha_Value = alphaValues[0] / alphaValues[1]
        weight = inv_score_Diff * alphaValues[contextScore.index(c_score)]
        contextWeights.append(weight)
    
    # Calculate final item scores
    finalItemScores = []
    for item in range(0, len(itemScores[0])):
        num_total = 0
        den_total = 0
        for context in range(0, 4):
            num_total += itemScores[context][item][1] * contextWeights[context]
            den_total += contextWeights[context]
        finalItemScore = num_total / den_total
        finalItemScores.append([itemScores[context][item][0], finalItemScore])
    
    # Sort the final item ratings by value
    finalItemScores.sort(key=lambda x: x[1], reverse=True)

    # Put top items in dataframe, merge, and remove unuseful information
    topItems = pd.DataFrame(finalItemScores[0:10], columns=["ItemID", "Rating"])
    items = pd.merge(music_data, topItems, on="ItemID", how="right", sort="False")
    items = items.sort_values("Rating", ascending=False)
    items = items.drop("imageurl", 1); items = items.drop("description", 1)
    items = items.drop("mp3url", 1); items = items.drop("album", 1)
    items = items.drop("category_id", 1)
    
    # Return the recommendations
    return items

def generateItemScores(userID, context):
    # Reduce dimensions to remove unnecessary context
    df = ratings_data[ratings_data['weather'] == context]
    # Converts input data to Users x Items table
    result = df.pivot_table(index='UserID', columns="ItemID", values="Rating").fillna(0)

    # Get user rating information
    userData = result.loc[userID,:]
    print(userData)
    userAverageRating = userData.sum() / ((userData != 0).sum())
    
    # Predict user similarity
    bestUsers = []
    for userID_2, userData_2 in result.iterrows():
        if(userID_2 != userID):
            result_cos = 1 - cosine(userData, userData_2)
            if(math.isnan(result_cos)):             
                bestUsers.append([userID_2, 0]) # Set equal to 0 if nan value responded
            else:
                bestUsers.append([userID_2, result_cos])

    # Determine most similar users
    bestUsers.sort(key=lambda x: x[1], reverse=True)

    # Put top users in dataframe
    topUsers = pd.DataFrame()
    for user in bestUsers[0:10]:
        topUsers = topUsers.append(result.loc[user[0],:])

    # Compute average ratings of top rated users
    averageRatings = []
    for index, row in topUsers.iterrows():
        averageRating = row.sum() / ((row != 0).sum())
        averageRatings.append(averageRating)

    # Predict item scores
    bestItems = []
    userData = userData.to_numpy()
    itemPos = 0
    for item, col in topUsers.iteritems():
        predicted_Score = 0
        if(userData[itemPos] == 0):
            num_total = 0
            den_total = 0
            position = 0
            for user_item, value in col.iteritems():
                if(value > 0):
                    num_total += (bestUsers[position][1] * (value - averageRatings[position]))
                    den_total += bestUsers[position][1]
                position += 1
            if den_total != 0:
                predicted_Score = userAverageRating + (num_total / den_total)
        bestItems.append([item, predicted_Score])
        itemPos += 1
    
    # Returns the predicted rating of all items for the given context
    return bestItems, userAverageRating




def signInUser():
    signedIn = False
    users = ratings_data.UserID.unique()
    while(not signedIn):  
        displaySignInMenu()
        user = int(input(""))
        clear()
        if(user in users):
            print("Sign in successful!")
            return user
        else:
            print("That user does not exist within the system")
        
def clear():
    _ = system('cls')

def displaySignInMenu():
    print("CARS Recommendation System")
    print("==========================")
    print("Please enter your user ID")
    
def displayMainMenu(UserID):
    _ = system('cls')
    print("CARS Recommendation System")
    print("==========================")
    print("Signed in as: " + str(userID))
    print("Press any enter to generate recommendations")

def displayItems(items):
    _ = system('cls')
    print("Your recommendations are:")
    print(items)

def mainMenu(userID):
    completed = False
    while(not completed):
        displayMainMenu(userID)
        generate = input("")
        items = generateRecommendations(userID, 'snowing')
        displayItems(items)
        x = input()

# Read in data
_ = system('cls')
ratings_data = pd.read_csv("ContextualRatings_InCarMusic.csv", index_col=False, delimiter=',', encoding="utf-8-sig")
music_data = pd.read_csv("MusicData_InCarMusic.csv", index_col=False, delimiter=',', encoding="utf-8-sig")
userID = signInUser()
mainMenu(userID)
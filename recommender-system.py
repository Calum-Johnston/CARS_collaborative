import pandas as pd
import numpy as np
from os import system
from scipy.spatial.distance import cosine
import math

contextList = ['sunny', 'cloudy', 'rainy', 'snowing']
contextScore = [0.8, 0.6, 0.4, 0.2]

# Generates recommendations based on user-user collaborative recommendations
def generateRecommendations(data, userID, context):

    # Get the predicted item scores for each context
    itemScores = []
    userTotalRatings = []
    for c in contextList:
        itemScore, userRatingCount = singleContextRecommendation(data, userID, c)
        count = 0
        for item in items:
            try:
                if(itemScore[count][0] != item):
                    lst = [item, 0]
                    itemScore.insert(count, lst)
                
                if(itemScore[count][1] > 5):
                    itemScore[count][1] = 5
                elif(itemScore[count][1] < 1):
                    itemScore[count][1] = 1
            except:
                lst = [item, 0]
                itemScore.insert(count, lst)
            count += 1
        itemScores.append(itemScore)
        userTotalRatings.append(userRatingCount)


    # Calculate the weighting each context has
    contextWeights = []
    count = 0
    currentContextScore = contextScore[contextList.index(context)]
    for c_score in contextScore:
        score_Diff = abs(c_score - currentContextScore)
        if(score_Diff == 0):
            score_Diff = 1
        inv_score_Diff = 1 / score_Diff
        weight = inv_score_Diff * (userTotalRatings[count] / sum(userTotalRatings))
        contextWeights.append(weight)
    
    # Predict the item ratings
    finalItemRatings = {}
    for item in range(0, len(itemScores[0])):
        num_total = 0
        den_total = 0
        for context in range(0, 4):
            num_total += itemScores[context][item][1] * contextWeights[context]
            den_total += contextWeights[context]
        finalItemRating = num_total / den_total
        finalItemRatings[itemScores[context][item][0]] = finalItemRating

    # Return the item recommendations
    return finalItemRatings


# Calculates user similarity and returns an ordered list of them
def userSimilarity(result, userID, userRatings):
    similarUsers = []
    for userID_2, userData_2 in result.iterrows():
        if(userID_2 != userID):
            result_cos = 1 - cosine(userRatings, userData_2)
            if(math.isnan(result_cos)):             
                similarUsers.append([userID_2, 0]) # Set equal to 0 if nan value responded
            else:
                similarUsers.append([userID_2, result_cos])
    similarUsers.sort(key=lambda x: x[1], reverse=True)
    return similarUsers

# Put most similar users into a dataframe
def topUsersDataframe(result, similarUsers):
    mostSimilarUsers = pd.DataFrame()
    for user in similarUsers[0:10]:
        mostSimilarUsers = mostSimilarUsers.append(result.loc[user[0],:])
    return mostSimilarUsers
    
# Compute average ratings of top rated users
def getAverageRatings(mostSimilarUsers):
    averageRatings = []
    for index, row in mostSimilarUsers.iterrows():
        averageRating = row.sum() / ((row != 0).sum())
        averageRatings.append(averageRating)
    return averageRatings

# Predict item scores
def getPredictedItemScores(mostSimilarUsers, similarUsers, userAverageRating, userData, averageRatings):
    itemRatings = []
    userData = userData.to_numpy()
    itemPos = 0
    for item, col in mostSimilarUsers.iteritems():
        predicted_Score = 0
        if(userData[itemPos] == 0):
            num_total = 0
            den_total = 0
            position = 0
            for user_item, value in col.iteritems():
                if(value > 0):
                    num_total += (similarUsers[position][1] * (value - averageRatings[position]))
                    #num_total += bestUsers[position][1] * value
                    den_total += similarUsers[position][1]
                position += 1
            if den_total != 0:
                predicted_Score = userAverageRating + (num_total / den_total)
                #predicted_Score = num_total / den_total
        itemRatings.append([item, predicted_Score])
        itemPos += 1
    return itemRatings

# Gets the recommendations for a single context
def singleContextRecommendation(data, userID, context):
    
    # Reduce dimensions to remove unnecessary context
    df = data[data['weather'] == context]

    # Converts input data to Users x Items table
    result = df.pivot_table(index="UserID", columns="ItemID", values="Rating").fillna(0)

    # Get user rating information
    try:
        userRatings = result.loc[userID,:]
    except:
        itemRatings = []
        for item in items:
            itemRatings.append([item, 0])
        return itemRatings, 0
    userAverageRating = userRatings.sum() / ((userRatings != 0).sum())

    # Get similar users
    similarUsers = userSimilarity(result, userID, userRatings)

    # Get dataframe of most similar users
    mostSimilarUsers = topUsersDataframe(result, similarUsers)

    # Get average ratings of most similar users
    averageRatings = getAverageRatings(mostSimilarUsers)

    # Get the predicted item ratings
    itemRatings =  getPredictedItemScores(mostSimilarUsers, similarUsers, userAverageRating, userRatings, averageRatings)

    # Return item ratings and the users average score
    return itemRatings, ((userRatings != 0).sum())



        
def clear():
    _ = system('cls')

def displaySignInMenu():
    print("CARS Recommendation System")
    print("==========================")
    print("Please enter your user ID")
    
def displayMainMenu(UserID):
    print("CARS Recommendation System")
    print("Signed in as: " + str(userID))
    print("==========================")
    print("Press 1 to generate recommendations")
    print("Press 2 to update the weather type")
    print("Press 3 to evaluate the system")
    print("Press 4 to sign in as another user")
    print("Press any other key to QUIT")

def displayContextMenu(UserID):
    print("CARS Recommendation System")
    print("Signed in as: " + str(userID))
    print("==========================")
    print("Please enter the current weather state")
    print("Note: Choose from 'sunny', 'cloudy', 'rainy' or 'snowing'")

def displayRecommendations(itemRatings):
    # Convert the dictionary back to a list
    temp = list(map(list, itemRatings.items()))
    sortedList = sorted(temp, key=lambda x: x[1], reverse=True)

    # Put top items in dataframe, merge, and remove unuseful information
    topItems = pd.DataFrame(sortedList[:10], columns=["ItemID", "Rating"])
    items = pd.merge(item_data, topItems, on="ItemID", how="right", sort="False")
    items = items.sort_values("Rating", ascending=False)
    items = items.drop("imageurl", 1); items = items.drop("description", 1)
    items = items.drop("mp3url", 1); items = items.drop("album", 1)
    items = items.drop("category_id", 1); items = items.drop("Rating", 1)
    print("Your recommendations are:")
    print(items)

def displayMAE(mae, precision, recall):
    print("The system has an MAE of: " , mae)
    print("The system has a precision of: ", precision)
    print("The system has a recall of ", recall)

def mainMenu(ratings_data, userID):
    completed = False
    context = getContext(userID)
    while(not completed):
        clear()
        displayMainMenu(userID)
        generate = input("")
        if(generate == '1'):
            clear()
            itemRatings = generateRecommendations(ratings_data, userID, context)
            displayRecommendations(itemRatings)
            x = input("Press any key to continue")
        elif(generate == '2'):
            clear()
            context = getContext(userID)
        elif(generate == '3'):
            clear()
            mae, precision, recall = evaluateSys(ratings_data)
            displayMAE(mae, precision, recall)
            x = input("Press any key to continue")
        elif(generate == '4'):
            clear()
            return True
        else:
            return False

def signInUser():
    signedIn = False
    users = ratings_data.UserID.unique()
    while(not signedIn):  
        displaySignInMenu()
        user = int(input(""))
        clear()
        if(user in users):
            return user
        else:
            print("That user does not exist within the system")

def getContext(userID):
    displayContextMenu(userID)
    complete = False
    while(not complete):
        x = input("")
        if(x == "snowing" or x == "sunny" or x == "cloudy" or x == "rainy"):
            complete = True
        else:
            print("Please enter one of the valid options")
    return x




def evaluateSys(ratings_data):
    # Important imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # Split data into testing data and training data
    train_data, test_data = train_test_split(ratings_data,train_size=0.8)

    # Need to calculate predicted ratings for each user in each context
    contextList = ['sunny', 'cloudy', 'rainy', 'snowing']
    y_true = []
    y_pred = []
    TP = 0
    FP = 0
    FN = 0
    test_data = test_data.sort_values('UserID')
    for user in users:
        for context in contextList:
            recommendations = generateRecommendations(train_data, user, context)
            for index, row in test_data.iterrows():
                if(row['UserID'] == user and row['weather'] == context):
                    currentItem = row['ItemID'] 
                    if(not math.isnan(recommendations[currentItem])):
                        actual_value = row['Rating']
                        recommended_value = round(recommendations[currentItem])
                        y_true.append(actual_value)
                        y_pred.append(recommended_value)

                        # Calculate precision
                        if(actual_value >= 2.5 and recommended_value >= 2.5):
                            TP += 1
                        if(actual_value < 2.5 and recommended_value >= 2.5):
                            FP += 1
                        if(actual_value >= 2.5 and recommended_value < 2.5):
                            FN += 1
                        

    mae = mean_absolute_error(y_true, y_pred)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return mae, precision, recall




# Read in data
clear()
ratings_data = pd.read_csv("ContextualRatings_InCarMusic.csv", index_col=False, delimiter=',', encoding="utf-8-sig")
item_data = pd.read_csv("MusicData_InCarMusic.csv", index_col=False, delimiter=',', encoding="utf-8-sig")
user_data = pd.read_csv("UserData_InCarMusic.csv", index_col=False, delimiter=',', encoding="utf-8-sig")
# Pre-process database to average multiple ratings on same item
ratings_data = ratings_data.groupby(['UserID', 'ItemID', 'weather'], as_index = False)['Rating'].mean()

users = ratings_data.UserID.unique()
items = ratings_data.ItemID.unique()
users.sort()
items.sort()

complete = True
while(complete):
    userID = signInUser()
    complete = mainMenu(ratings_data, userID)
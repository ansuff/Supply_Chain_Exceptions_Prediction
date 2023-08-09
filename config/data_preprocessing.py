import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_name='Training_Test_Data.csv'):
    path_to_file = os.path.join(os.getcwd(),'data',file_name)
    data = pd.read_csv(path_to_file, sep=';', decimal=',')

    return data

def clean_data(df):
    
    df = df.rename(columns={"Loading_meter [ldm]" : "Loading_meter_ldm", 
                                              "Gross_weight [kg]" : "Gross_weight_kg",
                                              "Volume [m3]" : "Volume_m3",
                                              "Handling_unit_quantity [qty]" :"Handling_unit_quantity_qty",
                                              "Billed freight weight [kg]" : "Billed_freight_weight_kg",
                                              "Carrier ID": "Carrier_ID",
                                              "Custom clearance needed" : "Custom_clearance_needed",
                                              "Pickup_timewindow_length [hrs]" : "Pickup_timewindow_length_hrs",
                                              "Delivery_timewindow_length [hrs]" : "Delivery_timewindow_length_hrs",
                                              "Plant Shutdown on pickup or delivery day" : "Plant_Shutdown_on_pickup_or_delivery_day",
                                              "Exception happened 1week ago": "Exception_happened_1week_ago",
                                              "Exception happened 2weeks ago" : "Exception_happened_2weeks_ago",
                                              "Holiday during pickup or delivery day" : "Holiday_during_pickup_or_delivery_day",
                                              "distance cluster" : "distance_cluster",
                                             })
    
    #target variable into categorical
    le = LabelEncoder()
    df['Exception_output'] = le.fit_transform(df['Exception_output'])
    df['Number_of_Stops'] = df['Number_of_Stops'].astype(float)
    df['Weeks_after_project_GoLive'] = df['Weeks_after_project_GoLive'].astype(float)
    df['Pickup_Month'] = df['Pickup_Month'].astype(float)
    df['Pickup_Year'] = df['Pickup_Year'].astype(float)
    df['Custom_clearance_needed'] = df['Custom_clearance_needed'].astype(float)

    #remove missing values
    df = df[df.Pickup_timewindow_length_hrs != "#NAME?"]
    df = df[df.Delivery_timewindow_length_hrs != "#NAME?"]
    df['Pickup_timewindow_length_hrs'] = df['Pickup_timewindow_length_hrs'].astype(float)
    df['Delivery_timewindow_length_hrs'] = df['Delivery_timewindow_length_hrs'].astype(float)

    # drop Transport Order id
    df = df.drop("Transport Order id", axis=1)

    #drop rows with Nan
    df = df.dropna()

    #make all transportation modes upper case (to add 3 ltl to LTL)
    df.loc[:,"Mode_of_Transportation"] = df["Mode_of_Transportation"].str.upper()

    #make all means of transportation lower case
    df.loc[:,"Means_of_transportation"] = df["Means_of_transportation"].str.lower()

    #translate all weekdays in english
    df.loc[df["Pickup_weekday"] == "Montag", "Pickup_weekday"] = "Monday"
    df.loc[df["Pickup_weekday"] == "Dienstag", "Pickup_weekday"] = "Tuesday"
    df.loc[df["Pickup_weekday"] == "Mittwoch", "Pickup_weekday"] = "Wednesday"
    df.loc[df["Pickup_weekday"] == "Donnerstag", "Pickup_weekday"] = "Thursday"
    df.loc[df["Pickup_weekday"] == "Freitag", "Pickup_weekday"] = "Friday"
    df.loc[df["Pickup_weekday"] == "Samstag", "Pickup_weekday"] = "Saturday"
    df.loc[df["Pickup_weekday"] == "Sonntag", "Pickup_weekday"] = "Sunday"
    df.loc[df["Delivery_weekday"] == "Montag", "Delivery_weekday"] = "Monday"
    df.loc[df["Delivery_weekday"] == "Dienstag", "Delivery_weekday"] = "Tuesday"
    df.loc[df["Delivery_weekday"] == "Mittwoch", "Delivery_weekday"] = "Wednesday"
    df.loc[df["Delivery_weekday"] == "Donnerstag", "Delivery_weekday"] = "Thursday"
    df.loc[df["Delivery_weekday"] == "Freitag", "Delivery_weekday"] = "Friday"
    df.loc[df["Delivery_weekday"] == "Samstag", "Delivery_weekday"] = "Saturday"
    df.loc[df["Delivery_weekday"] == "Sonntag", "Delivery_weekday"] = "Sunday"

    #transform distance cluster into categorical variable
    df.loc[df["distance_cluster"] == "0-50 km", "distance_cluster"] = "0"
    df.loc[df["distance_cluster"] == "50-200 km", "distance_cluster"] = "1"
    df.loc[df["distance_cluster"] == "200-500 km", "distance_cluster"] = "2"
    df.loc[df["distance_cluster"] == "500-800 km", "distance_cluster"] = "3"
    df.loc[df["distance_cluster"] == "800-1300 km", "distance_cluster"] = "4"
    df.loc[df["distance_cluster"] == "1300-1800 km", "distance_cluster"] = "5"
    df.loc[df["distance_cluster"] == ">1800 km", "distance_cluster"] = "6"

    #transform distance cluster into categorical variable
    df['distance_cluster'] = df['distance_cluster'].astype(int)

    df[["Mode_of_Transportation", 
                          "Means_of_transportation", 
                          "Pickup_weekday",
                           "Delivery_weekday",
                          ]] = df[["Mode_of_Transportation",
                                                          "Means_of_transportation", 
                                                          "Pickup_weekday",
                                                          "Delivery_weekday",
                                                         ]].apply(lambda x: pd.factorize(x)[0])
    
    #transform all other categorical variables into dummy variables
    df['Number_of_Stops'] = df['Number_of_Stops'].astype(int)
    df['Weeks_after_project_GoLive'] = df['Weeks_after_project_GoLive'].astype(int)
    df['Pickup_Month'] = df['Pickup_Month'].astype(int)
    df['Pickup_Year'] = df['Pickup_Year'].astype(int)
    df['Pickup_timewindow_length_hrs'] = df['Pickup_timewindow_length_hrs'].astype(int)
    df['Delivery_timewindow_length_hrs'] = df['Delivery_timewindow_length_hrs'].astype(int)

    df['Custom_clearance_needed'] = df['Custom_clearance_needed'].astype('boolean')
    df['Carrier_ID'] = le.fit_transform(df['Carrier_ID'])
    df['Consignor_country'] = le.fit_transform(df['Consignor_country'])
    df['Recipient_country'] = le.fit_transform(df['Recipient_country'])
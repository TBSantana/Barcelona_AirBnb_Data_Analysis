#!/usr/bin/env python
# coding: utf-8

# # Data analisys of AirBnB bookings in Barcelona city
# 
# In this notebook we are going to explore data from AirBnb Barcelona dataset http://insideairbnb.com/get-the-data.html in the 2019 year and extract some insights of interest related to prices, neigbourhoods, avalability or room types in the city.
# 
# We also will answer the following questions with statiscal data and visualizations:
#     
# Which neighbourhood are in the city and which are your features?
# 
# Which is the median availability in days per year of each apartment according the district and type of room?
# 
# Which is the occupation of apartments per month in the last two years?
# 
# Which are the prices according the type of room and district in the city?
# 
# To answer these questions i have separate this notebook in the next sections:
# 
# 1. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# 2. [Data Preparation](#Data-Preparation)
# 3. [Analisys and Visualization of Categorical Variables](#Categorical-Variables)
# 4. [Relation between districts and neigbourhoods](#Districts-Neigbourhoods)
# 5. [Data related to Room type](#Data-related-to-Room-type)
# 6. [Data related to availabily of rooms in 2019](#Data-Availability-Rooms)
# 7. [Occupation in the last 2 years: 2018 and 2019 of AirBnb listings in Barcelona](#Ocupation-2018-2019)
# 8. [Data related to 2019 booking prices](#Data-Booking-Prices)
# 9. [Price comparison about three of the most populars districts in Barcelona city: Eixample, Ciutat Vella and Les Corts](#Comparison-Prices)
# 
#     
#     

# In[2]:


#Import libraries, read and show head datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')

#Read reviews dataset (relation between listings and review dates)
df_reviews = pd.read_csv('reviews_.csv',parse_dates=['date'])

#Read neighbourhood dataset (relation between neighbourhoods_groups and neighbourhoods)
df_neighbourhood = pd.read_csv('neighbourhoods.csv')

#Read listings dataset (information about each airBnb listing in Barcelona city)
df = pd.read_csv('listings.csv', parse_dates=['last_review'])


# In[3]:


df_reviews.head()


# In[4]:


df_reviews.shape


# <a id='Exploratory-Data-Analysis'></a>
# ## Exploratory Data Analysis

# In[5]:


df_neighbourhood.head()


# In[6]:


#Dimension of neighbourhoods dataset in number of rows and columns
df_neighbourhood.shape


# In[7]:


#General info about neighbourhood dataset (number of rows not null and type) about each column
df_neighbourhood.info()


# Now we go to explore listings dataset, where we get the most interest information for the analisys.

# In[8]:


df.head()


# In[9]:


#Dimension of listing dataset in number of rows and columns
df.shape


# In[10]:


#Statistical information about numerical variables
df.describe()


# In[11]:


#Histogram for each numerical attribute
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[12]:


#Map of correlation beetwen variables
corrmat = df.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[13]:


#General info about listings dataset (number of rows not null and type) about each column
df.info()


# In[14]:


#Finding out the earliest and latest review date
df['last_review'] = pd.to_datetime(df['last_review'])
oldest_date = df['last_review'].min()
newest_date = df['last_review'].max()
print(oldest_date)
print(newest_date)


# ## Data Preparation

# In[15]:


#Data wrangling

#First will rename neigbourhood_group column to district in both datatsets
df_neighbourhood = df_neighbourhood.rename(columns={"neighbourhood_group": "district"})
df = df.rename(columns={"neighbourhood_group": "district"})

#The are not necessary id columns(id, host_id) neither description columns (name, host_name)
df = df.drop(['id', 'host_id', 'name', 'host_name'], axis=1)

#Print all columns without NA values
no_nulls = set(df.columns[df.isnull().mean()==0])
print(no_nulls)


# We can see columns like name, last_review and rewiews_per_month, have missing values since they have less than 20428 rows informed as non-null in the above info() form. Only rewiews_per_month is an useful column for our analysis in the future (if we need to build a linear regression model for example), then the best option is fill this nulls with the column mean.

# In[16]:


#Function that calculates mean and fill the missing values of the column passed by parameter.
fill_mean = lambda col: col.fillna(col.mean())
df['reviews_per_month'] =fill_mean(df.reviews_per_month)


# In[17]:


df.head()


# <a id='Categorical-Variables'></a>
# ## Analisys and Visualization of Categorical Variables

# Let's go to take a look about categorical variables of interest in our analysis.

# In[18]:


#Show number of listings by district
plt.figure(figsize = (10, 5))
ax = sns.countplot(x='district', data=df, order = df.district.value_counts().sort_values(ascending=False).index)
ax.set_xlabel('Districts', weight='normal', size=15)
ax.set_ylabel('Number of AirBnb', weight='normal', size=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=12)
plt.title('The number of Airbnb in Barcelona', fontsize=18)

plt.show()

df.district.value_counts().sort_values(ascending=False)


# <a id='Districts-Neigbourhoods'></a>
# ## Relation between districts and neigbourhoods
# 
# In Barcelona city exists 10 districts all of them have diferent neigbourhoods, AirBnb listings are distributed around these and each one belong to one neibourhood and district (in dataset called neigbourhood_group).

# In[19]:


#Visualization of bookings in the Barcelona map (according on latitude and longitude columns)
plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.district)
plt.ioff()


# In[20]:


#All Neighbourhoods per district
pd.DataFrame(df_neighbourhood.groupby(["district", "neighbourhood"]).count())


# ### Data related to Room type

# In[21]:


#Show the number of room types 
plt.figure(figsize = (10, 5))
ax = sns.countplot(x='room_type', data=df)
plt.title('Number of Room Types', fontsize=18)

#set the axes
ax.set_xlabel('Type of room', weight='normal', size=15)
ax.set_ylabel('Number of Rooms', weight='normal', size=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size =12)

plt.show()

df.room_type.value_counts()


# In[22]:


#Show number and type of rooms by neighbourhood
plt.figure(figsize = (15, 6))
ax = sns.countplot(x='district', hue='room_type', data=df)

#set the axes
ax.set_xlabel('Districts', weight='normal', size=15)
ax.set_ylabel('Number of Room types', weight='normal', size=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.title('Room type by district', fontsize=18)

plt.show()


# In[113]:


# Barplot function
def show_barplot(col_x, col_y, x_label, y_label, tittle_, ci):
    '''
    INPUT 
        col_x - x column to barplot
        col_y - y column to barplot
        x_label - x label barplot text
        y_label - y label barplot text
        tittle_ - barplot tittle
        ci - Size of confidence intervals to draw around estimated values
        
    OUTPUT
        Displays a barplot of pretty things related to the col_x and col_y.
    '''
    
    #Standard sizes
    plt.figure(figsize = (10, 6))
    #Draw plot
    ax = sns.barplot(col_x, col_y, ci=ci)
    #Set the axes
    ax.set_xlabel(x_label, weight='normal', size=15)
    ax.set_ylabel(y_label, weight='normal', size=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size= 12)
    #Set tittle
    plt.title(tittle_, fontsize=18)
    #Show plot
    plt.show()


# In[120]:


#Show relation beetwen price and district
show_barplot(df.district, df.price, 'Districts', 'Price in $', 'Price according the district', 80)


# In[121]:


#Relation beetwen price and type room
df_price_room_type = pd.DataFrame(df.groupby(['room_type']).mean()['price'].sort_values())
print(df_price_room_type)
#Show price per room type
show_barplot(df.price, df.room_type, 'Price', 'Room type', 'Price according the room type', 80)


# In[26]:


#Highest median prices according the district and the room type
pd.DataFrame(df.groupby(['district','room_type']).mean()['price'].sort_values(ascending=False)).head(15)


# In[27]:


#Lowest median prices according the district and the room type
pd.DataFrame(df.groupby(['district','room_type']).mean()['price'].sort_values()).head(15)


# In[126]:


#Relation beetwen minimum_nights in days and type room
df_price_room_type = df.groupby(['room_type']).mean()['minimum_nights'].sort_values()
print(df_price_room_type)

#Show minimum nights stand in days according the room type
show_barplot(df.room_type, df.minimum_nights, 'Room type', 'Minimum nights', 'Minimum nights according the room type', None)


# <a id='Data-Availability-Rooms'></a>
# ### Data related to availabily of rooms in 2019

# In[29]:


#Relation beetwen district and availability in days per year
df_price_room_type=pd.DataFrame(df.groupby(['district']).mean()['availability_365'].sort_values(ascending=False))
df_price_room_type.availability_365 = df_price_room_type.availability_365.round(1)
#df_price_room_type = df_price_room_type.rename({"availability_365" : "Avalability"})
df_price_room_type


# In[123]:


#Show availability in days according district
show_barplot(df.district, df.availability_365, 'District', 'Days Available', 'Availability according the district',95)


# In[125]:


#Relation beetwen minimum_nights stand and type room
df_price_room_type = df.groupby(['room_type']).mean()['availability_365'].sort_values()
print(df_price_room_type)

#Show availability in days according room type
show_barplot(df.availability_365, df.room_type, 'Days Available', 'Romm type', 'Availability according the room type', None)


# In[32]:


#Lowest median availability per year in days according the neigbourhood and district
pd.DataFrame(df.groupby(['neighbourhood','district']).mean()['availability_365'].sort_values()).head(15)


# In[33]:


#Highest median availability per year in days according the neigbourhood and district
pd.DataFrame(df.groupby(['neighbourhood','district']).mean()['availability_365'].sort_values(ascending=False)).head(15)


# In[34]:


#Highest median availability per year in days according the district and room type
pd.DataFrame(df.groupby(['district','room_type']).mean()['availability_365'].sort_values(ascending=False)).head(15)


# In[35]:


#Lowest median availability per year in days according the district and room type
pd.DataFrame(df.groupby(['district','room_type']).mean()['availability_365'].sort_values()).head(15)


# <a id='Ocupation-2018-2019'></a>
# ## Occupation in the last 2 years: 2018 and 2019 of AirBnb listings in Barcelona
# 
# According to AirBnb data information in your website http://insideairbnb.com/barcelona: "Airbnb guests may leave a review after their stay, and these can be used as an indicator of airbnb activity." 

# In[36]:


# make new dataframe for number of reviews
rev_freq = pd.DataFrame(df_reviews['date'].value_counts().values,
                        index=df_reviews['date'].value_counts().index,
                        columns=['Number of reviews'])

# Select 2019 year
rev_freq_2019 = rev_freq.loc['2019']

# Calculates review per month in selected year
rev_2019_months = rev_freq_2019.resample('M').sum()
rev_2019_months['%'] = (rev_2019_months['Number of reviews']*100)/rev_2019_months['Number of reviews'].sum()

rev_2019_months


# In[37]:


# Select 2018 year
rev_freq_2018 = rev_freq.loc['2018']

# Calculates review per month in selected year
rev_2018_months = rev_freq_2018.resample('M').sum()
rev_2018_months['%'] = (rev_2018_months['Number of reviews']*100)/rev_2018_months['Number of reviews'].sum()

rev_2018_months


# In[38]:


#Show Reviews per mont in 2019
fig1 = plt.figure(figsize=(10, 5))
ax = fig1.add_subplot(1, 1, 1, aspect='auto')
sns.barplot(x=rev_2018_months.index.month_name(), y=rev_2018_months['%'])

# Set axis labels
ax.set_xlabel('Months', weight='normal', size=12)
ax.set_ylabel('% Reviews', weight='normal', size=12)
plt.title('Ocupation per month in 2018', fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=13)
plt.show()


# In[39]:


#Show Reviews per mont in 2019
fig1 = plt.figure(figsize=(10, 5))
ax = fig1.add_subplot(1, 1, 1, aspect='auto')
sns.barplot(x=rev_2019_months.index.month_name(), y=rev_2019_months['%'])

# Set axis labels
ax.set_xlabel('Months', weight='normal', size=15)
ax.set_ylabel('% Reviews', weight='normal', size=15)
plt.title('Ocupation per month in 2019', fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=12)
plt.show()


# <a id='Data-Booking-Prices'></a>
# ## Data related to 2019 booking prices

# In[40]:


#Highest median prices in dollars according the neigbourhood and district
pd.DataFrame(df.groupby(['neighbourhood','district']).mean()['price'].sort_values(ascending=False)).head(15)


# In[41]:


#Lowest median prices according the neigbourhood and district
pd.DataFrame(df.groupby(['neighbourhood','district']).mean()['price'].sort_values()).head(15)


# <a id='Comparison-Prices'></a>
# ## Price comparison about three of the most populars districts in Barcelona city: Eixample, Ciutat Vella and Les Corts

# In[42]:


#Function that calculates the price median of each room type in the district and shows the plot visualization
def calculate_price_mean_and_plot(filtered_df, plot=True):
    '''
    INPUT 
        filtered_df - a dataframe filtered by the district column
        plot - bool providing whether or not you want a plot back
        
    OUTPUT
        df_price_room_type - a dataframe with the price per room type in the district filtered
        Displays a barplot of pretty things related to the district and type of room columns.
    '''

    #Get the name of the district in filtered dataset
    distric_name = filtered_df['district'].iloc[0]
    
    #Agroupation beetwen price and type room
    df_price_room_type = pd.DataFrame(filtered_df.groupby(['room_type']).mean()['price'].sort_values(ascending=False))

    #Plot the agroupation
    if plot:
        #Show price per room type
        plt.figure(figsize = (10, 5))
        ax = sns.barplot(filtered_df.price, filtered_df.room_type, ci = None, palette = 'magma')
        ax.set_xlabel('Price', weight='normal', size=12)
        ax.set_ylabel('Room type', weight='normal', size=12)
        plt.title('Price according the room type in {}'.format(distric_name), fontsize=18)

        plt.show()
        
    return df_price_room_type
    


# ### Les Corts district:

# In[43]:


#Call to calculate_price_mean_and_plot function with Les Corts district data 
calculate_price_mean_and_plot(df[df.district == 'Les Corts'])


# In[44]:


#Median price booking table of each neigbourhood of Les Corts district per room type
pd.DataFrame(df.query("district == 'Les Corts'").groupby(['district','neighbourhood','room_type']).mean()['price'])


# ### Eixample district:

# In[45]:


#Call to calculate_price_mean_and_plot function with Eixample district data 
calculate_price_mean_and_plot(df[df.district == 'Eixample'])


# In[46]:


#Median price booking table of each neigbourhood of Eixample district per room type
pd.DataFrame(df.query("district == 'Eixample'").groupby(['district','neighbourhood','room_type']).mean()['price'])


# ### Ciutat Vella district:

# In[47]:


#Call to calculate_price_mean_and_plot function with Ciutat Vella district data
calculate_price_mean_and_plot(df[df.district == 'Ciutat Vella'])


# In[48]:


#Median price booking table of each neigbourhood of Ciutat Vella district per room type
pd.DataFrame(df.query("district == 'Ciutat Vella'").groupby(['district','neighbourhood','room_type']).mean()['price'])


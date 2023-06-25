import sqlite3
import pandas as pd

# Establish a DB connection
con = sqlite3.connect("Carsharing.db", isolation_level=None)
cur = con.cursor()

# Create CarSharing Table
car_sharing = pd.read_csv("CarSharing.csv")
car_sharing.to_sql("CarSharing", con, if_exists='replace', index=False)

#Create Backup table and copy data
cur.execute(
    "CREATE TABLE IF NOT EXISTS Backup (Id INTEGER NOT NULL, Timestamp INTEGER NOT NULL, Season TEXT NOT NULL, Holiday TEXT, Working_Day TEXT, Weather TEXT, Temp REAL, Temp_feel REAL, Humidity INTEGER, Windspeed REAL, Demand REAL NOT NULL)"
)

cur.execute("INSERT INTO Backup SELECT * FROM CarSharing")

#Adds temp_category column
cur.execute("ALTER TABLE CarSharing ADD COLUMN temp_category TEXT")
cur.execute("""
    UPDATE CarSharing
    SET temp_category = CASE
        WHEN Temp_Feel < 10 THEN 'Cold'
        WHEN Temp_Feel <= 25 THEN 'Mild'
        WHEN Temp_Feel > 25 THEN 'Hot'
    END
    WHERE Temp_Feel IS NOT NULL
""")

#Create temperature table and restructure columns in CarSharing
cur.executescript("""
    BEGIN;
    CREATE TABLE temperature AS SELECT temp, temp_feel, temp_category FROM CarSharing;
    ALTER TABLE CarSharing DROP COLUMN temp;
    ALTER TABLE CarSharing DROP COLUMN temp_feel;
    COMMIT;
""")

#Add weather_code column
cur.execute("ALTER TABLE CarSharing ADD COLUMN weather_code TEXT")
cur.execute("""
    UPDATE CarSharing
    SET weather_code = CASE
        WHEN weather == 'Clear or partly cloudy' THEN 1
        WHEN weather == 'Light snow or rain' THEN 2
        WHEN weather == 'Mist' THEN 3
        WHEN weather == 'heavy rain/ice pellets/snow + fog' THEN 4
    END
    WHERE weather IS NOT NULL
""")

#Create weather table and restructure columns in CarSharing
cur.executescript("""
    BEGIN;
    CREATE TABLE weather AS SELECT weather, weather_code FROM CarSharing;
    ALTER TABLE CarSharing DROP COLUMN weather;
    COMMIT;
""")

#Create time table

cur.execute(
    "CREATE TABLE time AS SELECT timestamp, strftime('%H',timestamp) as 'hour', strftime('%w',timestamp) as 'weekday name', strftime('%m',timestamp) as 'month name' from CarSharing"
)

#SELECT  timestamp, max(demand) FROM CarSharing WHERE (timestamp BETWEEN "2017-01-01 00:00:00 " AND "2017-12-31 23:59:59 " )
# the query to fetch the date and time we had the highest demand rate in 2017.
cur.execute('''select max(demand) from CarSharing where timestamp like '%2017%' ;''')
print(cur.fetchone()) 
cur.execute('''create table seasonalavg as select 
case cast (strftime('%w', timestamp) as integer)
  when 0 then 'Sunday'
  when 1 then 'Monday'
  when 2 then 'Tuesday'
  when 3 then 'Wednesday'
  when 4 then 'Thursday'
  when 5 then 'Friday'
  else 'Saturday' end as weekdayname,
  case cast (strftime('%m', date(timestamp)) as integer)
  when '01' then 'January'
  when '02' then 'February'
  when '03' then 'March'
  when '04' then 'April'
  when '05' then 'May'
  when '06' then 'June'
  when '07' then 'July'
  when '08' then 'August'
  when '09' then 'September'
  when '10' then 'October'
  when '11' then 'November'
  else 'December' end as month,season,demand, avg(demand) as demand_average from CarSharing where timestamp like '%2017%' 
  group by weekdayname, month, season;''')
cur.execute('select weekdayname, month, season from seasonalavg where demand_average = (select max(demand_average) from seasonalavg) ;')
print(cur.fetchall())
cur.execute('select weekdayname, month, season from seasonalavg where demand_average = (select min(demand_average) from seasonalavg) ;')
print(cur.fetchall())



cur.execute(''' create table sundays_demand as select
strftime('%H',timestamp) as "Hour",
case cast (strftime('%w', timestamp) as integer)
  when 0 then 'Sunday'
  when 1 then 'Monday'
  when 2 then 'Tuesday'
  when 3 then 'Wednesday'
  when 4 then 'Thursday'
  when 5 then 'Friday'
  else 'Saturday' end as weekday,
  case cast (strftime('%m', date(timestamp)) as integer)
  when '01' then 'January'
  when '02' then 'February'
  when '03' then 'March'
  when '04' then 'April'
  when '05' then 'May'
  when '06' then 'June'
  when '07' then 'July'
  when '08' then 'August'
  when '09' then 'September'
  when '10' then 'October'
  when '11' then 'November'
  else 'December' end as monthname,season,demand from CarSharing where timestamp like '%2017%' and weekday = 'Sunday';''')


cur.execute(''' create table mondays_demand as select
strftime('%H',timestamp) as "Hour",
case cast (strftime('%w', timestamp) as integer)
  when 0 then 'Sunday'
  when 1 then 'Monday'
  when 2 then 'Tuesday'
  when 3 then 'Wednesday'
  when 4 then 'Thursday'
  when 5 then 'Friday'
  else 'Saturday' end as weekday,
  case cast (strftime('%m', date(timestamp)) as integer)
  when '01' then 'January'
  when '02' then 'February'
  when '03' then 'March'
  when '04' then 'April'
  when '05' then 'May'
  when '06' then 'June'
  when '07' then 'July'
  when '08' then 'August'
  when '09' then 'September'
  when '10' then 'October'
  when '11' then 'November'
  else 'December' end as monthname,season,demand from CarSharing where timestamp like '%2017%' and weekday = 'Monday';''')

cur.execute('select * from sundays_demand;')
print(cur.fetchall())
cur.execute('select * from mondays_demand;')
print(cur.fetchall())
cur.execute('''select max(temp_category) from CarSharing where timestamp like '%2017%' ;''')
print(cur.fetchall())
cur.execute('''select max(weather) from CarSharing where timestamp like '%2017%' ;''')
print(cur.fetchall())
cur.execute('''select case cast (strftime('%m', date(timestamp)) as integer)
  when '01' then 'January'
  when '02' then 'February'
  when '03' then 'March'
  when '04' then 'April'
  when '05' then 'May'
  when '06' then 'June'
  when '07' then 'July'
  when '08' then 'August'
  when '09' then 'September'
  when '10' then 'October'
  when '11' then 'November'
  else 'December' end as monthname, avg(windspeed), max(windspeed), min(windspeed) from CARSHARING 
  where timestamp like '%2017%' group by monthname order by monthname;''')
print(cur.fetchall())


cur.execute('''select case cast (strftime('%m', date(timestamp)) as integer)
  when '01' then 'January'
  when '02' then 'February'
  when '03' then 'March'
  when '04' then 'April'
  when '05' then 'May'
  when '06' then 'June'
  when '07' then 'July'
  when '08' then 'August'
  when '09' then 'September'
  when '10' then 'October'
  when '11' then 'November'
  else 'December' end as monthname, avg(humidity), max(humidity), min(humidity) from CARSHARING 
  where timestamp like '%2017%' group by monthname order by monthname;''')


print(cur.fetchall())


cur.execute('''select temp_category, avg(demand) from CARSHARING where timestamp like '%2017%' group by temp_category order by avg(demand) desc;''')
print(cur.fetchall())
cur.close()


#Data Analytics 
import sqlite3
import pandas as pd
conn1 = sqlite3.connect("Car_sharing.db", isolation_level=None)
data = pd.read_sql_query("SELECT * FROM CarSharing", conn1)
data.to_csv('CarSharing.csv', index=False)
data_frame = pd.read_csv("CarSharing.csv",header=0, na_values="?" )
data_frame2 = pd.DataFrame(data_frame)
data_frame3=data_frame2.drop_duplicates()
data_frame4=data_frame3.dropna()
print(data_frame4.info())

#2
import statsmodels.api as sm
from scipy.stats import f_oneway
dataframe = pd.read_csv("CarSharing.csv",header=0, na_values="?" )
dataframe2 = pd.DataFrame(dataframe)
dataframe3=dataframe2.drop_duplicates()
dataframe4=dataframe3.dropna()
variable1 = dataframe4['demand']
variable2 = dataframe4[['windspeed','temp', 'temp_feel','humidity']]
#add constant to predictor variables
x = sm.add_constant(variable2)
#fit linear regression model
model = sm.OLS(variable1, x).fit()
#view model summary
print(model.summary())
falldata=(dataframe4['demand'].where(dataframe4['season'] == "falldata"))
falldataseasondata = pd.DataFrame(falldata).dropna()
winterdata=(dataframe4['demand'].where(dataframe4['season'] == "winterdata"))
winterdataseasondata = pd.DataFrame(winterdata).dropna()
springdata=(dataframe4['demand'].where(dataframe4['season'] == "springdata"))
springdataseasondata = pd.DataFrame(springdata).dropna()
summerdata=(dataframe4['demand'].where(dataframe4['season'] == "summerdata"))
summerdataseasondata = pd.DataFrame(summerdata).dropna()
print(f_oneway(falldataseasondata,winterdataseasondata, springdataseasondata, summerdataseasondata))
yes=(dataframe4['demand'].where(dataframe4['holiday'] == "Yes"))
yes_holiday = pd.DataFrame(yes).dropna()
no=(dataframe4['demand'].where(dataframe4['holiday'] == "No"))
no_holiday = pd.DataFrame(no).dropna()
b=f_oneway(yes_holiday, no_holiday)
print(b)

yesday=(dataframe4['demand'].where(dataframe4['workingday'] == "Yes"))
yesday_workingday = pd.DataFrame(yesday).dropna()
noday=(dataframe4['demand'].where(dataframe4['workingday'] == "No"))
noday_workingday = pd.DataFrame(noday).dropna()
c=f_oneway(yesday_workingday, noday_workingday)
print(c)
clear=(dataframe4['demand'].where(dataframe4['weather'] == "Clear or partly cloudy"))
clearweatherdata = pd.DataFrame(clear).dropna()
mist=(dataframe4['demand'].where(dataframe4['weather'] == "Mist"))
mistweatherdata = pd.DataFrame(mist).dropna()
snow=(dataframe4['demand'].where(dataframe4['weather'] == "Light snow or rain"))
snowweatherdata = pd.DataFrame(snow).dropna()
rain=(dataframe4['demand'].where(dataframe4['weather'] == "heavy rain/ice pellets/snow + fog"))
rainweatherdata = pd.DataFrame(rain).dropna()
d=f_oneway(clearweatherdata, mistweatherdata, snowweatherdata,rainweatherdata)
print(d)

#3
import matplotlib.pyplot as plt
data = pd.read_csv("CarSharing.csv", parse_dates=['timestamp'], index_col='timestamp')
# Draw Plot
def plot_df(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(data, x=data.index, y=data.temp, title='Temp Seasonal and Cyclic Pattern.')
plot_df(data, x=data.index, y=data.humidity, title='Humidity Seasonal and Cyclic Pattern.')
plot_df(data, x=data.index, y=data.windspeed, title='Windspeed Seasonal and Cyclic Pattern.')
plot_df(data, x=data.index, y=data.demand, title='Demand Seasonal and Cyclic Pattern.')

#4
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
series = pd.DataFrame(data)
result = adfuller(series.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
#P value is greater than 0.05, since the time series is not stationery, we need to do differencing to make it stationery.
result = adfuller(series.diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
#Time series is now stationery after one differencing
df=pd.DataFrame(series.values,columns=['value'])
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.diff().value); axes[0].set_title('1st Order Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.diff().value.dropna(), ax=axes[1])
plt.show()
#we can see from the PACF plot, lag=0 is significant,we therefore set p=1

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
# Import data
df=pd.DataFrame(series.values,columns=['value'])
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.diff().value); axes[0].set_title('Original time series')
axes[1].set(ylim=(0,1.2))
plot_acf(df.diff().value.dropna(), ax=axes[1])
plt.show()
#we can see from the ACF plot that lag=0 is significant, so we set q=1

# Preparing the training and test data
X = series.values
size = int(len(X) * 0.7)
train, test = df.loc[0:size,:], df.loc[size:len(X),:]
model = ARIMA(train.value, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

X = series.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#5
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
dataframe = pd.DataFrame(data).dropna()
val1=dataframe['temp']
val2=dataframe['demand']
x=np.array(val1).reshape(-1,1)
y=np.array(val2).reshape(-1,1)
x_train, x_test, y_train,y_test = train_test_split(np.array(x),np.array(y))
cat_result=RandomForestRegressor(random_state=18)
cat_result.fit(x_train,y_train.ravel())
score=(cat_result.score(x_test,y_test))
print("model is accurate by" +str(score))
result=(cat_result.predict(x_train))
print("Demand Predicted is {}%:".format(result))
mean_sqr_error=(mean_squared_error(y_train,result))
print("mean score is {}%:".format(mean_sqr_error))

#6
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#to confirm average demand in table CarSharing Table
cur.execute('select avg(demand) from CarSharing;')
print(cur.fetchall())
#Average demand in CarSharing Table is 4.45
#Demand rate that is greater than 4.45 will be categorised as Label 1
#Demand rate that is less than 4.45 will be categorised as Label 2
cur.close()
data = pd.read_csv('CarSharing.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
df = pd.DataFrame(data)
df["label"] = np.nan
df.loc[df["demand"] <= 4.45, "label"] = 1
df.loc[df["demand"] > 4.45, "label"] = 2
X=df['demand']
Y=df['label']
x=np.array(X).reshape(-1,1)
y=np.array(Y).reshape(-1,1).ravel()

x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=23)
clf = KNeighborsClassifier(n_neighbors=3)
clf2=SVC(kernel='linear',C=3) 
clf3=RandomForestClassifier()#usuallyaccurate
clf.fit(x_train,y_train)
clf2.fit(x_train,y_train)
clf3.fit(x_train,y_train)
print("KNeighborsClassifier accuracy is "+str(clf.score(x_test,y_test)))
print("SVC accuracy is "+str(clf2.score(x_test,y_test)))
print("Random Forest accuracy is "+str(clf3.score(x_test,y_test)))
#predicting modelling below
print(clf.predict(x_test))##predicting 30 percent of data
print(clf2.predict(x_test))
print(clf3.predict(x_test))

#7
from sklearn.cluster import KMeans
#single csv file with only temp column used
dataframe = pd.read_csv("tempdata.csv",header=0, na_values="?" )
df = pd.DataFrame(dataframe).dropna()
data=np.array(df).reshape(-1, 1)

inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
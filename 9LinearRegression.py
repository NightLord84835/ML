import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/temperatures.csv')

df.head(10)

x = df['YEAR']
y = df['ANNUAL']

plt.title("Temp plot of INDIA")
plt.xlabel("Year")
plt.ylabel("Annual Average Temp")
plt.scatter(x, y)

x = x.reshape(117,1)

x.shape

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y)
print("Slope:       ", reg.coef_)
print("Y-intercept: ", reg.intercept_)

pred = reg.predict(x)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(mean_absolute_error(y, pred))
print(mean_squared_error(y, pred))
print(r2_score(y, pred))

plt.title("Temp plot of INDIA")
plt.xlabel("Year")
plt.ylabel("Annual Average Temp")
plt.scatter(x, y, label='actual', color='r', marker='.')
plt.plot(x, pred, label='predicted', color='b')

sns.regplot(x='YEAR', y='ANNUAL', data=df)

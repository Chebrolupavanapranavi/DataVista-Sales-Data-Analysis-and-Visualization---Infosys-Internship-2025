import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

x = [1,2,3,4,5,6,7]
y = [20,40,30,40,50,30]

df = pd.DataFrame({"Days":x, "No of people": y})
df.head(7)
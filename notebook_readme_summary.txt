=== Adult Income Prediction System/Adult_Income_Prediction_System_DT.ipynb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
print(df["Bachelors"].value_counts())
print(df["State-gov"].value_counts())
print(df["Never-married"].value_counts())
print(df["Occupation"].value_counts())
print(df["Relationship"].value_counts())
print(df["Race"].value_counts())
print(df["Gender"].value_counts())
print(df["Native Country"].value_counts())
print(df["Income"].value_counts())
    sns.lineplot(data=df, x=col, y='Income', ax=axs[i//4, i%4])
    sns.scatterplot(data=df, x=col, y='Income', ax=axs[i//4, i%4])
#df1 = df2.fillna(df.columns.mean()) 
df1 = pd.DataFrame()
    print('Lower Bound for', column, ':', lower_bound) 
    print('Upper Bound for', column, ':', upper_bound) 
    print("-------------")
    df1[column] = d[(d[column] >= lower_bound) & (d[column] <= upper_bound)][column]
plt.bar(df1.columns, df1.isnull().sum())
imputer=SimpleImputer(strategy="median")
imputer.fit(df1)
df1=imputer.transform(df1)
print('missing value:',df1.isnull().any(axis=1).sum())
    plt.boxplot(df1[column], vert=False)
df1.columns
from sklearn.preprocessing import LabelEncoder  
encoder = LabelEncoder()
#for col in columns_to_encode:
d3=pd.merge(d2, df1, left_index=True, right_index=True)
df1.drop(columns=['Capital Loss','Salary'], inplace=True) 
from sklearn.preprocessing import OneHotEncoder
columns_to_encode = [' State-gov', ' Bachelors', ' Never-married',
onehot_encoder = OneHotEncoder(sparse_output=False)
df3=pd.merge(df2, df1, left_index=True, right_index=True)
print(df3["Incom_enc"].value_counts())
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df3)
X = df4.loc[:, df4.columns != 'Income_enc']
print("Training data distribution (after SMOTE):")
print(pd.Series(Y_OverSmote).value_counts())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_OverSmote, Y_OverSmote, test_size = 0.3, random_state=42)
print('Training set score: {:.4f}'.format(model_gini.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model_gini.score(X_test, y_test)))
model_entropy = DecisionTreeClassifier(criterion='entropy')
print('Training set score: {:.4f}'.format(model_entropy.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model_entropy.score(X_test, y_test)))
model = GridSearchCV(clf, tree_params, cv = 10, scoring='accuracy')
print('Training set score: {:.4f}'.format(model_optimal.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model_optimal.score(X_test, y_test)))
print('Confusion matrix\n\n', cm)
print(classification_report(y_test, y_pred_en_op))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_en_op), display_labels = [0, 1])
print(comparison_df.head(50))

=== Adult Income Prediction System/Adult_Income_Prediction_System_KNN.ipynb
print(df[" Bachelors"].value_counts())
print(df[" State-gov"].value_counts())
print(df[" Never-married"].value_counts())
print(df["Occupation"].value_counts())
print(df["Relationship"].value_counts())
print(df["Race"].value_counts())
print(df["Gender"].value_counts())
print(df["Native Country"].value_counts())
print(df["Income"].value_counts())
#df1 = df2.fillna(df.columns.mean()) 
df1 = pd.DataFrame()
    print('Lower Bound for', column, ':', lower_bound) 
    print('Upper Bound for', column, ':', upper_bound) 
    print("-------------")
    df1[column] = d[(d[column] >= lower_bound) & (d[column] <= upper_bound)][column]
imputer=SimpleImputer(strategy="median")
imputer.fit(df1)
df1=imputer.transform(df1)
df1.describe()
plt.boxplot(df1['Age'], vert=False)
df1.columns
from sklearn.preprocessing import LabelEncoder  
encoder = LabelEncoder()
#for col in columns_to_encode:
from sklearn.preprocessing import OneHotEncoder
columns_to_encode = [' State-gov', ' Bachelors', ' Never-married',
onehot_encoder = OneHotEncoder(sparse_output=False)
df3=pd.merge(df2, df1, left_index=True, right_index=True)
print(df3["Incom_enc"].value_counts())
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df3)
X = df4.loc[:, df4.columns != 'Income_enc']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, pred1)
print('Accuracy = ', accuracy)
print(confusion_matrix(y_test, pred1))
print(classification_report(y_test, pred1))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred1), display_labels = [0, 1])
accuracy2 = accuracy_score(y_test, pred2)
print('Accuracy = ', accuracy2)
print(confusion_matrix(y_test, pred2))
print(classification_report(y_test, pred2))
print("Minimum error:",min(error_rate),"at K =",error_rate.index(min(error_rate)))
accuracy3 = accuracy_score(y_test, pred3)
print('Accuracy = ', accuracy3)
print(confusion_matrix(y_test, pred3))
print(classification_report(y_test, pred3))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred3), display_labels = [0, 1])

=== Bank Campaign Success Predictor/bank_marketing_dataset.ipynb
from sklearn.preprocessing import LabelEncoder
# apply the LabelEncoder preprocessing on the <housing> column:
L1=LabelEncoder();
# apply the LabelEncoder preprocessing on the <loan> column:
L2=LabelEncoder();
# apply the LabelEncoder preprocessing on the <y> column:
L3=LabelEncoder();
# apply the LabelEncoder preprocessing on the <default> column:
L4=LabelEncoder();
df = pd.get_dummies(df , columns = ['job'] , dtype = int)
df = pd.get_dummies(df , columns = ['marital'] , dtype = int)
df = pd.get_dummies(df , columns = ['education'] , dtype = int)
df = pd.get_dummies(df , columns = ['contact'] , dtype = int)
df = pd.get_dummies(df , columns = ['poutcome'] , dtype = int)
df = pd.get_dummies(df , columns = ['month'] , dtype = int)
#         s = SimpleImuter(missing_values = nan , strategy = "the wanted strategy like <mean , max , min , .....>")
#         s = SimpleImputer(strategy = "mean")
y = df[['Y']]
X = df[['age', 'poutcome_success', 'day', 'campaign', 'Loan', 'previous',
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
print('accuracy score (training)', clf.score(x_train,y_train))
print('accuracy score (test)', clf.score(x_test,y_test))
clf2 = LogisticRegression(max_iter = 1000000 , C=20,  penalty='l1' , solver='liblinear').fit(x_train,y_train)
print('accuracy score (training)', clf2.score(x_train,y_train))
print('accuracy score (test)', clf2.score(x_test,y_test))
clf2 = LogisticRegression(max_iter = 1000000 , C=50,  penalty='l1' , solver='liblinear').fit(x_train,y_train)
print('accuracy score (training)', clf2.score(x_train,y_train))
print('accuracy score (test)', clf2.score(x_test,y_test))
model = SVC()
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = SVC(C = 1.5)
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = SVC(C = 20)
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = DecisionTreeClassifier()
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = DecisionTreeClassifier(ccp_alpha = 2)
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = DecisionTreeClassifier(ccp_alpha = 3)
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = DecisionTreeClassifier(min_samples_split=100 ,max_features=20 )
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
model = DecisionTreeClassifier(min_samples_split=200 ,max_features=30 )
print(model.score(x_train ,y_train))
print(model.score(x_test ,y_test))
print('Support vector machine classifier (linear kernel , C = 1)\n' , confusion)
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
print('ACC is {}'.format(accuracy_score(y_test , y_predict)))
print('Precision is {:.2f}'.format(precision_score(y_test , y_predict)))
print('recall is {:.2f}'.format(recall_score(y_test , y_predict)))
print('f1 is {:.2f}'.format(f1_score(y_test , y_predict)))
print(classification_report(y_test,y_predict))

=== Bank Term Deposit Prediction System/Bank_Term_Deposit_Prediction_System.ipynb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
print(df["job"].value_counts())
print(df["marital"].value_counts())
print(df["education"].value_counts())
print(df["default"].value_counts())
print(df["housing"].value_counts())
print(df["loan"].value_counts())
print(df["contact"].value_counts())
print(df["month"].value_counts())
print(df["poutcome"].value_counts())
print(df["Target"].value_counts())
    sns.lineplot(data=df, x=col, y='Target', ax=axs[i//4, i%4])
    sns.scatterplot(data=df, x=col, y='Target', ax=axs[i//4, i%4])
#df1 = df2.fillna(df.columns.mean()) 
df1 = pd.DataFrame()
    print('Lower Bound for', column, ':', lower_bound) 
    print('Upper Bound for', column, ':', upper_bound) 
    print("-------------")
    df1[column] = d[(d[column] >= lower_bound) & (d[column] <= upper_bound)][column]
plt.bar(df1.columns, df1.isnull().sum())
imputer=SimpleImputer(strategy="median")
imputer.fit(df1)
df1=imputer.transform(df1)
print('missing value:',df1.isnull().any(axis=1).sum())
    plt.boxplot(df1[column], vert=False)
df1.columns
from sklearn.preprocessing import LabelEncoder  
encoder = LabelEncoder()
#for col in columns_to_encode:
d3=pd.merge(d2, df1, left_index=True, right_index=True)
df1.drop(columns=['day','age', 'campaign'], inplace=True) 
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
df3=pd.merge(df2, df1, left_index=True, right_index=True)
print(df3["Target_enc"].value_counts())
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df3)
from sklearn.model_selection import train_test_split
X = df4.loc[:, df4.columns != 'Target_enc']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)
model = GridSearchCV(clf, tree_params, cv = 10, scoring='accuracy')
print('Training set score: {:.4f}'.format(model_optimal.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model_optimal.score(X_test, y_test)))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_dt = accuracy_score(y_test, pred_dt)
print('Accuracy = ', accuracy_dt)
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred_dt), display_labels=[0, 1])
print(comparison_df.head(50))
print("Minimum error:",min(error_rate),"at K =",error_rate.index(min(error_rate)))
from sklearn.metrics import accuracy_score 
accuracy3 = accuracy_score(y_test, pred3)
print('Accuracy = ', accuracy3)
print(confusion_matrix(y_test, pred3))
print(classification_report(y_test, pred3))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred3), display_labels = [0, 1])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_rf = accuracy_score(y_test, pred_rf)
print('Random Forest Accuracy = ', accuracy_rf)
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))
print("عدد العينات في كل Cluster:")
print(df4['cluster'].value_counts().sort_index())
from sklearn.metrics import silhouette_score
silhouette_score(df4, clusters)
from sklearn.metrics import silhouette_score
X = df4[numeric_cols]
score = silhouette_score(X, df4['cluster'])
print(f"Silhouette Score: {score:.4f}")
from sklearn.preprocessing import StandardScaler
print(f"عدد الكلاسترز: {n_clusters}")
from sklearn.metrics import silhouette_score
score = silhouette_score(df4, model_DBSCAN.labels_)
print(f"Silhouette Score: {score:.4f}")

=== California Housing Price Predictor/housing_DBSCAN.ipynb
sns.scatterplot(data=df, x='longitude', y='latitude')
X =  lat_long.to_numpy()
model = DBSCAN(min_samples=15, eps=0.2).fit(X)
sns.scatterplot(data=df, x='longitude', y='latitude', hue=df.Cluster, palette='tab20')
from sklearn.metrics import silhouette_score
silhouette_score(X, model.labels_)
    dbscan_cluster_model = DBSCAN(eps=eps, min_samples=num_samples).fit(X)
      print(f"Combination {c} on iteration {i+1} of {N} has {num_clusters} clusters. Moving on")
    scores.append(silhouette_score(X, labels))
    print(f"Index: {i}, Score: {scores[-1]}, Labels: {all_labels_list[-1]}, NumClusters: {num_clusters}")
sns.scatterplot(data=df, x='longitude', y='latitude', hue=df.Cluster, palette='tab20')
model = DBSCAN(min_samples=18, eps=0.33).fit(X)
silhouette_score(X, model.labels_)
df['Cluster2'] = model.labels_
df['Cluster2'].value_counts()
sns.scatterplot(data=df, x='longitude', y='latitude', hue=model.labels_, palette='tab20')
sns.scatterplot(data=df, x=df.total_rooms, y=df.housing_median_age)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X2)
sns.scatterplot(data=X2_scaled, x=X2_scaled.total_rooms, y=X2_scaled.housing_median_age)
model = DBSCAN(eps=1, min_samples=2).fit(X2_scaled)
silhouette_score(X2_scaled, model.labels_)
sns.scatterplot(data=df, x=df.total_rooms, y=df.housing_median_age, hue=model.labels_, palette='tab20')
from sklearn.metrics import silhouette_score
def silhouette_scorer(estimator, X):
    return silhouette_score(X, labels)
grid = GridSearchCV(dbscan, param_grid, scoring=silhouette_scorer, cv=[(slice(None), slice(None))], refit=True)
print("أفضل معلمات:", grid.best_params_)
print("أفضل قيمة مقياس silhouette:", grid.best_score_)
print("التسميات الخاصة بأفضل نموذج:", grid.best_estimator_.labels_)
 model = DBSCAN(eps=0.7631578947368421, min_samples=15).fit(X2_scaled)
silhouette_score(X2_scaled, model.labels_)
 model = DBSCAN(eps=0.7631578947368421, min_samples=15).fit(X)
silhouette_score(X, model.labels_)

=== Course np & pd/Course np & pd.ipynb
print(l,type(l),l.dtype)
print(t,type(t),t.dtype)
print(a.ndim,a.dtype)
print(b.ndim,b.dtype)
print(a[1][0],a[1,0])# نفس الشي 
print(b[0][1][2],b[1,0,0])
print("-----------------")
print(b.shape,b.shape[0],b.shape[1],b.shape[2])
print("-----------------")
print(a.size,a.nbytes)# يوجد العديد من التوابع 
print("-----------------")
print(c,c.ndim)
print(ones," and ",zero)
print("-----------------")
print(numbers)
print("-----------------")
print(randomNumbers)
print(np.random.randint(20))
print(np.random.rand(2,5))
print(np.random.randn(5))
print(np.random.random(5))
print("-----------------")
print(reshape)
print(reshape.ndim,reshape.shape)
print(numbers)
print(numbers[4:8])
print(numbers[:8])
print(numbers[4:])
print(numbers[::-1])
print(numbers[-1::])
print("jumb: ",numbers[::2]) # يقفز خطوتين 
print("jumb: ",numbers[::-2]) # يقفز خطوتين لكن يعكس المصفوفه
print(numbers == 5)
print("The index of 5 is: ",int(idx)+1)
print(a)
print("forth row:",a[4,:])
print("3&4 row:",a[3:5,:])
print("reverse:",a[::-1,::-1])
print("2 col & all row:",a[:,1])
print(la.inv(a))
print(a)
print("sort col: ")
print(np.sort(a,axis=0)) # a.sort(a,axis=0)
print("sort row: ")
print(np.sort(a,axis=1)) #a.sort(a,axis=1)
print(a)
print(a[[1,5,9]])
print(a[[True,True,False,True,False,True,True,False,True,False]])
print(a[a<8],a[a==8],a[(a<8)&(a>4)])
print(a)
print(a)
print(b)
print(arr1)
arr2 = np.round(np.random.rand(2,2)*10)
print(arr2)
arr3 = np.hstack((arr1,arr2))
print(arr3)
print(arr4)
print(arr5)
print(arr6)
print(pd.__version__)
print(data.dtype,type(data))
print(data.values,data.index)
print(data['a'])
print(data['a':4])
print("dict1: ",marks_dict)
print("series: \n",marks)
print("dict2: ",marks_dict2)
print("series: \n",marks2)
print(marks['A+':'B'])
print(marks[0:5])
print(dataframe)
print(dataframe.T)
print(dataframe[0:4])
print("marks: ",dataframe['marks']['A'],dataframe.values[0,0])
print("valuse:\n",dataframe.values)
print("insdex: ",dataframe.index)
print("columns: ",dataframe.columns)
print(dataframe)
print(dataframe)
print(dataframe)
print(g)
print(d)
print(d.index)
print(d)
print(d[1:5])
print("loc:")
print(d.loc[1:3])
print("iloc:")
print(d.iloc[0:2])
print(dataframe.iloc[3,:]) #values تشبه عمل 
print(dataframe.iloc[::-1,::-1])
sns.swarmplot(data=data_to_plot, x='features', y='value',hue='target')

=== Diamond Price Predictor/Diamond_Price_Predictor_DT.ipynb
from sklearn.preprocessing import MinMaxScaler
print(df["carat"].value_counts())
print(df["cut"].value_counts())
print(df["color"].value_counts())
print(df["clarity"].value_counts())
print(df["price"].value_counts())
print(df["x"].value_counts())
print(df["y"].value_counts())
print(df["z"].value_counts())
print(df["depth"].value_counts())
print(df["table"].value_counts())
    sns.lineplot(data=df, x=col, y='price', ax=axs[i//3, i%3])
    sns.scatterplot(data=df, x=col, y='price', ax=axs[i//3, i%3])
#df1 = df2.fillna(df.columns.mean()) 
df1 = pd.DataFrame()
    print('Lower Bound for', column, ':', lower_bound) 
    print('Upper Bound for', column, ':', upper_bound) 
    print("-------------")
    df1[column] = d[(d[column] >= lower_bound) & (d[column] <= upper_bound)][column]
plt.bar(df1.columns, df1.isnull().sum())
imputer=SimpleImputer(strategy="median")
imputer.fit(df1)
df1=imputer.transform(df1)
print('missing value:',df1.isnull().any(axis=1).sum())
    plt.boxplot(df1[column], vert=False)
from sklearn.preprocessing import OneHotEncoder
columns_to_encode = ['cut', 'color', 'clarity']
onehot_encoder = OneHotEncoder(sparse_output=False)
df3=pd.merge(df2, df1, left_index=True, right_index=True)
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df3)
print(df4.describe())
X = df4.loc[:, df4.columns != 'price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)
    r2=metrics.r2_score(y_true, y_pred)
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
model = DecisionTreeRegressor(criterion='poisson')  
print("Train:")
print(regression_results(y_train,model.predict(X_train)))
print("Test:")
print(regression_results(y_test,model.predict(X_test)))
model = DecisionTreeRegressor(criterion='friedman_mse')  
print("Train:")
print(regression_results(y_train,model.predict(X_train)))
print("Test:")
print(regression_results(y_test,model.predict(X_test)))
model = DecisionTreeRegressor(criterion='squared_error')  
print("Train:")
print(regression_results(y_train,model.predict(X_train)))
print("Test:")
print(regression_results(y_test,model.predict(X_test)))
print("Train:")
print(regression_results(y_train,model2.predict(X_train)))
print("Test:")
print(regression_results(y_test,model2.predict(X_test)))

=== Diamond Price Predictor/Diamond_Price_Predictor_LR.ipynb
print(dataset["carat"].value_counts())
print(dataset["cut"].value_counts())
print(dataset["color"].value_counts())
print(dataset["clarity"].value_counts())
print(dataset["price"].value_counts())
print(dataset["x"].value_counts())
print(dataset["y"].value_counts())
print(dataset["z"].value_counts())
print(dataset["depth"].value_counts())
print(dataset["table"].value_counts())
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(dataset,test_size=0.2,
print(train_set)
X=dataset.drop(["depth", "price","Unnamed: 0"],axis=1)
y=dataset["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,
corr_one=pd.concat([X_train,y_train],axis=1).corr(numeric_only=True)
corr_two = dataset.corr(numeric_only=True)
df.sort_values(by=["corr_two"],ascending=False,inplace=True)
print(train_set_num)
print(train_set_obj)
#print()
#imputer=SimpleImputer(strategy="median")
print("train_set_num sum of null values: ",train_set_num.isnull().sum().sum())
print("test_set_num sum of null values: ",test_set_num.isnull().sum().sum())
print("ratio of outliers in train_set : ",(train_outliers==-1).sum()/len(train_outliers))
from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
print(train_set_obj.value_counts())
print("---------------------")
print(train_set_num.value_counts())
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
model = LinearRegression()
    r2=metrics.r2_score(y_true, y_pred)
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
r2:  0.918
r2:  0.9201
r2:  0.9023
r2:  0.9015
r2:  0.9023
r2:  0.9015
r2:  0.9187
r2:  0.9197
r2:  0.9187
r2:  0.9197

=== Exploratory-Data-Analysis-EDA/Matplotlib & Seaborn.ipynb
print()
print()
plt.scatter(x = df['enginesize'], y = df ['horsepower'], s=20, marker='x', alpha=0.5, c='m',label='A')  
df1 = pd.read_csv('AirPassengers.csv')
df1.head()
df1.info()
df1.isnull().sum()
df1.any()
df1.any(axis=0).sum()
df1['#Passengers'].value_counts()
df1['Month'].value_counts()
plt.plot(df1['Month'], df1['#Passengers'],  marker = '^', color = 'r', markersize = 5, label='A')
print(v_count,"\n","__________________________________")
print(type(fig)," __ ", type(ax),"\n__________________________________________")
print("__________________________________________")
bar2 = plt.bar(doornumber_carbode.columns,doornumber_carbode.loc['two'],alpha=0.5,width=0.4)
bar2 = plt.bar(np.arange(len(doornumber_carbode.columns)) + 0.2, doornumber_carbode.loc['two'],alpha=0.5,width=0.4)
print(bins)
sns.countplot(data = df, y = 'carbody', order =df['carbody'].value_counts().index)
sns.barplot(data = df, y = 'carbody', x ='enginesize', estimator = 'mean', errorbar = None)
sns.barplot(data = df, y = 'carbody', x ='enginesize', estimator = 'median', errorbar = 'sd', palette = 'muted')
bar_plot = sns.barplot(data = df, y = 'carbody', x ='enginesize', hue='doornumber', estimator = 'mean', errorbar = 'sd', palette = 'hls')
bar_plot = sns.catplot(data = df, kind = 'bar', y = 'carbody', x ='enginesize', col='doornumber', estimator = 'mean', errorbar = 'sd')
bar_plot.fig.suptitle('Avarage engine size by car body type split over number of doors', y=1.52)
sns.lineplot(data = amzn, x = 'Date', y = 'Open',color='red',marker='^')
sns.lineplot(data = amzn, x = 'Date', y = 'Open',color='red',marker='^', label = 'Open Price', alpha=0.5)
sns.lineplot(data = amzn, x = 'Date', y = 'Close',color='blue'i,marker='o', label = 'Close Price')
sns.boxplot(data = df, x = 'carbody', y = 'citympg', palette = 'hls')
sns.boxplot(data = df, x = 'carbody', y = 'price', hue = 'doornumber', palette = 'hls')

=== Handwritten Digit Recognition System/load_digits_PCA.ipynb
y = dataset.target
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.score(x_train, y_train)
model.score(x_test, y_test)
print(sum(pca.explained_variance_ratio_))
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.score(x_train, y_train)
model.score(x_test, y_test)
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.score(x_train, y_train)
model.score(x_test, y_test)
print(sum(pca.explained_variance_ratio_))

=== Handwritten Digit Recognition System/load_digits_TSNE.ipynb
y = digits.target
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
tsn = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42, verbose=1)
sns.scatterplot(x= x_tsne[:,0], y=x_tsne[:,1], hue=y, palette='tab20')
from sklearn.metrics import silhouette_score
silhouette_score(x_tsne, y)
sns.scatterplot(x= x_tsne2[:,0], y=x_tsne2[:,1], hue=y, palette='tab20')
silhouette_score(x_tsne2, y)

=== Healthcare Insurance Cost Predictor/Scikit Learning.ipynb
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.5f}")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.5f}")
sns.scatterplot(data = medical_df, x = 'age', y = 'charges', hue = 'smoker') 
sns.scatterplot(data = medical_df, x = 'bmi', y = 'charges', hue = 'smoker') 
sns.barplot(data = medical_df, x = 'children', y = 'charges') 
sns.violinplot(data = medical_df, x = 'children', y = 'charges')
sns.scatterplot(data = medical_df, x = 'age', y = 'charges', hue = 'smoker')
sns.scatterplot(data = no_smoker_df, x = 'age', y = 'charges', s = 15)
def rmse(target, prediction):
rmse(target, prediction)
rmse(target, prediction)
    loss = rmse(target, prediction)
    print("RMSE Loss: ", loss)
model = LinearRegression()
print("input.shape", input.shape)
print("target.shape", target.shape)
rmse (target, prediction)
print("w= ",model.coef_)
print("w= ",model.intercept_)
rmse(target, prediction)
print("w= ",model1.coef_)
print("w= ",model1.intercept_)
sns.scatterplot(data = smoker_df, x = 'age', y = 'charges', hue = 'smoker')
sns.scatterplot(data = smoker_df, x = 'age', y = 'charges', hue = 'smoker')
    loss = rmse(target, prediction)
    print("RMSE Loss: ", loss)
print("input.shape", input.shape)
print("target.shape", target.shape)
rmse (target, prediction)
print("w= ",model.coef_)
print("w= ",model.intercept_)
rmse(target, prediction)
print("w= ",model1.coef_)
print("w= ",model1.intercept_)
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
sns.stripplot(data = no_smoker_df, x = 'children', y = 'charges',  marker ='o')
sns.violinplot(data = no_smoker_df, x = 'children', y = 'charges')
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
sns.scatterplot(data = medical_df, x = 'age', y = 'charges', hue = 'smoker');
sns.barplot(data = medical_df, x = 'smoker', y = 'charges');
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
sns.barplot(data = medical_df, x = 'sex', y = 'charges', color = 'red');
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
sns.barplot(data = medical_df, x = 'region', y = 'charges');
enc = preprocessing.OneHotEncoder()
model = LinearRegression().fit(input, target)
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", model.coef_)
print("b=", model.intercept_)
print("w=", model.coef_)
print("b=", model.intercept_)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.mean_) #  حساب المتوسط لكل عامود
print(scaler.var_)  #  حساب التباين لكل عامود
loss = rmse(target, prediction)
print("Loss: ", loss)
print("w=", z.coef_)
print("b=", z.intercept_)
sns.scatterplot(data = raw_df, x = 'MinTemp', y = 'MaxTemp', hue ='RainToday', palette = 'hls', alpha = 0.5)
sns.scatterplot(data = raw_df, x = 'MinTemp', y = 'MaxTemp', hue ='RainTomorrow', palette = 'hls', alpha = 0.5)
sns.stripplot(data = raw_df, x = 'Humidity3pm', y = 'Temp3pm', hue ='RainToday', palette = 'hls', alpha = 0.5)
sns.stripplot(data = raw_df, x = 'Temp3pm', y = 'Humidity3pm', hue ='RainTomorrow', palette = 'hls')
from sklearn.model_selection import train_test_split
train_val_df, test_df = train_test_split(raw_df, test_size = 0.2, random_state = 42)
train_df, val_df = train_test_split(train_val_df, test_size = 0.25, random_state = 42)
print("train_df.shape :", train_df.shape)
print("val_df.shape :", val_df.shape)
print("test_df.shape :", test_df.shape)
print("train_df.shape :", train_df.shape)
print("val_df.shape :", val_df.shape)
print("test_df.shape :", test_df.shape)
imputer = SimpleImputer(strategy = 'mean')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print("Minimum")
print("Maximum")
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
?OneHotEncoder
print('train_input:', train_inputs.shape)
print('train_target:', train_target.shape)
print('val_input:', val_inputs.shape)
print('val_target:', val_target.shape)
print('test_input:', test_inputs.shape)
print('test_target:', test_target.shape)
print('train_input:', train_inputs.shape)
print('train_target:', train_target.shape)
print('val_input:', val_inputs.shape)
print('val_target:', val_target.shape)
print('test_input:', test_inputs.shape)
print('test_target:', test_target.shape)
model = LogisticRegression(solver = 'liblinear')
print(numeric_cols + categorical_cols)
print(model.coef_.shape)
print(model.intercept_)
sns.barplot(data = weight_df.sort_values('weight', ascending = False).head(10), x =  'feature', y = 'weight')
from sklearn.metrics import accuracy_score
accuracy_score(train_target, train_preds)
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
accuracy_score(test_target, random_guess(x_test))
accuracy_score(test_target, all_no(x_test))
accuracy_score(test_target, test_pred)
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore').fit(raw_df[categorical_cols])
from sklearn.metrics import accuracy_score, confusion_matrix
model = LogisticRegression(solver = 'liblinear')
accuracy_score(train_targets, train_preds)
# Helper function to predict compute accuracy & plot confustion matrix
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

=== Retail Sales & Customer Purchase Pattern Analysis/Online_Retail.ipynb
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler().fit_transform(numeric_df)
sns.scatterplot(x=components[:, 0], y=components[:, 1])
sns.barplot(x=non_null_counts.index, y=non_null_counts.values, palette="Set2")
print(ratios)
df = pd.get_dummies(df, columns=['Country'], drop_first=False)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = df  # ← تأكد أنك استعملت StandardScaler أو MinMaxScaler قبلها
print("عدد العينات في كل Cluster:")
print(df['Cluster'].value_counts().sort_index())
from sklearn.metrics import silhouette_score
X = df[numeric_cols]
score = silhouette_score(X, df['Cluster'])
print(f"Silhouette Score: {score:.4f}")

=== Retail Sales Forecasting System/XGBoost.ipynb
print(xgboost.__version__)
print(sys.executable)
print(sys.version)
ross_df = pd.read_csv('./rossmann-store-sales/train.csv', low_memory=False)
sns.scatterplot(data = temp_df, x = temp_df.Sales, y = temp_df.Customers, hue = pd.to_datetime(temp_df.Date).dt.year, alpha = 0.8)
sns.scatterplot(data = temp_df, x = temp_df.Sales, y = temp_df.Customers, hue = 'StoreType', alpha = 0.8)
sns.barplot(data = merged_df, x = 'DayOfWeek', y = 'Sales')
sns.barplot(data = merged_df, x = 'DayOfWeek', y = 'Sales', hue = 'Promo')
sns.barplot(data = merged_df, x = 'DayOfWeek', y = 'Sales', hue = 'SchoolHoliday')
sns.barplot(data = merged_df, x = 'Assortment', y = 'Sales')
sns.barplot(data = merged_df, x = pd.to_datetime(temp_df.Date).dt.year, y = 'Sales')
sns.barplot(data = merged_df, x = pd.to_datetime(temp_df.Date).dt.month, y = 'Sales', hue ='SchoolHoliday' )
sns.barplot(data = merged_df, x = pd.to_datetime(temp_df.Date).dt.month, y = 'Sales', hue = 'StateHoliday' )
sns.barplot(data = merged_df, x = pd.to_datetime(temp_df.Date).dt.day, y = 'Sales')
sns.scatterplot(data = merged_df.sample(40000), x = 'Store', y = 'Sales',  hue = pd.to_datetime(temp_df.Date).dt.year)
sns.catplot(kind='bar', data = merged_df, x=pd.to_datetime(merged_df.Date).dt.month, y='Sales', row=pd.to_datetime(merged_df.Date).dt.year , col='Promo', hue='Promo2')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(inputs[numeric_cols])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(inputs[categorical_cols])
X = inputs[numeric_cols + encoded_cols]
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)
print(preds)
print(targets)
print(preds2)
print(trees[0])
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    train_rmse = root_mean_squared_error(model.predict(X_train), train_targets)
    val_rmse = root_mean_squared_error(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse
    model, train_rmse, val_rmse = train_and_evaluate(X_train, 
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))
model = XGBRegressor(random_state=42, n_jobs=-1, max_depth=4, n_estimators=20)
scores = cross_val_score(model, X, targets, cv=kf, scoring='neg_mean_squared_error')
print("MSE for (KFold):", scores)
rmse_scores = np.sqrt(mse_scores)
print("RMSE for (KFold):", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
overall_rmse = root_mean_squared_error(targets, y_pred)
print("RMSE cross_val_predict:", overall_rmse)
model = XGBRegressor(random_state=42, n_jobs=-1, max_depth=4, n_estimators=100)
scores = cross_val_score(model, X, targets, cv=kf, scoring='neg_mean_squared_error')
print("MSE for (KFold):", scores)
rmse_scores = np.sqrt(mse_scores)
print("RMSE for (KFold):", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
overall_rmse = root_mean_squared_error(targets, y_pred)
print("RMSE cross_val_predict:", overall_rmse)
    train_rmses, val_rmses, models = [], [], []
        model, train_rmse, val_rmse = train_and_evaluate(X_train, train_targets, X_val, val_targets, **params)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    print('Train RMSE: {}, Validation RMSE: {}'.format(np.mean(train_rmses), np.mean(val_rmses)))
from sklearn.model_selection import train_test_split
X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size = 0.1)
    model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    train_rmse = root_mean_squared_error(model.predict(X_train), train_targets)
    val_rmse = root_mean_squared_error(model.predict(X_val), val_targets)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))
plt.plot(errors_df['n_estimators'], errors_df['train_rmse'])
plt.plot(errors_df['n_estimators'], errors_df['val_rmse'])
    model = XGBRegressor(max_depth = md, random_state = 42, n_jobs = -1)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
print("Best params:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)
    model, train_rmse, val_rmse = train_and_evaluate(X_train, 
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))
        'eval_metric': 'rmse',  # مقياس جذور متوسط الخطأ التربيعي
    model = XGBRegressor(**param, early_stopping_rounds=20)
study = optuna.create_study(direction='maximize')  # لأننا نُعيد سالب MSE
print("أفضل المعاملات:", study.best_params)
print("أفضل قيمة MSE:", -study.best_value)
model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=1000, 
model.score(X, targets)

=== Spotify Audio Feature-Based Recommendation System/spotify.ipynb
print(spotify_songs.columns,spotify_songs.dtypes)
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
sns.barplot(x='danceability',y='genre',data=spotify_songs)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
sns.scatterplot(x='liveness', y='danceability', data=spotify_songs, hue='label', palette='Set1', s=50) 
from sklearn.metrics import silhouette_score
silhouette_score(spotify_songs, y_pred)

=== Stroke Risk Prediction System/Stroke_Risk_Prediction.ipynb
print('Lower Bound :',lower_bound)
print('Upper Bound :',upper_bound)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df4 = pd.get_dummies(df4, columns=['smoking_status'],dtype=int)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df4)
print(df5['gender_enc'].corr(df5['stroke']))
X = df5.loc[:, df5.columns != 'stroke']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred), display_labels = [0, 1])
print(confusion_matrix(y_test, pred2))
print(classification_report(y_test, pred2))
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size = 0.2)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
X_train, X_test, y_train, y_test = train_test_split(X_OverSmote, Y_OverSmote, test_size = 0.2)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

=== Taxi Fare Prediction System/nyc-taxi-fare-prediction-filled.ipynb
from sklearn.model_selection import train_test_split
mean_model = MeanRegressor()
train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
train_rmse
val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
val_rmse
linreg_model = LinearRegression()
train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
train_rmse
val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
val_rmse
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    return train_rmse, val_rmse, train_preds, val_preds
    model = ModelClass(**params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
accuracy = (y_pred == y_test).mean()
print(f'دقة النموذج: {accuracy:.2f}')

=== Titanic Survival Predictive Analytics/Titanic_Survival_Predictive_Analytics.ipynb
print(df["Name"].value_counts())
print(df["Sex"].value_counts())
print(df["Ticket"].value_counts())
print(df["Cabin"].value_counts())
print(df["Embarked"].value_counts())
imputer=SimpleImputer(strategy="median")
imputer = SimpleImputer(strategy='most_frequent')
df1 = pd.DataFrame()
    print('Lower Bound for', column, ':', lower_bound) 
    print('Upper Bound for', column, ':', upper_bound) 
    print("-------------")
    df1[column] = d[(d[column] >= lower_bound) & (d[column] <= upper_bound)][column]
df1.isnull().sum()
imputer=SimpleImputer(strategy="median")
imputer.fit(df1)
df1=imputer.transform(df1)
df1.isnull().sum()
df1.describe()
df1.columns
from sklearn.preprocessing import OneHotEncoder
columns_to_encode = ['Sex', 'Ticket', 'Embarked']
onehot_encoder = OneHotEncoder(sparse_output=False)
df3=pd.merge(df2, df1, left_index=True, right_index=True)
print(df3["Survived"].value_counts())
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df3)
X = df4.loc[:, df4.columns != 'Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)
print("Minimum error:",min(error_rate),"at K =",error_rate.index(min(error_rate)))
from sklearn.metrics import accuracy_score 
accuracy3 = accuracy_score(y_test, pred3)
print('Accuracy = ', accuracy3)
print(confusion_matrix(y_test, pred3))
print(classification_report(y_test, pred3))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred3), display_labels = [0, 1])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_dt = accuracy_score(y_test, pred_dt)
print('Accuracy = ', accuracy_dt)
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred_dt), display_labels=[0, 1])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_svm = accuracy_score(y_test, pred_svm)
print('Accuracy = ', accuracy_svm)
print(confusion_matrix(y_test, pred_svm))
print(classification_report(y_test, pred_svm))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred_svm), display_labels=[0, 1])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_lr = accuracy_score(y_test, pred_lr)
print('Logistic Regression Accuracy = ', accuracy_lr)
print(confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_rf = accuracy_score(y_test, pred_rf)
print('Random Forest Accuracy = ', accuracy_rf)
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_ensemble = accuracy_score(y_test, pred_ensemble)
print('Ensemble Methods Accuracy = ', accuracy_ensemble)
print(confusion_matrix(y_test, pred_ensemble))
print(classification_report(y_test, pred_ensemble))

=== Unsupervised Learning and Recommendations/Unsupervised_Learning&Recommendations.ipynb
sns.scatterplot(data=iris_df, x='sepal_length', y='sepal_width', hue='species')
sns.scatterplot(data=iris_df, x='petal_length', y='petal_width', hue='species')
sns.scatterplot(data=iris_df, x='petal_length', y='sepal_length', hue='species')
sns.scatterplot(data=iris_df, x='sepal_width', y='petal_width', hue='species')
X = iris_df[numeric_cols]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model = KMeans(n_clusters=3, random_state=42)
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds);
centers_x, centers_y = model.cluster_centers_[:,0], model.cluster_centers_[:,2]
model = KMeans(n_clusters=3, random_state=42)
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=preds);
centers_x, centers_y = model.cluster_centers_[:,0], model.cluster_centers_[:,2]
    model = KMeans(n_clusters, random_state=42).fit(X)
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
def purity_score(y, y_pred):
purity_score(iris_df.species, preds)
nmi = normalized_mutual_info_score(iris_df.species, preds)
rand = adjusted_rand_score(iris_df.species, preds)
from sklearn.metrics import silhouette_score
silhouette_score(X, preds)
model = DBSCAN(eps=.4, min_samples=5)
from sklearn.metrics import silhouette_score
silhouette_score(X, model.labels_)
silhouette_score(X, labels)
sns.scatterplot(data=X, x='sepal_length', y='petal_length', hue=model.labels_);
sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df.species)
sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=model.labels_)
sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=iris_df['species'])

=== Weather Forecasting System using Machine Learning/Decision Tree.ipynb
sns.scatterplot(data = rain_df, x = 'MinTemp', y = 'MaxTemp', hue = 'RainToday')
sns.scatterplot(data = rain_df, x = 'Temp3pm', y = 'Temp9am', hue = 'RainToday')
imputer = SimpleImputer(strategy = 'mean').fit(rain_df[numeric_cols])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore').fit(train_inputs[categorical_cols])
model = DecisionTreeClassifier(random_state = 42)
from sklearn.metrics import accuracy_score, confusion_matrix
print(pd.Series(train_preds).value_counts())
print(pd.Series(train_targets).value_counts())
    accuracy = accuracy_score(targets, preds)
    print('Accuracy {:.2f}%'.format(accuracy * 100))
print(pd.Series(model.predict(x_val)).value_counts())
print(pd.Series(val_targets).value_counts())
print(text_tree[:5000])
sns.barplot(data = Features_Weigh.head(20), x = 'Importance', y = 'Features', palette = 'hls')
model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
model.score(x_train, train_targets)
model.score(x_val, val_targets)
    model = DecisionTreeClassifier(max_depth = md, random_state = 42)
    train_acc = 1 - model.score(x_train, train_targets)
    val_acc = 1 - model.score(x_val, val_targets)
model = DecisionTreeClassifier(max_depth = 7, random_state = 42)
model = DecisionTreeClassifier(max_leaf_nodes = 128, random_state = 42)
print(model_text)
model = DecisionTreeClassifier(max_leaf_nodes = 128, max_depth = 7, random_state = 42)
    model = DecisionTreeClassifier(max_leaf_nodes = md, random_state = 42)
    train_acc = 1 - model.score(x_train, train_targets)
    val_acc = 1 - model.score(x_val, val_targets)
model = DecisionTreeClassifier(class_weight = 'balanced', criterion = 'gini', max_depth = 7, random_state = 42)
print("أفضل المعاملات:", grid_search.best_params_)
print("أفضل دقة:", grid_search.best_score_)
print("أفضل المعاملات:", grid_search.best_params_)
print("أفضل دقة:", grid_search.best_score_)
    pruned_model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
# Find the model with the best accuracy on test data
best_accuracy = 0
best_pruned_model = None
    accuracy = pruned_model.score(x_test, test_targets)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_pruned_model = pruned_model
accuracy_after_pruning = best_pruned_model.score(x_test, test_targets)
print("Accuracy after pruning:", accuracy_after_pruning)
print(classification_report(test_targets, y_test_pred))

=== Weather Forecasting System using Machine Learning/Random Forest.ipynb
imputer = SimpleImputer(strategy = 'mean').fit(train_inputs[numeric_cols])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore').fit(train_inputs[categorical_cols])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model = RandomForestClassifier(n_jobs = -1, random_state = 42)
    accuracy = accuracy_score(targets, preds)
    print('Accuracy {:.2f}%'.format(accuracy * 100))
sns.barplot(data = importance_df.head(10), y = 'features', x = 'importance')
base_model = RandomForestClassifier(n_jobs = -1, random_state = 42).fit(x_train, train_targets)
base_train_acc = base_model.score(x_train, train_targets)
base_val_acc = base_model.score(x_val, val_targets)
model = RandomForestClassifier(n_jobs = -1, random_state = 42, n_estimators = 10).fit(x_train, train_targets)
model.score(x_train, train_targets),model.score(x_val, val_targets)
model = RandomForestClassifier(n_jobs = -1, random_state = 42, n_estimators = 500).fit(x_train, train_targets)
model.score(x_train, train_targets),model.score(x_val, val_targets)
    model = RandomForestClassifier(n_jobs = -1, random_state = 42, n_estimators = md)
    train_acc = 1 - model.score(x_train, train_targets)
    val_acc = 1 - model.score(x_val, val_targets)
    model = RandomForestClassifier(max_depth = md, random_state = 42, n_jobs = -1)
    train_acc = 1 - model.score(x_train, train_targets)
    val_acc = 1 - model.score(x_val, val_targets)
    model = RandomForestClassifier(**params, random_state = 42, n_jobs = -1).fit(x_train, train_targets)
    return model.score(x_train, train_targets),model.score(x_val, val_targets)
 model = RandomForestClassifier(random_state = 42,
model.score(x_train, train_targets),model.score(x_val, val_targets)
model.score(x_train, train_targets),model.score(x_val, val_targets)
model.score(x_test, test_targets)
sns.barplot(data = importance_df.head(10), y = 'features', x = 'importance')
print("best params: ", grid_search.best_params_)
print("best score: ", grid_search.best_score_)

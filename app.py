import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd 

df=pd.read_csv("data/behaviourData.csv")
df=df[['Prospect ID','Lead Number','Lead Origin','Lead Source',
        'Converted','TotalVisits','Total Time Spent on Website',
        'Page Views Per Visit','Specialization','Lead Quality','City']]

#Leads for a conversion
conv_list=list(df['Converted'])
tot_leads=len(set(df['Lead Number']))
tot_conversions=conv_list.count(1)
leadsForOneConversions=tot_conversions/tot_leads
#print(round(leadsForOneConversions,2))
#print("\n 40 leads out of 100 get converted")

#avg no of Page views it took for a customer to get converted
views=tot_converted=visits=tot_not_converted=0
for i in range(len(df)):
    if df.loc[i,'Converted']==1 and df.loc[i,'Page Views Per Visit']==df.loc[i,'Page Views Per Visit'] and df.loc[i,'TotalVisits']==df.loc[i,'TotalVisits']:
        views=views+df.loc[i,'Page Views Per Visit']*df.loc[i,'TotalVisits']
        visits+=df.loc[i,'TotalVisits']
        tot_converted+=1
    elif df.loc[i,'Converted']==0:
        tot_not_converted+=1

conOrNot=tot_converted+tot_not_converted
avgVisitsPerConversion=visits/tot_converted    
avgViewsPerConversion=views/tot_converted
#print("\n Total no of customers converted",tot_converted)
#print("\n Conversion Ratio: ", tot_converted/conOrNot)
#print("\n Avg no of views by a customer to get converted: ",round(avgViewsPerConversion,2))
#print("\n Avg no of visits by a customer to get converted: ",round(avgVisitsPerConversion,2))

#Average time on site by all customers
avg_time_on_site=sum(df['Total Time Spent on Website'])
avg_time_on_site/=len(df)
#print(round(avg_time_on_site,2))

#avg time spent by a customer on the website to get converted
tot_converted=timeSpent=0
for i in range(len(df)):
    if df.loc[i,'Converted']==1 and df.loc[i,'Total Time Spent on Website']==df.loc[i,'Total Time Spent on Website']:
        timeSpent+=df.loc[i,'Total Time Spent on Website']
        tot_converted+=1
avgTimeOnSiteConverted=timeSpent/tot_converted    
#print("\n Avg time spent on site by a customer to get converted: ",round(avgTimeOnSiteConverted,2),"seconds")
conversionRatio=tot_converted/conOrNot
#Repeat Visitor ratio
tot_returning=0
for i in range(len(df)):
    if df.loc[i,'TotalVisits']==df.loc[i,'TotalVisits'] and df.loc[i,'TotalVisits']>1:
        tot_returning+=1
repeatVisitorRatio=tot_returning/len(df)
#print("\n Repeat Visitor Ratio = ",round(repeatVisitorRatio,2))


#lead quality
lead_qual=list(df['Lead Quality'])
lead_quality=[]
leadss=[]
for x in lead_qual:
    if x==x:
        leadss.append(x.lower())
for x in leadss:
    if x!='might be' and x!='not sure':
        lead_quality.append(x)

no_qual=len(lead_quality)

#Specialization
spec=list(df['Specialization'])
spec_list=[]
leadss=[]
for x in spec:
    if x==x:
        leadss.append(x.lower())
for x in leadss:
    if x!='select':
        spec_list.append(x)

freq_spec={} 
for item in spec_list: 
    if (item in freq_spec): 
        freq_spec[item] += 1
    else: 
        freq_spec[item] = 1

#City
city=list(df['City'])
cities=[]
leadss=[]
for x in city:
    if x==x:
        leadss.append(x.lower())
for x in leadss:
    if x!='select':
        cities.append(x)
freq_city={} 
for item in cities: 
    if (item in freq_city): 
        freq_city[item] += 1
    else: 
        freq_city[item] = 1

# Classifying converting and non converting customers based on total visits, time on site and page views
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X=df[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']]
X.replace()
X=X.dropna()
x_data=X.iloc[:,:3]
y_data=X.iloc[:,3:4]
scores={}

x_train, x_test, y_train, y_test=train_test_split(x_data,y_data,test_size=0.2)
y_train=list(y_train['Converted'])

for k in range(1,25,2):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores[k]=metrics.f1_score(y_test,y_pred)
#print(scores)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_train_pred=knn.predict(x_train)
y_pred=knn.predict(x_test)

color_index={'1':'Converted','0':'Not converted'}
#print(freq_city)

fig1 = px.scatter(x_train, x="TotalVisits", y="Total Time Spent on Website", color=y_train_pred, color_discrete_map=color_index, title="True class")
fig2 = px.scatter(x_train, x="TotalVisits", y="Total Time Spent on Website", color=y_train, color_discrete_map=color_index, title="Predicted class" )

labelss=list(freq_city.keys())
num_values=list(freq_city.values())
fig3 = px.pie(values=num_values, names=labelss,color=labelss,title="Customer city") 

labelss=list(freq_spec.keys())
num_values=list(freq_spec.values())
fig4 = px.pie(values=num_values, names=labelss,color=labelss,title="Customer Specialization") 

app=dash.Dash(__name__)

colors = {
    'graphbg':'#000000',
    'background': '#000000',
    'text': '#fff8dc',
    'graphtext':'#99ccff'
}

bar_colors = ['lightslategray',] * 5

#Lead source
lead_source=list(df['Lead Source'])
leads=[]
leadss=[x for x in lead_source if x==x]
for lead in leadss:
    leads.append(lead.lower())

#Lead origin
lead_origins=list(df['Lead Origin'])
lead_origin=[]
leadss=[x for x in lead_origins if x==x]
for lead in leadss:
    lead_origin.append(lead.lower())

source_qual={}
for i in range(len(df)):
    if df.loc[i,'Lead Quality'] == df.loc[i,'Lead Quality'] and df.loc[i,'Lead Quality'] == 'Worst':
        item=df.loc[i,'Lead Source']
        if item in source_qual:
            source_qual[item]+=1
        else:
            source_qual[item]=1
#print(source_qual)

overall_qual={}
for i in range(len(df)):
    if df.loc[i,'Lead Quality'] == df.loc[i,'Lead Quality'] and df.loc[i,'Lead Quality']!='Might be' and df.loc[i,'Lead Quality']!='Not sure':
        item=df.loc[i,'Lead Source']
        if item in overall_qual:
            overall_qual[item]+=1
        else:
            overall_qual[item]=1
#print(overall_qual)

perc_poor_qual={}
for key in source_qual.keys():
    perc_poor_qual[key]=source_qual[key]/overall_qual[key]*100
#print(perc_poor_qual)


app.layout=html.Div(
    style={'backgroundColor': colors['background']},
    children=[
    dcc.Tabs([
    dcc.Tab(label='LEAD SOURCE',
    
    style={'backgroundColor': colors['background'],'color':'#99ccff'},
    children=[
        html.H1(children='WEB ANALYTICS REPORT ON ONLINE EDUCATION SITE',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.H2(children='TRAFFIC SEGMENTS',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children=[
        dcc.Graph(
            id="country_conversion",
            figure={
                'data':[
                    { 'x':leads, 'type':'histogram','name':'referrer website traffic','marker':{'color':['#ff9966',]*len(leads)}}
                ],
            'layout':{
                'title':'LEAD SOURCE',
                'xaxis':{
                    'title':'Source'
                },
                'yaxis':{
                     'title':'No of leads'
                },
                'tickangle':45,
                'paper_bgcolor':colors['graphbg'],
                'plot_bgcolor':colors['graphbg'],
                'font': {
                    'color': colors['graphtext']
                }
            }
            }
            )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        ),
    
        html.Div(children=[
        dcc.Graph(
            id="lead_origin",
            figure={
                'data':[
                    { 'x':lead_origin, 'type':'histogram','name':'lead origin traffic','marker':{'color':['#ff9966',]*len(lead_origin)}}
                ],
                'layout':{
                'title':'LEAD ORIGIN',
                'xaxis':{
                    'title':'Origin'
                },
                'yaxis':{
                     'title':'No of leads'
                },
                'tickangle':45,
                'paper_bgcolor':colors['graphbg'],
                'plot_bgcolor':colors['graphbg'],
                'font': {
                    'color': colors['graphtext']
                }
            }
            }
            )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )
    ]
    ),

    dcc.Tab(label='LEAD QUALITY',
    
    style={'backgroundColor': colors['background'],'color':'#99ccff'},
    children=[
        html.H1(children='WEB ANALYTICS REPORT ON ONLINE EDUCATION SITE',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.H2(children='QUALITY OF TRAFFIC',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children=[
        dcc.Graph(
            id="Lead qualtiy",
            figure={
                'data':[
                    { 'x':lead_quality, 'type':'histogram','name':'lead quality type','marker':{'color':['#ff9966',]*len(lead_qual)}}
                ],
            'layout':{
                'title':'LEAD QUALITY TYPE',
                'xaxis':{
                    'title':'Lead Quality'
                },
                'yaxis':{
                     'title':'No of leads'
                },
                'tickangle':45,
                'paper_bgcolor':colors['graphbg'],
                'plot_bgcolor':colors['graphbg'],
                'font': {
                    'color': colors['graphtext']
                }
            }
            }
            )
        ]
        )   
    ]
    ),

    dcc.Tab(label='CUSTOMER TYPE',
    
    style={'backgroundColor': colors['background'],'color':'#99ccff'},
    children=[
        html.H1(children='WEB ANALYTICS REPORT ON ONLINE EDUCATION SITE',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.H2(children='CUSTOMER SEGMENTS',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children=[
        dcc.Graph(
            id='customer_specialization',
            figure=fig4
        )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        ),
        html.Div(children=[
        dcc.Graph(
            id='City division',
            figure=fig3
        )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )
    ]
    ),

    dcc.Tab(label='CLASSIFICATION OF CUSTOMERS',
    
    style={'backgroundColor': colors['background'],'color':'#99ccff'},
    children=[
        html.H1(children='WEB ANALYTICS REPORT ON ONLINE EDUCATION SITE',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.H2(children='PREDICTING CUSTOMER CONVERSION',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.P(children='1-Converted',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.P(children='0-Not Converted',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children=[
        dcc.Graph(
            id='totVisitVsTOSime',
            figure=fig1
        )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        ),
        html.Div(children=[
        dcc.Graph(
            id='totVisitVsTOSimePred',
            figure=fig2
        )
        ],style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )
    
    ]
    ),

    dcc.Tab(label='METRICS TAB',
    
    style={'backgroundColor': colors['background'],'color':'#99ccff'},
    children=[
        html.H1(children='WEB ANALYTICS REPORT ON ONLINE EDUCATION SITE',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.H2(children='WEBSITE PERFORMANCE EVALUATION - METRICS & SUMMARY',    
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div(children=[
            html.P(children=["Conversion ratio = ",round(conversionRatio,2)]),
            html.P(children=["On average, 40 out of 100 leads get converted "]),
            html.P(children=["Average of Time spent on site by all customers = ",round(avg_time_on_site,2),"seconds"]),
            html.P(children=["Average of Time spent on site by customers who get converted = ",round(avgTimeOnSiteConverted,2),"seconds"]),
            html.P(children=["Avg no of views by a customer to get converted = ",round(avgViewsPerConversion,2)]),
            html.P(children=["Avg no of visits by a customer to get converted = ",round(avgVisitsPerConversion,2)]),
            html.P(children=["Repeat Visitor Ratio = ",round(repeatVisitorRatio,2)])
        ],style={'textAlign': 'center','color': colors['text'],'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        ),
        html.Div(children=[
            html.P(children=["Most traffic to the website is through Google and originate from landing page submission "]),
            html.P(children=["Lead quality is almost equally divided into high, medium, and poor."]),
            html.P(children=["Around 38 percent of traffic from Olark Chat is of poor quality"]),
            html.P(children=["Poor quality traffic has to be discarded"]),
            html.P(children=["Website has customers from various fields, the highest from finance management"]),
            html.P(children=["More than half of the website's customers reside in Mumbai"]),
            html.P(children=["Classifying customers based on time spent and visits gives an f1-score of 0.62."]),
            html.P(children=["So more information is necessary in order to provide proper classification of customers"])
        ],style={'textAlign': 'center','color': colors['text'],'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}
        )
    ]
    )    

    ]
    
    )]
)

if __name__=="__main__":
    app.run_server(debug=True)


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[3]:


NFA = pd.read_csv('NFA 2018.csv')
NFA.head(10)


#  We can look at the world that we live in and its resources in terms of supply and demand.
# * Demand side : Ecological Footprint (EF) - Measures the quantity of nature it takes to support people or an economy
# 
# * Supply side : Biocapacity (BC) - Measures the capacity of a given biologically productive area to generate an on-going supply of   renewable resources and to absorb its spillover wastes.
# 
# * Like in economics, when the demand (EF) exceeds the supply (BC) we are in a state of Ecological Deficit.
# When the supply (BC) exceeds the demand (EF) we are in a state of Ecological Reserve.
# 
# * Global Hectare (GHA) - A unit of land normalized by biological productivity across landtype. It starts with the total biological production and waste assimilation in the world, including crops, forests (both wood production and CO2 absorption), grazing and fishing. The total of these kinds of production, weighted by the richness of the land they use, is divided by the number of hectares used.
# 
# * GHA can be devided per person in its corresponding geographical area (city, country, continent, world). In our data it appears as BC/EF per capita.
# 
# EF and BC are calculated for:
# 
# - Crop land - GHA of crop land available or demanded.
# - Grazing land - GHA of grazing land (used for meat, dairy, leather, etc.) Includes global hectares used for grazing, but not crop land used to produce feed for animals.
# - Forest land - GHA of forest land available (for sequestration and timber, pulp, or timber products) or demanded (for timber, pulp, or timber products).
# - Fishing grounds - GHA of marine and inland fishing grounds (used for fish & fish products).
# - Built up land - GHA of built-up land (land cover of human infrastructure).
# - Carbon - GHA of average forest required to sequester carbon emissions (for EF only. For BC it's calculated within the forest land section).

# In[4]:


# dataset informations:
NFA.info()


# In[5]:


# Columns names :
NFA.columns


# In[6]:


# the Percapita GDP (2010 USD) Columnn has space and this may cause issues later so we will rename it :
NFA.rename(columns={'Percapita GDP (2010 USD)': 'Percapita_GDP_2010_USD'}, inplace=True)
NFA.columns


# In[7]:


# Renaming ISO alpha-3 code to ISO_alpha-3_code
NFA.rename(columns={'ISO alpha-3 code': 'ISO_alpha-3_code'}, inplace=True)
NFA.columns


# In[8]:


NFA.isnull().sum()


# In[8]:


# range of years in dataset : (1961, 2014)
years_max = NFA['year'].max()
years_min = NFA['year'].min()
years_min,years_max


# In[9]:


# Let's see the types of records we have for each country :
NFA['record'].unique()


# * For each country and year included in the data, we can find mesearments of EF and BC in GHA and in per-capita units (4 out of 10 types of "record"), across all types of land/resource mentioned in the previous section and their sum total.
# * We note that EF represented in 4 categories.
# * The EF Consumption accounts for area required by consumption, while the EF of Production accounts only for the area required for production in the country only. They are related by the following equation:
# EF Consumption = EF Production + EF Imports - EF Exports
# *EF Production, EF Imports and EF Exports also appear in both GHA and in per-capita units (remaining 6 out of 10)

# In[10]:


# Let's count NaN records for each type of record.
def num_nan(df):
    return df.shape[0]-df.count()

land_types=['crop_land', 'grazing_land', 'forest_land', 'fishing_ground','built_up_land', 'carbon', 'total']

(NFA[['record']+land_types]
    .groupby('record')
    .agg(num_nan))


#  One thing we notice that "World" is listed as a country, so for global calculations we can use a dataframe that includes only records for "World"

# In[11]:


world = NFA.loc[NFA['country']=='World']
world.head()


# In[12]:


# For regional calculations we can use :
regions = NFA.loc[NFA['country']!='World']
regions.head()


# In[13]:


# 10 first Countries with high Biocapacity Per Capita during 1961-2014 :
countries_bc = (pd.pivot_table(regions,values = 'total',index=['country'],columns=['record'],aggfunc='sum')      ['BiocapPerCap'].reset_index().set_index('country'))
top10_countries_bc = countries_bc.sort_values(by='BiocapPerCap',ascending=False).head(10)
top10_countries_bc.plot(kind='bar',figsize=(12,5),color = 'steelblue',rot=70)
plt.title('10 first Countries with high Biocapacity Per Capita during 1961-2014')
plt.xlabel('Countries')
plt.ylabel('Biocapacity Per capita')


# In[14]:


# 10 Countries with high Ecological Footprint Consumption  Per capita during 1961-2014 :
countries_ef = (pd.pivot_table(regions,values = 'total',index=['country'],columns=['record'],aggfunc='sum')      ['EFConsPerCap'].reset_index().set_index('country'))
top10_countries_ef = countries_ef.sort_values(by='EFConsPerCap',ascending=False).head(10)
top10_countries_ef.plot(kind='bar',figsize=(12,5))
plt.title('10 Countries with high Ecological Footprint Consumption  Per capita during 1961-2014')
plt.xlabel('Ecological Footprint Consumption Per capita')
plt.ylabel('Countries')


# In[15]:


world_fp_bc = pd.pivot_table(world,values = 'total',index=['year'],columns=['record'],aggfunc='sum')[['BiocapPerCap','EFConsPerCap']]

world_fp_bc.plot(kind='line', figsize=(13,8))

plt.annotate('Ecological Footprint Exceeds Biocapacity',
             xy=(1970, 2.75),  
             xytext=(1970, 3), 
             xycoords='data', 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

plt.title('World Footprint & Biocapacity Trends')
plt.ylabel('Ecological Footprint & Biocapacity Per Capita')
plt.show()

# The plot shows that starting in 1970, human demand on nature was greater than the ability of the environment
# to sustainable provide for it.


# In[16]:


#plt.figure(figsize=(10,5))
NFA.groupby('UN_region')['country'].nunique().plot.pie()


# - We want to calculate the BC and EF for each region.
# - For this we will use 2 pivot tables to sum the BC and EF total GHA and population for each region and year.

# In[17]:


pt1 = (pd.pivot_table(regions,values = 'total',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')      [['BiocapTotGHA','EFConsTotGHA']].reset_index().set_index('UN_region'))
pt1


# In[18]:


pt2 = pd.pivot_table(regions,values = 'population',index=['UN_region', 'year'],columns=['record'],aggfunc='sum')[['BiocapTotGHA']].rename(index=str, columns={'BiocapTotGHA': 'population'}).reset_index().drop(['year'],axis=1)        .set_index('UN_region')
pt2


# - Then we concatenate the 2 tables and create 2 new columns, calculating BC and EF per capita of the region

# In[21]:


# joining two tables and creating two columns "BiocapPerCap_region" & "EFConsPerCap_region"
result_pt = pd.concat([pt1, pt2], axis=1)
result_pt['BiocapPerCap_region']=result_pt['BiocapTotGHA']/result_pt['population']
result_pt['EFConsPerCap_region']=result_pt['EFConsTotGHA']/result_pt['population']
result_pt2 = (result_pt[['year','BiocapPerCap_region','EFConsPerCap_region']]
                .reset_index()
                .set_index(['UN_region','year'])
            )
result_pt2


# In[23]:


N = 3
fig = plt.figure(figsize=(15, 15))
for i,k in enumerate(regions.UN_region.unique()):   
    ax_num = fig.add_subplot(N, N, i+1)
    ax_num.set_title (k)
    ax_num.set_ylim ((0,20))
    result_pt2.loc[k].plot(ax=ax_num)
    
    
fig.tight_layout()
plt.show()

# "BiocapPerCap_region" & "EFConsPerCap_region" for the 6 regions 


# - We will look at the EF GHA, by land types, to try to determine which segments affects the general EF calculations.
# - We will use an area plot for that.

# In[24]:


EF_GHA_by_land= (world[world['record']=='EFConsTotGHA'][['year']+land_types[:-1]].set_index(['year']))

EF_GHA_by_land.plot.area(figsize=(10, 10))
plt.title('EF Comsumption by land types  Trends')
plt.ylabel('EF Comsumption')
plt.show()

#There's an increase in all land types along the years, but we can see that carbon emmisions has the mose significant growth


# In[25]:


world_bc_gha = world[world['record']=='BiocapTotGHA']
world_ef_gha = world[world['record']=='EFConsTotGHA']
world_ef_bc_gha = world_bc_gha.append(world_ef_gha)
world_ef_bc_gha.head(2)


# In[26]:


from plotnine import ggplot, aes, ylab,geom_line,ggtitle
ggplot(world_ef_bc_gha)+aes(x="year",y="total",color="record")+geom_line()+ylab('Ecological Footprint Per GHA ')+ ggtitle('World Footprint & Biocapacity Trends Per GHA')


# In[27]:


import warnings
warnings.filterwarnings('ignore')


# In[28]:


region_bc_percap = regions[regions['record']=='BiocapPerCap']
region_ef_percap = regions[regions['record']=='EFConsPerCap']
region_ef_consum = regions[regions['record']=='EFConsTotGHA']


# In[29]:


from plotnine import ggplot, aes, ylab,geom_boxplot

ggplot(data=region_ef_percap)+geom_boxplot(aes(x='UN_region',y='total',fill='UN_region'))+ylab('Ecological Footprint Per capita')


# In[30]:


region_ef_percap_2014 = region_ef_percap[region_ef_percap['year']==2014]
region_bc_percap_2014 = region_bc_percap[region_bc_percap['year']==2014]
region_ef_consu_2014 = region_ef_consum[region_ef_consum['year']==2014]


# In[31]:


# Ecological Footprint total Consumption for countries during the year 2014 :
px.choropleth(region_ef_consu_2014,locations="ISO_alpha-3_code",hover_name="country",color="total",height=500,             title="Ecological Footprint total Consumption for countries during the year 2014 (GHA)")


# In[32]:


# Biocapacity per Capita for countries during the year 2014 :
px.choropleth(region_bc_percap_2014,locations="ISO_alpha-3_code",hover_name="country",color="total",height=500,             title="Biocapacity per Capita for countries during the year 2014")


# In[33]:


# Ecological Footprint per Capita for countries during the year 2014 :
px.choropleth(region_ef_percap_2014,locations="ISO_alpha-3_code",hover_name="country",color="total",height=500)


# In[34]:


px.scatter(region_ef_percap_2014,y="total",x="Percapita_GDP_2010_USD",hover_name="country",height=500,color="UN_region",           size="population",size_max=60,log_x=True,title="EF Consumption Per Capita VS GDP Per Capita for 2014",          labels={"total":"EF Consumption Per Capita","Percapita_GDP_2010_USD":"GDP Per Capita (2010 USD)"})


# In[35]:


px.scatter(region_ef_percap_2014,y="total",x="Percapita_GDP_2010_USD",hover_name="country",height=500,color="country",           size="population",size_max=60,log_x=True,title="EF Consumption Per Capita VS GDP Per Capita for 2014",          labels={"total":"EF Consumption Per Capita","Percapita_GDP_2010_USD":"GDP Per Capita (2010 USD)"})


# In[36]:


countries_ef_pop = (pd.pivot_table(regions,values = 'total',index=['country','year','population','Percapita_GDP_2010_USD'],columns=['record'],aggfunc='sum')      ['EFConsPerCap'].reset_index().set_index('country'))
countries_ef_pop[countries_ef_pop['year']==2014].sort_values(by='EFConsPerCap',ascending=False)

top15_countries_ef_pop = countries_ef_pop[countries_ef_pop['year']==2014].sort_values(by='EFConsPerCap',ascending=False).head(15)

top15_countries_ef_pop = top15_countries_ef_pop.reset_index()
top15_countries_ef_pop.head(2)


# In[37]:


px.scatter(top15_countries_ef_pop,y="EFConsPerCap",x="Percapita_GDP_2010_USD",hover_name="country",height=500,color="country",size="EFConsPerCap",size_max=60,log_x=True,title="EF Consumption Per Capita VS GDP Per Capita for top 15 countries year 2014 ",labels={"total":"EF Consumption Per Capita","Percapita_GDP_2010_USD":"GDP Per Capita (2010 USD)"})


# In[38]:


world_cons = world[world['record']=='EFConsTotGHA']
world_carbon_percap = world[world['record']=='EFConsPerCap']


# In[39]:


px.line(world_cons, x="year", y=["carbon","crop_land","grazing_land","fishing_ground","forest_land","built_up_land"],        title='Evolution of the total Consumption per land type as well as Carbon Emission for the world',
       labels={"value":"Total EF Consumption (GHA)"})


# In[40]:


px.line(world_carbon_percap, x="year", y=["carbon"],        title='Evolution of the carbon emission per capita for the world',
       labels={"value":"Carbon emission per capita (GHA)"})


# In[41]:


regional = result_pt.reset_index()
reserve_deficit = regional.groupby('UN_region')[['BiocapPerCap_region','EFConsPerCap_region']].sum()
reserve_deficit['reserve_or_deficit'] = reserve_deficit['BiocapPerCap_region'] - reserve_deficit['EFConsPerCap_region']
final = reserve_deficit.reset_index()
final


# In[42]:


px.bar(final, x="reserve_or_deficit", y="UN_region", orientation='h',color='reserve_or_deficit'
       ,title='Regional Biocapacity Reserve(+) or Deficit(-)'
       ,labels={"UN_region":"World's Regions","reserve_or_deficit":"Biocapacity Reserve or Deficit Per Capita"})


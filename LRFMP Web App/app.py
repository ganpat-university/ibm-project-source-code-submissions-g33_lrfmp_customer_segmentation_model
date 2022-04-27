# Import all the Important Modules
import streamlit as st 
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Favicon and Title 
st.set_page_config(
    page_title="LRFMP Analysis",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Upload File
uploaded_file = st.file_uploader("Choose a file")

# Read File as CSV Using Pandas Modules
if uploaded_file is not None:
    data_df = pd.read_csv(uploaded_file,  encoding= 'unicode_escape')

    # Show Data
    if st.checkbox("Show Sample Data"):
        st.write(data_df.head(40))      # Display First 10 Rows
    if st.checkbox('Show Full Data'):
        data_df                         # Display Full Data

    # review the value counts
    freq_items = data_df['itemDescription'].value_counts().head(15)
    chart_data = pd.DataFrame(freq_items)
    st.bar_chart(chart_data)

    # list items by member IDs
    user_id = data_df['Member_number'].unique()
    items = [list(data_df.loc[data_df['Member_number'] == id, 'itemDescription']) for id in user_id]

    # create a item matrix
    TE = TransactionEncoder()
    TE.fit(items)
    item_transformed = TE.transform(items)
    item_matrix = pd.DataFrame(item_transformed, columns = TE.columns_)
    st.write(item_matrix.head(10))

    # get the support value by Apriori algorithm
    freq_items = apriori(item_matrix, min_support=0.01, use_colnames=True, max_len=2)
    freq_items.sort_values(by = "support", ascending = False)

    # create a datafram with product support, confidence , and lift values
    rules = association_rules(freq_items, metric = "confidence", min_threshold = 0)

    # add a column for a Zhang's core
    def zhangs_rule(rules):
        rule_support = rules['support'].copy()
        rule_ante = rules['antecedent support'].copy()
        rule_conseq = rules['consequent support'].copy()
        num = rule_support - (rule_ante * rule_conseq)
        denom = np.max((rule_support * (1 - rule_ante).values, 
                            rule_ante * (rule_conseq - rule_support).values), axis = 0)
        return num / denom

    rules_zhangs_list = zhangs_rule(rules)
    rules = rules.assign(zhang = rules_zhangs_list)

    # regarding the whole mike has the highest support, choose it as the item for the basket analysis
    rules_sel = rules[rules["antecedents"].apply(lambda x: "whole milk" in x)]
    rules_sel.sort_values('confidence', ascending=False)

    # pd.read_csv()
    if st.checkbox("Show Combo Data"):
        combo_df = pd.read_csv('/Users/jaypatel/Documents/University Material/Semester 8/IBM Project Code Files/Market Basket Analysis (MBA)/data/basket.csv')
        st.write(combo_df)
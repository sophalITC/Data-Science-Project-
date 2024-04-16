import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide")

# Function to load data
def view_all_data():
    data = pd.read_csv("motorbile_good.csv")
    return data

# Fetch data
df = view_all_data()

# Sidebar
st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7tH92kh2edh33_dAzRY-NPMUxv4QK3Nk4Iw&usqp=CAU',
                 caption='Project Analytics')

# Filter options
region = st.sidebar.multiselect("Select Location", options=df["locations "].unique(), default=df["locations "].unique())
location = st.sidebar.multiselect("Select Model", options=df["model"].unique(), default=df["model"].unique())
condition = st.sidebar.multiselect("Select Condition", options=df["condition"].unique(), default=df["condition"].unique())

# Filter the dataframe
df_selection = df[(df["locations "].isin(region)) & (df["model"].isin(location)) & (df["condition"].isin(condition))]

# Function to display home section
def Home():
    # Display data based on filter selection
    with st.expander("⏰ My Excel WorkBook"):
        showData = st.multiselect('Filter', df_selection.columns, default=[])
        st.write(df_selection[showData])

    # Compute top analytics
    total_investment = df_selection['price'].sum()
    investment_mode = df_selection['price'].mode().iloc[0]
    investment_mean = df_selection['price'].mean()
    investment_median = df_selection['price'].median()
    rating = df_selection['price'].sum()

    total1, total2, total3, total4, total5 = st.columns(5, gap='large')

    with total1:
        st.info('Total Price', icon="📌")
        st.metric(label="sum TZS", value=f"{total_investment:,.0f}")

    with total2:
        st.info('Most frequent', icon="📌")
        st.metric(label="mode TZS", value=f"{investment_mode:,.0f}")

    with total3:
        st.info('Average', icon="📌")
        st.metric(label="average TZS", value=f"{investment_mean:,.0f}")

    with total4:
        st.info('Central Earnings', icon="📌")
        st.metric(label="median TZS", value=f"{investment_median:,.0f}")

    with total5:
        st.info('Ratings', icon="📌")
        st.metric(label="Rating", value=f"{rating:,.0f}", help=f"Total Rating: {rating}")

    st.markdown("---")

# Function to display graphs
def graphs():
    # Simple bar graph
    investment_by_model = df_selection.groupby("model").count()["price"].sort_values()
    fig_investment = px.bar(
        investment_by_model,
        x=investment_by_model.index,
        y="price",
        orientation="h",
        title="<b> Investment by Model </b>",
        color_discrete_sequence=["#0083B8"] * len(investment_by_model),
        template="plotly_white"
    )

    fig_investment.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False)
    )

    # Simple line graph
    investment_by_location = df_selection.groupby("locations ").count()["price"]
    fig_location = px.line(
        investment_by_location,
        x=investment_by_location.index,
        y="price",
        orientation="v",
        title="<b> Investment by Location </b>",
        color_discrete_sequence=["#0083b8"] * len(investment_by_location),
        template="plotly_white"
    )

    fig_location.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False)
    )

    left, right = st.columns(2)
    left.plotly_chart(fig_location, use_container_width=True)
    right.plotly_chart(fig_investment, use_container_width=True)


# Function to display categorical visualization
def categorical_visualization(df, cols):
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    sns.countplot(x=cols, data=df, palette="Set2", order=df[cols].value_counts().index)
    plt.title(f"{cols} Distribution", pad=10, fontweight="black", fontsize=18)
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 2)
    sns.boxplot(x=cols, y="price", data=df, palette="Set2")
    plt.title(f"{cols} vs price", pad=20, fontweight="black", fontsize=18)
    plt.xticks(rotation=90)
    
    plt.subplot(1, 3, 3)
    x = pd.DataFrame(df.groupby(cols)["price"].mean().sort_values(ascending=False))
    sns.barplot(x=x.index, y="price", data=x, palette="Set2")
    plt.title(f"{cols} vs Average Price", pad=20, fontweight="black", fontsize=18)
    plt.xticks(rotation=90)
    plt.tight_layout()

    st.pyplot(plt.gcf())


# Function to read data
def read_data():
    

    st.write('\n')
    st.markdown("##")

    st.markdown("""---""")
    st.markdown("<div style='padding:20px; color:white; margin:10; font-size:170%; text-align:left; display:fill; border-radius:5px; background-color:#222222; overflow:hidden; font-weight:700'>1. <span style='color:#CDA63A'>|</span> Exploring The Dataset (EDA)</div>", unsafe_allow_html=True)

    st.write(
        """
        - ✔️ Load The Dataset
        - ✔️ Data Description
        - ✔️ Good understanding of statistical principles but struggles to apply them
        - ✔️ Data Cleaning
           - ✔️ Missing Value
           - ✔️ Duplicated
        """
    )
    df = pd.read_csv("motorbile_good.csv")
    a = df.describe(include=['O'])
    st.dataframe(a)

    ohio_panel = st.container()

    with ohio_panel:
        columns = st.columns([3, 3])

        with columns[0]:
            d = ((df.isnull().sum() / df.shape[0])).sort_values(ascending=False)
            d.plot(kind='bar',
                    color=sns.cubehelix_palette(start=2, rot=0.3, dark=0.15, light=0.9, reverse=True, n_colors=24),
                    figsize=(4, 2))
            plt.title("\nProportions of Missing Values:\n", fontsize=40)
            plt.tight_layout()
            fig = plt.gcf()
            st.pyplot(fig)

        with columns[1]:
            st.write(
                """
                - ✔️ The missing values are in the "model" column (0.0024% missing values) and the "color" column (0.0099% missing values).
                - ✔️ The missing data seems to be Missing Completely At Random (MCAR), which means that there is no relationship between the missing values and other observed or unobserved variables in the dataset.
                """
            )

    st.markdown("""---""")
    ohio_pane2 = st.container()

    with ohio_pane2:
        columns = st.columns([3, 3])

        with columns[0]:
            fig, ax = plt.subplots(figsize=(4, 4))
            color_palette = ['lightblue']
            sns.set_style("whitegrid")
            sns.boxplot(df['price'], color=color_palette[0])
            ax.set_xlabel('Price')
            ax.set_title('Distribution of Price')

            st.pyplot(fig)

        with columns[1]:
            st.write(
                """
                - ✔️ The data contains outliers due to higher prices, which can have an impact on the predictions.
                - ✔️ The labels for prediction do not follow a perfect normal distribution.
                """
            )

    st.write(
        """
        - ✔️ The dataset contains some duplicated rows (3 rows).
        - ✔️ The data has errors in the dataset wrapping.
        - ✔️ The dataset contains errors in the data, and we are trying to predict prices for motorbikes from 2012 to 2022.
        - ✔️ Based on observations, the dataset is not perfect for prediction, and we cannot use clustering due to the low accuracy of dummy index and negative silhouette score.
        """
    )
    st.markdown("""---""")

    st.markdown("<div style='padding:20px; color:white; margin:10; font-size:170%; text-align:left; display:fill; border-radius:5px; background-color:#222222; overflow:hidden; font-weight:700'>2. <span style='color:#CDA63A'>|</span> Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)

    st.write(
        """
        - ✔️ Target variable analysis (price)
        """
    )

    ohio_panel3 = st.container()
    
    with ohio_panel3:
        columns = st.columns([3, 3])

        with columns[0]:
            plt.figure(figsize=(15, 7))
            sns.histplot(df.price, bins=40, color='steelblue')
            plt.ylabel('Frequency')
            plt.xlabel('Price')
            plt.title('Distribution of Prices')
            plt.tight_layout()
            st.pyplot(plt)
            
        with columns[1]:
            st.markdown("""
            
                - ✔️ The price distribution is a long-tail distribution, which is typical for many items with low prices and very few expensive ones.
                - ✔️ We can have a closer look by zooming in and focusing on values below 1000.
           
            """)
            
    ohio_panel4 = st.container()
    
    with ohio_panel4:
        columns = st.columns([3, 3])

        with columns[0]:
            df['log_price'] = np.log1p(df.price)
            Log_Ave = df["price"].map(lambda i: np.log(i) if i > 0 else 0)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.distplot(Log_Ave, label="Skewness: %.2f" % (Log_Ave.skew()))
            ax.set_xlabel('Log Price')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Log-Transformed Prices')
            ax.legend()

            st.pyplot(fig)
            
        with columns[1]:
            st.markdown(
                """
                - ✔️ The log-transformed prices have reduced skewness compared to the original prices.
                - ✔️ Analyzing the transformed prices can provide insights into the relationship between features and price.
                - ✔️ We can use log-transformed prices for better modeling results.
                """
            )
            
    st.markdown("""---""")
    st.write(
        """
        <h3> Data Visualization </h1>
           - ✔️ Numerical variables
           - ✔️ Discrete variables
           - ✔️ Categorical variables
           - ✔️ Correlation
        """
        , unsafe_allow_html=True)
    
    numerical_features = [feature for feature in df.columns if df[feature].dtype != "O"]
    discrete_features = [feature for feature in numerical_features if len(df[feature].unique()) <= 6]
    continuous_features = [feature for feature in numerical_features if feature not in discrete_features]

    st.write(len(continuous_features))
    st.write(df[continuous_features].head())
    data = df.copy()
    colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'yellow']

    columns = st.columns(2)
    for i, feature in enumerate(continuous_features):
        with columns[i % 2]:
            fig, ax = plt.subplots()
            data.groupby(feature)['price'].median().plot.hist(color=colors[i % len(colors)], ax=ax)
            plt.xlabel(feature)
            plt.ylabel('Price')
            plt.title(f'Distribution of Price for {feature}')
            st.pyplot(fig)
            plt.close(fig)
    st.markdown("""---""")
    ohio_panel2 = st.container()
    
    with ohio_panel2:
        columns = st.columns([3, 3])

        with columns[0]:
           categorical_visualization(df, 'model')
        
        with columns[1]:
            st.markdown(
                """
                - ✔️ To find the best correlation, we need to remove outliers that can weaken the correlation and remove error data.
                - ✔️ Category variables may not have a significant effect on the prediction because using dummy or target coding can cause the effect feature to be dropped.
                - ✔️ The data has a low correlation with the labels.
                """
            )

    st.markdown("<div style='padding:20px; color:white; margin:10; font-size:170%; text-align:left; display:fill; border-radius:5px; background-color:#222222; overflow:hidden; font-weight:700'>3 <span style='color:#CDA63A'>|</span> Preprocessing</div>", unsafe_allow_html=True)

    st.markdown(
        """
        - ✔️ We can use VIF to identify features that are not significant.
        - ✔️ We need to handle multicollinearity.
        - ✔️ We want to find the best model with a high R-square and adjusted R-square.
        """
    )

    st.markdown("<div style='padding:20px; color:white; margin:10; font-size:170%; text-align:left; display:fill; border-radius:5px; background-color:#222222; overflow:hidden; font-weight:700'>4 <span style='color:#CDA63A'>|</span> Evaluate</div>", unsafe_allow_html=True)
    ohio_panel2 = st.container()
    
    with ohio_panel2:
        columns = st.columns([3, 3])

        with columns[0]:
            data = {
                "Algorithms": ["LinearRegression 1", "Ridge Regression Model 2", "Lasso Regresion Model 3", "Ridge Poly Regresion Model 4", "Lasso Poly Regresion Model 5"],
                "Training Score": [0.73, 0.75, 0.62, 0.82, 0.62],
                "Testing Score": [0.70, 0.69, 0.57, 0.73, 0.57]
            }
            df_model = pd.DataFrame(data)

            # Plot performance visualization
            plt.figure(figsize=(16, 6))
            df_model.plot(x="Algorithms", y=["Training Score", "Testing Score"], kind="bar",
                          title="Performance Visualization of Different Models", colormap="Set1")
            plt.xticks(rotation=0)
            plt.xlabel("Algorithms")
            plt.ylabel("Score")
            plt.tight_layout()

             # Display the plot
            st.pyplot(plt)
        
        with columns[1]:
            st.markdown(
            """
                - ✔️ The models achieved an accuracy of around 75%.
                - ✔️ The Ridge Poly Regression Model 4 performed the best with the highest training and testing scores.
                - ✔️ Some models may show signs of overfitting.
            """)
    st.markdown("""
    <div class="alert alert-block" style="background-color: aqua; font-size:14px; font-family:verdana; line-height: 1.7em; color:black">
        <center>Thank you for reading🙂</center>
        <center>If you have any feedback or find anything wrong, please let me know!</center>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    Home()
    graphs()
    read_data()

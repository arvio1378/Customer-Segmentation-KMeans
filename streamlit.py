import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🧑‍🤝‍🧑",
    layout="wide"
)

# Load dataset
@st.cache_data # fungsi agar tidak selalu meload dataset dari awal
def load_data():
    df = pd.read_csv("./data/customer_segmentation.csv")
    # Preprocessing Data
    # Hapus kolom yang tidak perlu
    df = df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"], axis=1)

    # Hapus data yang duplikat
    df = df.drop_duplicates()

    # Ubah tipe kolom "Dt_Customer" dari object ke datetime
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
    # Ubah ke numerik dan buat fitur baru
    df["DAYS_Dt_Customer"] = (datetime.today() - df["Dt_Customer"]).dt.days
    # Hapus kolom datetime
    df = df.drop(columns=["Dt_Customer"])
    # ganti nilai yang hilang dengan mean
    mean_income = df["Income"].mean()
    df["Income"] = df["Income"].fillna(mean_income)

    # numeric data
    df_num = ["Year_Birth", "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "DAYS_Dt_Customer"]

    # categorical data
    df_cat = ["Education", "Marital_Status", "Kidhome", "Teenhome", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2", "Complain", "Response"]

    # Outlier
    for col in df_num:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df

# Navigasi sidebar
st.sidebar.title("Navigation")
# Pilihan Navigasi
pages = st.sidebar.selectbox("Menu : ", ["Overview", "EDA", "Clustering", "Profile"])

# Page 1 : Overview
if pages == "Overview":
    st.title("🧑‍🤝‍🧑 Customer Segmentation Clustering")

    # Deskripsi aplikasi
    st.write("### 📋 Description")
    st.write("This project is my practice project to be able to group customers into several sections based on customer patterns using K-Means Clustering so that it can be useful for implementing strategies for business and marketing that are precisely targeted to customers.")

    # Tujuan aplikasi
    st.write("### 🎯 Objective")
    st.write("""
    - Grouping customers into groups
    - Providing conclusions from several customer groups
    - Providing strategies in terms of good business & marketing
    """)

    # Membuat tab
    tab1, tab2, tab3 = st.tabs(["Application Features", "Tools & Library", "Folder Structure"])
    with tab1:
        # Fitur aplikasi
        st.write("### 🚀 Application Features")
        st.write("""
        - The dataset is saved in CSV format
        - Decision Tree Model with scikit-learn
        """)
    with tab2:
        # Tools & Library
        st.write("### 🛠️ Tools & Library")
        st.write("""
        - Python
        - Matplotlib
        - Seaborn
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Plotly
        - Datetime
        """)
    with tab3:
        # Struktur folder
        st.write("### 📂 Folder Structure")
        st.write("""
        ```
        ├── data
        │   └── customer_segmentation.csv
            └── dataDescription.txt
        ├── src
        │   ├── main.ipynb
        ├── requirements.txt
        ├── streamlit.py
        └── README.md
        ```
        """)
    
    st.markdown("---")
    # Load Dataset
    df = load_data()
    st.write("### 📊 Dataset Overview")
    st.dataframe(df.head())
    st.write("#### Dataset : [`customer_segmentation.csv`](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering)")
    # Jumlah baris dan kolom
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")
    # Deskripsi dataset
    st.write("### 🗒️ Dataset Description")
    st.write("""
    Information about dataset attributes :\n
    -> Demographics & Family Features :
    - Id: Unique identifier for each individual in the dataset.
    - Year_Birth: The birth year of the individual.
    - Education: The highest level of education attained by the individual.
    - Marital_Status: The marital status of the individual.
    - Income: The annual income of the individual.
    - Kidhome: The number of young children in the household.
    - Teenhome: The number of teenagers in the household.
    - Dt_Customer: The date when the customer was first enrolled or became a part of the company's database.
    - Recency: The number of days since the last purchase or interaction.
    - Complain: Binary indicator (1 or 0) whether the individual has made a complaint.\n
    -> Product Purchase Features :
    - MntWines: The amount spent on wines.
    - MntFruits: The amount spent on fruits.
    - MntMeatProducts: The amount spent on meat products.
    - MntFishProducts: The amount spent on fish products.
    - MntSweetProducts: The amount spent on sweet products.
    - MntGoldProds: The amount spent on gold products.
    - NumDealsPurchases: The number of purchases made with a discount or as part of a deal.
    - NumWebPurchases: The number of purchases made through the company's website.
    - NumCatalogPurchases: The number of purchases made through catalogs.
    - NumStorePurchases: The number of purchases made in physical stores.
    - NumWebVisitsMonth: The number of visits to the company's website in a month.\n
    -> Marketing Response Feature : 
    - AcceptedCmp1 - AcceptedCmp5 : Binary indicator (1 or 0) whether the individual accepted the first - fifth marketing campaign.
    - Z_CostContact: A constant cost associated with contacting a customer.
    - Z_Revenue: A constant revenue associated with a successful campaign response.
    - Response: Binary indicator (1 or 0) whether the individual responded to the marketing campaign.
    """)

    st.markdown("---")
    # Deskripsi dataset
    st.write("### 📈 Statistics Summary")
    df_desc = df.describe()
    st.dataframe(df_desc)

# Page 2 : EDA
elif pages == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    df = load_data()
    st.write("### Dataset Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### What is the Age Distribution of Customers ?")

        year_now = datetime.today().year

        fig1 = px.histogram(year_now - df["Year_Birth"], title="Distribution of Average Customer Age")
        st.plotly_chart(fig1)

        st.write("Those aged 40-50 are most likely to engage in transactions because they are in their productive years and have established jobs, resulting in a decent income, leading to increased spending. As older adults, such as those aged 70-90, transactions decrease, as they are typically retired and need to conserve their spending.")
    with col2:
        st.write("#### What education do most customers have ?")

        education_count = df["Education"].value_counts()

        fig2 = px.pie(values=education_count, names=education_count.index, title="Customer Education")
        st.plotly_chart(fig2)

        st.write("Most customers who make transactions are recent graduates with a score of 50% or half of the data and usually already have sufficient expenses. The fewest are basic students, as they typically have no income at that time and often have to ask their parents for money to make purchases, thus lowering expenses.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### What products do customers buy the most ?")

        product = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum()
        # Harus dibuat dataframe
        products = pd.DataFrame({
            "Product" : ["Wine", "Fruits", "Meat", "Fish", "Sweet", "Gold"],
            "Amount" : product.values
        })

        fig1 = px.bar(products, x="Product", y="Amount", labels={"Product":"Product Types", "Amount":"Frequency"})
        st.plotly_chart(fig1)

        st.write("The highest sales are wine products because most people enjoy wine. Second is meat, because it's a common food source consumed by many people. The lowest sales are fruit products because people don't like fruits, which can taste sweet, sour, or bitter, despite their nutritional value and health benefits.")
    with col2:
        st.write("#### Is there a relationship between customer revenue and total expenses ?")
        # Harus dibuat dataframe
        df["total_purchases"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])

        fig = px.scatter(df, x="Income", y="total_purchases", labels={"Income":"Income", "total_purchases": "Purchases"}, title="Relationship between Income and Total Expenditures")
        st.plotly_chart(fig)

        st.write("In a straight line graph, the higher the income, the higher the expenditure, and if the income is low, the expenditure will also be low because it is necessary to save money.")

# Page 3 : Clustering
elif pages == "Clustering":
    st.title("Customer Clustering")
    st.markdown("---")
    
    # Load dataset
    df = load_data()

    # data encoding
    df_encoded = pd.get_dummies(df, columns=["Education", "Marital_Status"])

    # Scaler
    scaler = StandardScaler()
    array_scaled = scaler.fit_transform(df_encoded)
    # Jadi df scaler
    df_scaled = pd.DataFrame(data=array_scaled, columns=df_encoded.columns)

    # PCA
    pca = PCA()
    array_pca = pca.fit_transform(df_scaled)

    # scree plot of PCA
    st.write("### Scree plot of PCA")
    cumulative_explained_variance = np.cumsum(np.round(pca.explained_variance_ratio_,3)*100)
    st.write(cumulative_explained_variance)
    st.write("##### Take any value until it passes 95 and it will be made into a new, more concise feature.")

    # grafik scree plot PCA
    left, center, right = st.columns([1, 2, 1])
    with center:
        fig, ax = plt.subplots()
        ax.plot(cumulative_explained_variance, marker="o")
        ax.set_title("PCA Scree Plot")
        ax.set_xlabel("Principle Component")
        ax.set_ylabel("Cumulative Explained Variance (%)")
        ax.axhline(y=95, color='r', linestyle="--", label="Threshold 95%")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        st.write("### based on the meeting line is at 29 or more")
    
    st.write("##### The features that are more than 95 from the previous value are 29 values, so there are 29 new features.")
    pca = PCA(n_components=29)
    array_pca = pca.fit_transform(df_scaled)
    # Hitung PC dari cumulative_explained_variance sampai lebih dari 95
    df_after_pca = pd.DataFrame(data=array_pca, columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26','PC27','PC28','PC29']) # sesuaikan dengan komponen
    st.dataframe(df_after_pca.head())
    st.markdown("---")

    # K-Means
    st.subheader("K-Means Clustering")
    # Elbow Method
    st.write("### Elbow Method")

    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(df_after_pca.values)
        inertia.append(kmeans.inertia_)

    left, center, right = st.columns([1, 2, 1])
    with center:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.lineplot(x=range(1, 11), y=inertia, color='#000087', linewidth = 4, ax=ax)
        sns.scatterplot(x=range(1, 11), y=inertia, s=300, color='#800000',  linestyle='--', ax=ax)
        ax.set_title("Elbow Method")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Inertia")
        ax.grid(True)
        st.pyplot(fig)
    st.write("There is a slightly sharp elbow at a value of 2 or 3 which is useful as a reference for determining the number of clusters to be used")
    
    # Silhouette Score
    st.write("### Silhouette Score")
    range_n_clusters = list(range(2,11))

    arr_silhouette_score_euclidean = []
    for i in range_n_clusters:
        kmeans = KMeans(n_clusters=i, random_state=42).fit(df_after_pca)
        preds = kmeans.predict(df_after_pca)
        score_euclidean = silhouette_score(df_after_pca, preds, metric='euclidean')
        arr_silhouette_score_euclidean.append(score_euclidean)
    
    left, center, right = st.columns([1, 2, 1])
    with center:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(x=range(2,11), y=arr_silhouette_score_euclidean, color='#000087', linewidth = 4, ax=ax)
        sns.scatterplot(x=range(2,11), y=arr_silhouette_score_euclidean, s=300, color='#800000',  linestyle='--', ax=ax)
        ax.grid(True)
        st.pyplot(fig)
    st.write("From the results of the silhouette score graph, it can be determined that there can be two or three clusters, as these two values are the highest on the graph. However, for customer segmentation, three clusters will be used, as two clusters would be too simplistic and would only represent active and inactive customers.")

    # Training KMeans
    kmeans = KMeans(n_clusters=3, random_state=0) # 3 Cluster
    kmeans.fit(df_after_pca.values)
    df_after_pca['cluster'] = kmeans.labels_

    # Grafik untuk lihat hasil clustering
    st.write("#### Clustering Result")
    left, center, right = st.columns([1, 2, 1])
    with center:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=df_after_pca, x='PC1', y='PC2', hue='cluster', palette="Set1", ax=ax)
        st.pyplot(fig)
    
    # rata-rata dari tiap PCA dari setiap cluster
    st.write("#### The average of each PCA from each cluster")
    st.write(df_after_pca.groupby('cluster').mean())

    # Gabungkan cluster ke data sebelum PCA
    st.write("#### Average number of original features per cluster")
    df_scaled['cluster'] = df_after_pca['cluster']
    # Lihat rata-rata fitur asli per klaster
    st.write(df_scaled.groupby('cluster').mean())

    st.write("The purpose of applying the results of PCA to the data before PCA is to see the distribution of values with columns that have meaning so that the final result can be known.")
    st.write("""
    Based on the cluster classification, the following categories have been identified:
    - Cluster 0: Customers with many children, moderate income, and moderate product purchases. Cluster 0 is for family customers and can be used for family promotions.
    - Cluster 1: Customers with a high income and high product purchases. Cluster 1 is for premium customers and can be used for exclusive products.
    - Cluster 2: Customers with a low income and few product purchases. Cluster 2 is for frugal customers and can be used for price discounts.
    """)

# Page 4 : Profile
elif pages == "Profile":
    # Judul aplikasi dan huruf italic
    st.title("Arvio Abe Suhendar")
    # Subheader
    st.subheader("Career Shifter | From Network to AI | Designing Intelligent Futures | Ready to Make an Impact in AI | Python Developer | Machine Learning Engineer | Data Scientist")
    st.markdown("---")

    # About me
    st.write("### 📝 About Me")
    st.write("👨‍💻 I'm a tech enthusiast with a strong foundation in Informatics Engineering from Universitas Gunadarma, where I developed solid analytical thinking, programming, and problem-solving skills.")
    st.write("🔧 After graduating, I began my professional journey as a Junior Network Engineer, managing enterprise network services like VPNIP, Astinet, and SIP Trunk on Huawei and Cisco platforms—handling configurations, service activations, and troubleshooting.")
    st.write("🤖 Over time, my curiosity led me to explore the world of Artificial Intelligence & Machine Learning. I've been actively upskilling through bootcamps and self-learning—covering data preprocessing, supervised & unsupervised learning, and deep learning using Python.")
    st.write("🎯 I'm now transitioning my career into AI/ML, combining my network infrastructure background with my growing expertise in data and intelligent systems. I'm particularly interested in how AI can improve systems, automate operations, and drive smarter decision-making.")
    st.write("🤝 Open to collaborations, mentorship, and new opportunities in the AI/ML space.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Education", "Experience", "Skills"])
    with tab1:
        # Pendidikan
        st.write("### 🎓 Education")
        st.write("""
        - **Bachelor of Informatics Engineering**   
        Universitas Gunadarma, 2019 - 2023, GPA 3.82/4.00
            - Built multiple applications (web & desktop) using Java, Python, and PHP in individual and team projects.
            - Built and optimized database systems
            - Learn techniques for solving mathematical problems using programming, numerical integration, and solving equations.
        - **Bootcamp AI/ML**    
        Dibimbing.id Academy, 2025 - Present
            - Mastered core concepts of Python programming including variables, data types, control structures, and functions.
            - Understanding the fundamentals of Artificial Intelligence and Machine Learning, key concepts, and applications.
            - Techniques to clean, transform, and prepare data for analysis, including handling missing data and feature scaling.
        """)
    with tab2:
        # Pengalaman
        st.write("### 💼 Experience")
        st.write("""
        - **Junior Network Engineer**   
        PT. Infomedia Nusantara, 2023 - Present
            - Astinet & VPNIP Service Management (Huawei Routers) : 
                 Handled service activation, disconnection,isolation, modification, and resumption for enterprise clients.
            - Wifi.id Service Provisioning (Cisco & WPgen) :    
                 Performed end-to-end activation and troubleshooting for public Wi-Fi services.
            - SIP Trunk International Access Control :  
                 Managed blocking and unblocking processes for international SIP trunk services to ensure secure and compliant voice connectivity
        """)
    with tab3:
        # Keterampilan
        st.write("### 🛠️ Skills")
        st.write("""
        - **Programming Languages**: Python
        - **Machine Learning**: Scikit-learn, TensorFlow, Keras
        - **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
        - **Database Management**: MySQL, PostgreSQL
        - **Networking**: Huawei Routers, Cisco Routers, WPgen
        - **Tools & Technologies**: Git, Docker, Jupyter Notebook
        - **Soft Skills**: Attention to Detail, Team Collaboration, Adaptability
        """)
    
    st.markdown("---")
    # Kontak
    st.write("### 📞 Contact Information")
    st.write("I'm currently studying and building a career in AI/ML. This project is my practice in building a simple Python application. I want to further develop my skills in this field through existing projects.")
    st.write("Feel free to contact me if you have any questions or suggestions regarding this project.")
    st.write("Email: 4rv10suhendar@gmail.com")
    st.write("LinkedIn: [Arvio Abe Suhendar](https://www.linkedin.com/in/arvio-abe-suhendar/)")
    st.write("Location: Depok, Indonesia")
    st.write("GitHub: [Arvio1378](https://github.com/arvio1378)")

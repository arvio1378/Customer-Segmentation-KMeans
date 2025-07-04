# Customer Segmentation KMeans Clustering

## 📋 Deskripsi
Projek ini adalah projek latihan saya untuk dapat melakukan pengelompokkan pelanggan ke beberapa bagian berdasarkan dengan pola pelanggan menggunakan K-Means Clustering sehingga dapat berguna untuk melakukan strategi untuk bisnis dan pemasaran secara tepat sasaran kepada pelanggan.

## 🚀 Fitur
- Dataset disimpan dalam bentuk CSV
- Model Decision Tree dengan scikit-learn

## 🧠 Tools & Library
- Python 3.X
- Pandas
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn
- Datetime

## 📁 Struktur Folder
- Customer Segmentation K-Means/
  - data
      - customer_segmentation.csv
      - dataDescription.txt
  - src
      - main.ipynb
  - requirements.txt

## 📊 Dataset
- Id: Unique identifier for each individual in the dataset.
- Year_Birth: The birth year of the individual.
- Education: The highest level of education attained by the individual.
- Marital_Status: The marital status of the individual.
- Income: The annual income of the individual.
- Kidhome: The number of young children in the household.
- Teenhome: The number of teenagers in the household.
- Dt_Customer: The date when the customer was first enrolled or became a part of the company's database.
- Recency: The number of days since the last purchase or interaction.
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
- NumWebVisitsMonth: The number of visits to the company's website in a month.
- AcceptedCmp3: Binary indicator (1 or 0) whether the individual accepted the third marketing campaign.
- AcceptedCmp4: Binary indicator (1 or 0) whether the individual accepted the fourth marketing campaign.
- AcceptedCmp5: Binary indicator (1 or 0) whether the individual accepted the fifth marketing campaign.
- AcceptedCmp1: Binary indicator (1 or 0) whether the individual accepted the first marketing campaign.
- AcceptedCmp2: Binary indicator (1 or 0) whether the individual accepted the second marketing campaign.
- Complain: Binary indicator (1 or 0) whether the individual has made a complaint.
- Z_CostContact: A constant cost associated with contacting a customer.
- Z_Revenue: A constant revenue associated with a successful campaign response.
- Response: Binary indicator (1 or 0) whether the individual responded to the marketing campaign.

Source Link : https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

## 🧾 Data Preprocessing :
- Data type change
- Data Duplication
- Missing Values
- Outlier
- Encoding
- Standard scaler
- PCA

## 🖥️ Cara Menjalankan Program
1. Clone repositori
```bash
git clone https://github.com/arvio1378/Customer-Segmentation-KMeans.git
cd Customer-Segmentation-KMeans
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Jalankan program pada folder src/main.ipynb


## 📈 Hasil & Evaluasi
- Elbow Method :
![image](https://github.com/user-attachments/assets/97cafe26-b0bc-488d-b47f-4cc9b9741737)
Terdapat sedikit siku pada nilai 2 dan 3 untuk acuan jumlah clustering yang akan dilakukan
- Silhouette Score :
- ![image](https://github.com/user-attachments/assets/b9dbd66f-5eb1-416d-9a91-53054d931a24)
Silhouette score dapat ditentukan bahwa cluster bisa menjadi 2 atau 3 karena kedua nilai tersebut paling tinggi pada grafik. Tetapi dalam hal customer segmentation akan mengambil cluster sebanyak 3 karena kalau dengan 2 cluster terlalu simpel dan hanya seperti pelanggan tersebut aktif dan tidak aktif saja.
Oleh karena itu untuk jumlah clustering menjadi 3 kelompok. Berdasarkan pembagian cluster telah terbagi diantaranya :
- Cluster 0 : Pelanggan yang memiliki banyak anak, income sedang, dan pembelian produk sedang. Cluster 0 untuk pelanggan berkeluarga dan bisa diberikan untuk promo keluarga.
- Cluster 1 : Pelanggan dengan income yang besar dan pembelian produk yang tinggi. Cluster 1 untuk pelanggan yang premium dan bisa diberikan produk ekslusif.
- Cluster 2 : Pelanggan yang memiliki income rendah, dan pembelian produk sedikit. Cluster 2 untuk pelanggan yang hemat dan bisa diberikan diskon harga.

## 🏗️ Kontribusi
Dapat melakukan kontribusi kepada siapa saja. Bisa bantu untuk :
- Menggunakan data yang lebih besar
- Menambahkan antaramuka di web/streamlit

## 🧑‍💻 Tentang Saya
Saya sedang belajar dan membangun karir di bidang AI/ML. Projek ini adalah latihan saya untuk membangun aplikasi python sederhana. Saya ingin lebih untuk mengembangkan skill saya di bidang ini melalui projek-projek yang ada.
📫 Terhubung dengan saya di:
- Linkedin : https://www.linkedin.com/in/arvio-abe-suhendar/
- Github : https://github.com/arvio1378

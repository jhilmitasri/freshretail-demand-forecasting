# 🛒 Stockout-Aware Product Demand Forecasting using FreshRetailNet‑50K

This project aims to forecast hourly product demand in retail stores while accounting for real-world complexities such as stockouts, promotions, holidays, and weather effects. It uses the [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) dataset — a large-scale, real-world perishable goods sales dataset from 898 stores across 18 cities.

---

## 🌟 Problem Statement

> **How can we accurately forecast hourly product demand in retail stores while accounting for stockouts, promotions, and contextual factors like weather and holidays?**

This project focuses on building a forecasting system that:

* Recovers **true latent demand** during stockout periods
* Models **temporal and contextual patterns** driving demand
* Supports inventory planning and loss prevention strategies

---

## 🧠 Dataset Summary

* **Source**: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
* **Granularity**: Hourly sales for 863 perishable SKUs
* **Stores**: 898
* **Time Window**: 90 days
* **Key Columns**:

  * `hours_sale`: Units sold per hour
  * `hours_stock_status`: 1 = stockout, 0 = in-stock
  * `discount`, `activity_flag`: Promotion metadata
  * `holiday_flag`, `precpt`, `avg_temperature`: Contextual features

---


## 🧭 Industry Inspiration: How Major Retailers Forecast Demand

This project draws inspiration from how retail giants like Walmart, Target, and The Home Depot tackle forecasting at scale. Here’s a curated overview of their strategies:

---

🔹 **1. AI-Powered Demand Forecasting & Stockout Prevention**
- **Walmart** uses AI to tailor inventory based on regional trends and prevent stockouts by reallocating inventory across geographies ([Walmart AI Blog](https://corporate.walmart.com/newsroom/2021/07/27/how-walmart-is-using-ai-to-make-in-store-shopping-better)).
- **Target** built a real-time “Inventory Ledger” that uses AI to double stock coverage by factoring in real-time demand and logistics ([Target AI Ledger – CNBC](https://www.cnbc.com/2021/05/18/how-target-uses-ai-to-track-and-forecast-store-inventory.html)).

---

🔹 **2. Multi-Level Hierarchical Forecasting**
- Walmart Sam’s Club uses machine learning to forecast demand at the item, store, and day level, maintaining coherence across hierarchies ([Walmart Labs Forecasting](https://medium.com/walmartglobaltech/time-series-forecasting-at-scale-at-sams-club-cb13b0ce0b92)).

---

🔹 **3. Real-Time Data Integration & Supplier Collaboration**
- **RetailLink** gives Walmart suppliers real-time access to sales and inventory levels to align restocking ([RetailLink Overview](https://retaillink.wal-mart.com)).
- Walmart pioneered **Vendor Managed Inventory (VMI)** and **Collaborative Planning, Forecasting and Replenishment (CPFR)** in the 1990s ([Walmart VMI Strategy – Harvard](https://hbr.org/2004/11/the-power-of-collaboration)).

---

🔹 **4. Weather-Aware & Event-Based Forecasting**
- Retailers like Walmart and Home Depot integrate **weather APIs** to adjust inventory for expected local surges ([IBM/The Weather Company – Retail Use Case](https://www.ibm.com/blogs/industries/ai-retail-weather-forecasting/)).

---

🔹 **5. Product & Store Segmentation for Scalability**
- Product/store clustering enables better generalization and reduces model count ([Uber’s Time Series Segmentation Approach](https://eng.uber.com/ts-segmentation/)).
- Grouping by **category**, **region**, and **temporal behavior** helps scale forecasting systems.

---

🔹 **6. Cloud-Native Pipelines for Scalability**
- Retailers use **Airflow**, **SageMaker**, and internal MLOps tools to manage training, inference, and deployment ([Walmart’s MLOps Strategy](https://medium.com/walmartglobaltech/mlops-machine-learning-operations-64b4832b17f6)).

---

### ✅ Implications for This Project
- Use **granular features** (store, category, weather) in a global model.
- Apply **hierarchical forecasting** to enforce coherence across product/store levels.
- Prioritize **category-level modeling** to avoid modeling all 50k SKUs individually.
- Integrate **weather/holiday signals** to reflect contextual shifts.
- Plan for **automated retraining workflows** with tools like **Airflow** or **Prefect**.



## 📊 Exploratory Data Analysis (EDA)

### ✔️ Hourly Trends

* Peak demand between **6 AM to 9 AM**
* Stock replenishment typically occurs early in the day

### ✔️ Weekday Behavior

* Highest sales on **Tuesdays** and **Saturdays**
* Minimal activity mid-week

### ✔️ Stockout Analysis

* High stockout frequency overnight and late evening
* Low stockouts in early morning (restock hours)

### ✔️ Promotions & Discounts

* Discounts do not show clear uplift in current subset
* Needs further segmentation or larger-scale comparison

---

## 📈 Sample Visualizations

<p float="left">
  <img src="docs/hourly_sales.png" width="400"/>
    <img src="docs/stockout_rate.png" width="400"/>
  
</p>
<p float="left">
    <img src="docs/weekday_sales.png" width="400"/>
  <!-- <img src="docs/discount_impact.png" width="400"/> -->
</p>

---

## 🧩 Next Steps

* [x] **Latent Demand Recovery**: Estimate true demand during stockouts
* [ ] **Train forecasting models** (LightGBM, LSTM, TFT)
* [ ] **Evaluate performance on multiple time horizons**
* [ ] **Integrate store-level and category-level predictions**
* [ ] **Visualize final model predictions across store-product groups**

---

## 📂 Project Structure

```
.
├── notebooks/
│   ├── eda.ipynb
│   ├── category_store_analysis.ipynb
│   ├── latent_demand_forecasting.ipynb
│   └── product_level_demand_imputation.ipynb
├── data/
│   └── freshretailnet_full.csv / other loaded datasets
├── docs/
│   ├── hourly_sales.png
│   ├── stockout_rate.png
│   └── weekday_sales.png
├── README.md
└── requirements.txt
```

---

## 📌 Dependencies

```bash
pip install -r requirements.txt
```

---

## 🙌 Credits

* Dataset by [Dingdong-Inc](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
* Inspired by operational research in retail demand planning


## Notebooks Overview

🔹 eda.ipynb

Objective: Perform foundational exploratory data analysis on the full FreshRetailNet dataset.
Key Steps:
- Identified unique products (865), stores (898), and ~50,000 product-store combinations.
- Plotted category and store distributions to assess modeling feasibility.
- Helped guide whether to model per-product/store or via aggregation.

🔹 category_store_analysis.ipynb

Objective: Analyze demand distribution across categories and stores to design scalable modeling groups.
Key Steps:
- Identified top 20 third-level categories globally and top 5 per store.
- Found 87 unique categories cover 83.1% of demand.
- Mapped each of these top categories to associated store IDs for selective model training.

🔹 latent_demand_forecasting.ipynb

Objective: Set up demand forecasting pipelines focused on the 87 high-impact third-level categories.
Key Steps:
- Filtered dataset to only include relevant categories.
- Calculated total demand coverage.
- Prepared modeling granularity plan to balance performance with scalability.

🔹 product_level_demand_imputation.ipynb

Objective: Impute missing or latent demand signals at the product level prior to forecasting.
Key Steps:
- Designed a strategy to estimate imputed demand using hours-level stock/sales signals.
- Merged and aligned imputed values with the master dataset.
- Enabled cleaner downstream modeling by reducing signal sparsity.
  
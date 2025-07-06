
# 📈 Future Sales Prediction with Machine Learning

This project uses a **Linear Regression Machine Learning model** to predict future product sales based on advertising costs on **TV, Radio, and Newspaper**. It's built with **Streamlit** to provide a clean and interactive web interface.


## 🚀 Demo

🔗 Live Demo:https://future-sales-prediction-minipro.onrender.com/


## 📊 How It Works

The app takes three inputs:
- 💻 TV Advertising Spend
- 📻 Radio Advertising Spend
- 📰 Newspaper Advertising Spend

Based on these, it predicts the **number of units expected to be sold**, using a trained Linear Regression model.


## 📁 Folder Structure


future_sales_prediction/
|
├── notebooks/
│   └── explore_data.ipynb       ← For data analysis and plots
│
├── src/
│   └── model.py                 ← Model training & prediction code
│
├── main.py                      ← Final script to run prediction
|
|__app.py                        ← Streamlit web app
|
├── requirements.txt             ← List of Python libraries needed
|
└── README.md                    ← Description of the project




## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/future-sales-predictor.git
cd future-sales-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```


## 📦 Dataset

The dataset is sourced from [Aman Kharwal](https://github.com/amankharwal) and contains:

- `TV` – Advertising cost on TV
- `Radio` – Advertising cost on Radio
- `Newspaper` – Advertising cost on Newspaper
- `Sales` – Units sold



## 📈 Model Info

- Algorithm: **Linear Regression**
- Library: `scikit-learn`
- Accuracy: ~90% (R² Score on test set)



## 🙋‍♀️ Author

👩‍💻 **Tejasri Jumbarathi**  
🎓 B.Tech (Information Technology)  

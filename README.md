# 🚚 Logistics Optimizer – Delivery Delay Prediction

A Machine Learning project that predicts whether an order will be delivered **on time or delayed** using logistics and route-related data.  
This system helps logistics teams take proactive actions for potentially delayed deliveries.

---
## 📂 Project Structure

```
logistics-optimizer/
│
├── data/
│   ├── train_data.csv
│   └── new_orders.csv
│
├── models/
│   └── final_model.pkl
│
├── predictions/
│   └── new_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── features.py        # Feature creation logic
│   └── train_utils.py     # Model training helpers
│
├── train_final_model.py   # Training pipeline
├── predict_new.py         # Inference script
└── README.md
```
⚙️ Installation
📥 1) Clone the repository
```bash
git clone https://github.com/mohdzaid145256/logistics-optimizer.git
cd logistics-optimizer
```

🧰 2) Create & Activate Virtual Environment
```bash
python -m venv venv
```


Mac/Linux

```bash
source venv/bin/activate
```

📦 3) Install Dependencies
```bash
pip install -r requirements.txt
```
🚀 Train Model
```bash
python train_final_model.py
```

🔍 Predict New Orders
```bash
python predict_new.py
```

📊 Sample Output
```
order_id  predicted_delay  delay_probability
2001      1                0.56
2002      1                0.57
2003      1                0.56
2004      1                0.56
2005      1                0.57
```
| Metric   | Value   |
| -------- | ------- |
| Accuracy | **92%** |

📌 Feature Importance

1. distance_km          — 47%

2. distance_efficiency  — 26%

3. estimated_travel_time — 19%

4. vehicle_age_norm     — 8%

🔮 Future Enhancements

1. Live traffic & weather APIs

2. FastAPI / Flask deployment

3. Streaming real-time predictions

4. Analytics dashboard UI


👨‍💻 Author

Mohd Zaid

📍 Sikar, Rajasthan

📧 mohdzaid4919@gmail.com

🔗 GitHub: https://github.com/mohdzaid145256



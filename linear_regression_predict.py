import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load mô hình đã lưu
model = joblib.load("linear_regression_model.pkl")

def predict_weight():
    try:
        cm = float(entry_cm.get())
        inch = cm / 2.54  # Chuyển sang inch
        predicted_lbs = model.predict([[inch]])[0]
        predicted_kg = predicted_lbs * 0.4536

        result_var.set(f"Predicted weight:\n{predicted_lbs:.2f} lbs (~{predicted_kg:.2f} kg)")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number for height.")

# Tạo cửa sổ
root = tk.Tk()
root.title("Weight Predictor")
root.geometry("300x200")

# Nhãn & ô nhập
tk.Label(root, text="Enter your height (cm):").pack(pady=5)
entry_cm = tk.Entry(root)
entry_cm.pack()

# Nút Dự đoán
tk.Button(root, text="Predict", command=predict_weight).pack(pady=10)

# Kết quả
result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, fg="blue").pack()

# Chạy app
root.mainloop()

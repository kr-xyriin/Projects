# ==============================
# School Examination Management System
# ==============================

from tkinter import *
from tkinter import messagebox
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Database Connection
# ------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["school_exam_system"]

students_col = db["students"]
marks_col = db["marks"]

# ------------------------------
# Main Window
# ------------------------------
root = Tk()
root.title("School Examination Management System")
root.geometry("400x400")

# ------------------------------
# Login Function
# ------------------------------
def login():
    if username.get() == "admin" and password.get() == "admin":
        messagebox.showinfo("Success", "Login Successful")
        dashboard()
    else:
        messagebox.showerror("Error", "Invalid Login")

# ------------------------------
# Dashboard
# ------------------------------
def dashboard():
    dash = Toplevel()
    dash.title("Dashboard")
    dash.geometry("400x400")

    Button(dash, text="Add Student", width=20, command=add_student).pack(pady=10)
    Button(dash, text="Enter Marks", width=20, command=enter_marks).pack(pady=10)
    Button(dash, text="View Results", width=20, command=view_results).pack(pady=10)

# ------------------------------
# Add Student
# ------------------------------
def add_student():
    win = Toplevel()
    win.title("Add Student")
    win.geometry("300x300")

    Label(win, text="Student ID").pack()
    sid = Entry(win)
    sid.pack()

    Label(win, text="Name").pack()
    name = Entry(win)
    name.pack()

    Label(win, text="Class").pack()
    clas = Entry(win)
    clas.pack()

    def save_student():
        students_col.insert_one({
            "student_id": sid.get(),
            "name": name.get(),
            "class": clas.get()
        })
        messagebox.showinfo("Saved", "Student Added")

    Button(win, text="Save", command=save_student).pack(pady=10)

# ------------------------------
# Enter Marks
# ------------------------------
def enter_marks():
    win = Toplevel()
    win.title("Enter Marks")
    win.geometry("300x300")

    Label(win, text="Student ID").pack()
    sid = Entry(win)
    sid.pack()

    Label(win, text="Marks").pack()
    marks = Entry(win)
    marks.pack()

    def save_marks():
        marks_col.insert_one({
            "student_id": sid.get(),
            "marks": int(marks.get())
        })
        messagebox.showinfo("Saved", "Marks Entered")

    Button(win, text="Save", command=save_marks).pack(pady=10)

# ------------------------------
# View Results + Graph
# ------------------------------
def view_results():
    data = list(marks_col.find({}, {"_id": 0}))
    
    if not data:
        messagebox.showerror("Error", "No Data Found")
        return

    df = pd.DataFrame(data)

    avg = np.mean(df["marks"])
    result_text = f"Average Marks: {avg}"

    messagebox.showinfo("Results", result_text)

    # Plot Graph
    plt.bar(df["student_id"], df["marks"])
    plt.xlabel("Student ID")
    plt.ylabel("Marks")
    plt.title("Student Performance")
    plt.show()

# ------------------------------
# Login UI
# ------------------------------
Label(root, text="LOGIN", font=("Arial", 16)).pack(pady=20)

Label(root, text="Username").pack()
username = Entry(root)
username.pack()

Label(root, text="Password").pack()
password = Entry(root, show="*")
password.pack()

Button(root, text="Login", command=login).pack(pady=20)

root.mainloop()
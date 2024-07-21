from tkinter import *
window = Tk()
window.title("hand_gesture")
window.geometry('650x400')
lbl = Label(window, text="smart mirror show date time and news ")
lbl.grid(column=50, row=30)
def clicked():
    from subprocess import call
    call(["python", "circel.py"])

btn = Button(window, text="circle", command=clicked)
btn.grid(column=70, row=50)

def clicked():
    from subprocess import call
    call(["python", "star.py"])

btn1 = Button(window, text="star", command=clicked)
btn1.grid(column=90, row=70)


def clicked():
    from subprocess import call
    call(["python", "rectangle.py"])

btn1 = Button(window, text="rectangle", command=clicked)
btn1.grid(column=110, row=90)




def clicked():
    from subprocess import call
    call(["python", "triangle.py"])

btn1 = Button(window, text="triangle", command=clicked)
btn1.grid(column=130, row=110)





window.mainloop()

from tkinter import *
window = Tk()
window.title("hand_gesture")
window.geometry('650x400')
lbl = Label(window, text="smart mirror show date time and news ")
lbl.grid(column=50, row=30)
def clicked():
    from subprocess import call
    call(["python", "gui_shape.py"])

btn = Button(window, text="shape", command=clicked)
btn.grid(column=70, row=50)

def clicked():
    from subprocess import call
    call(["python", "cropped_image.py"])

btn1 = Button(window, text="cropped", command=clicked)
btn1.grid(column=90, row=70)


def clicked():
    from subprocess import call
    call(["python", "Webcam_Paint_OpenCV.py"])

btn1 = Button(window, text="draw", command=clicked)
btn1.grid(column=110, row=90)




window.mainloop()

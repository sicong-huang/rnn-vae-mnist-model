import numpy as np
from PIL import ImageTk, Image
import tkinter as tk

import data_utils

if __name__ == '__main__':
    root = tk.Tk()
    canvas = tk.Canvas(root)
    canvas.pack()
    
    root.mainloop()
    _, _, x_test, _ = data_utils.load_data()
    img = x_test[0]
    image=Image.fromarray(img)
    photo = ImageTk.PhotoImage(image)

    canvas.create_image(photo)
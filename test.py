import numpy as np
from PIL import ImageTk, Image
import tkinter as tk

import data_utils

if __name__ == '__main__':
    root = tk.Tk()
    _, _, x_test, _ = data_utils.load_data()
    img = x_test[0]
    img = img * 255

    print(img.shape)
    # height, width = img.shape
    canvas = tk.Canvas(root)
    canvas.pack()

    image = Image.fromarray(img).resize((140, 140), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    root.mainloop()

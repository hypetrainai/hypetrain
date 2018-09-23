import tkinter as tk
root = tk.Tk()

c = tk.Canvas(root, width=600, height=600)

c.pack()

lf = tk.Frame(c, width=100, highlightbackground="green", highlightthickness=1)
lf.pack(side=tk.LEFT, anchor="w")
rf = tk.Frame(c, width=500)
rf.pack(side=tk.LEFT)


def setInput(inp):
  global i
  i = inp
  print(i)

texts = ["Input", "Conv 2D", "Max Pool 2D", "Flatten", "Fully Conn"]
inputs = ["i", "c", "mp", "f", "fc"]

for i in range(0, len(texts)):
  f = tk.Frame(lf, width=100, height=100)
  f.pack(side=tk.TOP)
  def setin(inp):
    return lambda : setInput(inp)
  b = tk.Button(f, text=texts[i], width=10, anchor=tk.W, command=setin(inputs[i]))
  b.pack(side=tk.TOP)

tk.mainloop()










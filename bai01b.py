from sklearn.datasets import make_blobs
import joblib
import tkinter as tk
import numpy as np

np.random.seed(100)
N = 150
knn = joblib.load("knn_simple.pkl")
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('330x330')
        self.resizable(False, False)
        self.title('KNN')   
        self.data = None
        self.labels = None
        self.tao_so_lieu()
        # tao canvas
        self.cvs_data = tk.Canvas(self, width = 302, height = 302, relief = tk.SUNKEN, border = 2, background='white')
        self.cvs_data.bind("<Button-1>", self.xu_ly_mouse)
        # dua canvas len window
        self.cvs_data.place(x = 5, y = 5)
        self.ve_so_lieu()
        
        
    def tao_so_lieu(self):
        centers = [[2, 3], [5, 5], [1, 8]]
        n_classes = len(centers)
        data, labels = make_blobs(N, 
                                centers=np.array(centers),
                                random_state=1)
        min = np.min(data,0)
        min_x = min[0]
        min_y = min[1]

        max = np.max(data,0)
        max_x = max[0]
        max_y = max[1]
        for i in range(0,N):
            x = data[i][0]
            y = data[i][1]
            x_moi = (x - min_x)/(max_x-min_x)*300;
            y_moi = (y - min_y)/(max_y-min_y)*300;
            data[i][0]=x_moi
            data[i][1]=y_moi

        self.data = data;
        self.labels = labels;

    def ve_so_lieu(self):
        nhom_0 = []
        nhom_1 = []
        nhom_2 = []
        for i in range(150):
            if self.labels[i] == 0:
                nhom_0.append([self.data[i,0], self.data[i,1]])
            elif self.labels[i] == 1:
                nhom_1.append([self.data[i,0], self.data[i,1]])
            else:
                nhom_2.append([self.data[i,0], self.data[i,1]])
        nhom_0 = np.array(nhom_0)
        nhom_1 = np.array(nhom_1)
        nhom_2 = np.array(nhom_2)
        
        so_luong = nhom_0.shape[0]
        for i in range(0, so_luong):
            x = nhom_0[i,0]
            y = nhom_0[i,1]
            x1 = x-1
            y1 = y-1
            x2 = x+1
            y2 = y+1
            p = [x1,y1,x2,y2]
            self.cvs_data.create_rectangle(p, fill = 'green',outline = 'green')
            
        so_luong = nhom_1.shape[0]
        for i in range(0, so_luong):
            x = nhom_1[i,0]
            y = nhom_1[i,1]
            x1 = x-1
            y1 = y-1
            x2 = x+1
            y2 = y+1
            p = [x1,y1,x2,y2]
            self.cvs_data.create_rectangle(p, fill = 'red',outline = 'red')
            
        so_luong = nhom_2.shape[0]
        for i in range(0, so_luong):
            x = nhom_2[i,0]
            y = nhom_2[i,1]
            x1 = x-1
            y1 = y-1
            x2 = x+1
            y2 = y+1
            p = [x1,y1,x2,y2]
            self.cvs_data.create_rectangle(p, fill = 'blue',outline = 'blue')
                
    def xu_ly_mouse(self, event):
        x = event.x
        y = event.y
        x1 = x-3
        y1 = y-3
        x2 = x+3
        y2 = y+3
        p = [x1,y1,x2,y2]
        self.cvs_data.create_rectangle(p, fill = 'cyan',outline = 'cyan')
        my_test_data = np.array([[x, y]])
        predicted = knn.predict(my_test_data)
        print('Ket qua nhan dang la', predicted[0])
        text_id = self.cvs_data.create_text((x+10, y),fill ='cyan')
        s =str(predicted[0])
        self.cvs_data.itemconfig(text_id,text = s)
if __name__ == "__main__":
    app = App()
    app.mainloop()

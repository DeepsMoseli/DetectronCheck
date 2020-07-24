import numpy as np


class intersection_over_Union:

    def get_area(self,sq):
        length = (sq[1]-sq[0])
        width = (sq[3]-sq[2])
        area = length*width
        return area
    
    def intersaction(self,sq1,sq2):
        xs = []
        ys = []
        inter = []
        if sq1[1]>=sq2[0] or sq1[0]>=sq2[1] or sq1[3]>=sq2[2] or sq1[2]>=sq2[3]:
            for k in range(2):
                xs.append(sq1[k])
                xs.append(sq2[k])
                ys.append(sq1[2+k])
                ys.append(sq2[2+k])
            xs = np.sort(xs)
            ys = np.sort(ys)
            inter.append(xs[1])
            inter.append(xs[2])
            inter.append(ys[1])
            inter.append(ys[2])
        else:
            inter = [0,0,0,0]
        return self.get_area(inter)
    
    def union(self,sq1,sq2):
        a1 = self.get_area(sq1)
        a2 = self.get_area(sq2)
        inter = self.intersaction(sq1,sq2)
        union_ =  a1 + a2 - inter 
        #print("area1: ", a1)
        #print("area2: ", a2)
        #print("area inter: ", inter)
        return union_ 
    
    def IoU(self,sq1,sq2):
        return (self.intersaction(sq1,sq2)/self.union(sq1,sq2))
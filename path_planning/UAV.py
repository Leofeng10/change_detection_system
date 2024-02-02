######################################################################
# UAV Class
#################################################################
import math
import random

class UAV():
    def __init__(self,x,y,z,ev):
        self.x=x
        self.y=y
        self.z=z
        self.target=[ev.target[0].x,ev.target[0].y,ev.target[0].z]
        self.ev=ev
        self.bt=5000
        self.dir=0
        self.p_bt=10
        self.now_bt=4
        self.cost=0
        self.detect_r=5
        self.ob_space=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.nearest_distance=10
        self.dir_ob=None
        self.p_crash=0
        self.done=False
        self.distance=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])
        self.d_origin=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])
        self.step=0
    def cal(self,num):
        if num==0:
            return -1
        elif num==1:
            return 0
        elif num==2:
            return 1
        else:
            raise NotImplementedError
    def state(self):
        dx=self.target[0]-self.x
        dy=self.target[1]-self.y
        dz=self.target[2]-self.z
        state_grid=[self.x,self.y,self.z,dx,dy,dz,self.target[0],self.target[1],self.target[2],self.d_origin,self.step,self.distance,self.dir,self.p_crash,self.now_bt,self.cost]
        self.ob_space=[]
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i==0 and j==0 and k==0:
                        continue
                    if self.x+i<0 or self.x+i>=self.ev.len or self.y+j<0 or self.y+j>=self.ev.width or self.z+k<0 or self.z+k>=self.ev.h:
                        self.ob_space.append(1) 
                        state_grid.append(1)
                    else:
                        self.ob_space.append(self.ev.map[self.x+i,self.y+j,self.z+k])
                        state_grid.append(self.ev.map[self.x+i,self.y+j,self.z+k])
        return state_grid
    def update(self,action):
        dx,dy,dz=[0,0,0]
        temp=action
        b=3
        wt=0.005
        wc=0.07
        we=0
        c=0.05
        crash=0
        Ddistance=0

        
        dx=self.cal(temp%3)
        temp=int(temp/3)
        dy=self.cal(temp%3)
        temp=int(temp/3)
        dz=self.cal(temp)
        #if drone doesn't move, max penalty
        if dx==0 and dy==0 and dz==0:
            return -1000,False,False, 0, 0, 0, self.x, self.y, self.z
        self.x=self.x+dx
        self.y=self.y+dy
        self.z=self.z+dz
        Ddistance=self.distance-(abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2]))
        self.distance=abs(self.x-self.target[0])+abs(self.y-self.target[1])+abs(self.z-self.target[2])
        self.step+=abs(dx)+abs(dy)+abs(dz)

        flag=1
        if abs(dy)==dy:
            flag=1
        else:
            flag=-1

        if dx*dx+dy*dy!=0:
            self.dir=math.acos(dx/math.sqrt(dx*dx+dy*dy))*flag

        r_ob=0
        for i in range(-2,3):
            for j in range(-2,3):
                if i==0 and j==0:
                    continue
                if self.x+i<0 or self.x+i>=self.ev.len or self.y+j<0 or self.y+j>=self.ev.width or self.z<0 or self.z>=self.ev.h:
                        continue
                if self.ev.map[self.x+i,self.y+j,self.z]==1 and abs(i)+abs(j)<self.nearest_distance:
                    self.nearest_distance=abs(i)+abs(j)
                    flag=1
                    if abs(j)==-j:
                        flag=-1
                    self.dir_ob=math.acos(i/(i*i+j*j))*flag
        if self.nearest_distance>=4 or self.ev.WindField[0]<=self.ev.v0:
            self.p_crash=0
        else:
            self.p_crash=math.exp(-b*self.nearest_distance*self.ev.v0*self.ev.v0/(0.5*math.pow(self.ev.WindField[0]*math.cos(abs(self.ev.WindField[1]-self.dir_ob)-self.ev.v0),2)))

        r_climb = 0
        r_climb=-wc*(abs(self.z-self.target[2]))
        if self.distance>1:
            r_target=2*(self.d_origin/self.distance)*Ddistance
        else:
            r_target=2*(self.d_origin)*Ddistance 

        r=r_climb+r_target-crash*self.p_crash

        if self.x<=0 or self.x>=self.ev.len-1 or self.y<=0 or self.y>=self.ev.width-1 or self.z<=0 or self.z>=self.ev.h-1 or self.ev.map[self.x,self.y,self.z]==1 or random.random()<self.p_crash:
            return r-200,True,2, dx, dy, dz, self.x, self.y, self.z
        if self.distance<=2:
            return r+200,True,1, dx, dy, dz, self.x, self.y, self.z
        if self.step>=self.d_origin+2*self.ev.h:
            return r-20,True,5, dx, dy, dz, self.x, self.y, self.z

        return r,False,4, dx, dy, dz, self.x, self.y, self.z


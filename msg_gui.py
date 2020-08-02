import re,psutil,cv2,webbrowser,requests,pyowm,json,math,wikipedia
from moviepy.editor import *
import smtplib
s=smtplib.SMTP('smtp.gmail.com',587)
s.starttls()
app_id = '1625ba25'
app_key = 'd915d2104f9e5a626dac165336c289b3'
language = 'en'

from tkinter import *
root=Tk()
root.grid()
root.geometry("400x400")

pr=StringVar()
ni=Entry(root,textvariable=pr).place(x=130,y=50)
ch=StringVar()
bi=Entry(root,textvariable=ch).place(x=130,y=100)
head1=Label(root,text="Enter Mobile No.",fg='black').place(x=10,y=50)
head2=Label(root,text="Enter Message",fg='black').place(x=10,y=100)

def send_msg():
          #mob_no=input(" enter  mobile number")
          url = "https://www.fast2sms.com/dev/bulk"
          #text=input("enter message:")
          payload = f"sender_id=FSTSMS&message={ch.get()}&language=english&route=p&numbers={pr.get()}"
          headers = {
          'authorization': "7aEPsfe6YBxVS2AFR5zbZ08pD1MyLWu3HI9tJhnNkKTv4orcmX9l56RQr10eStLNMizaoyO8gWcnjPwT",
          'Content-Type': "application/x-www-form-urlencoded",
          'Cache-Control': "no-cache",
          }
          response = requests.request("POST", url, data=payload, headers=headers)
          #print(response.text)
          Label(root,text=response.text).place(x=20,y=150)

Button(root,text="send ",command=send_msg).place(x=30,y=200)

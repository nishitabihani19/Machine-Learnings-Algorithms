from tkinter import *
import re,psutil,cv2,webbrowser,requests,json,math,wikipedia
import smtplib
s=smtplib.SMTP('smtp.gmail.com',587)
s.starttls()

root=Tk()
root.grid()
root.geometry("500x500")
p=StringVar()
n=StringVar()
resp=StringVar()
msg=StringVar()
head1=Label(root,text="Enter Mail_id",fg='black').place(x=10,y=100)
head2=Label(root,text="Enter password",fg='black').place(x=10,y=140)
en1=Entry(root,textvariable=p).place(x=200,y=100)
en2=Entry(root,textvariable=n).place(x=200,y=140)
head3=Label(root,text="Enter Other mail",fg='black').place(x=10,y=180)
head4=Label(root,text="Enter Message",fg='black').place(x=10,y=220)
en3=Entry(root,textvariable=resp).place(x=200,y=180)
en4=Entry(root,textvariable=msg).place(x=200,y=220)
def send_mail():
          #username=input("enter your mail id:")
          #password=input("enter password:")
          print(p.get())
          print(n.get())
          print(resp.get())
          print(msg.get())
          
          s.login(p.get(),n.get())
          #resp=input("enter other mail:")
          #msg=input("enter text that u want to send")
          s.sendmail(p.get(),resp.get(),msg.get())
          #print("mail send success")
          head1=Label(root,text="mail send success",fg='black').place(x=100,y=240)

          s.quit()
          #print("logout")
          head2=Label(root,text="LOgout",fg='black').place(x=100,y=280)
butn=Button(root,text="submit",command=send_mail) .place(x=50,y=300)
root.mainloop()

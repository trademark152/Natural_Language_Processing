import sys
import os

# write out error
# sys.stderr.write('This is stderr text\n')
#
# sys.stderr.flush()
#
# sys.stdout.write('THis is a stdout text\n')

# print out all arguments you parsed through python, for now it is just the file name
# print(sys.argv)

''' you can go to windows, "Shift+Right-click" to open windows powershell and run command "python sysTutorial.py "look at that" to get a string list'''

''' to manipulate data coming in and run scripts'''
# if len(sys.argv) > 1:
#     print(sys.argv[1])

# if len(sys.argv) > 1:
#     print(float(sys.argv[1])+3)


# def main(arg):
#     print(arg)
#
# main(sys.argv[1])

folderPath = r'C:\\Users\\trade\\Dropbox\\Classes\\Spring 2019\\CSCI544\\HW\\PA\\PA1'

def listDir(dir):
    fileNames = os.listdir(dir)
    for fileName in fileNames:
        print('File Name: ' + fileName)
        print('Folder Path: ' + os.path.abspath(os.path.join(dir, fileName)),sep ='\n')

if __name__ == '__main__':
    listDir(folderPath)
